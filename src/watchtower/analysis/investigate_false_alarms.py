"""
WATCHTOWER - False Alarm Investigation
Investigates high-probability normal samples causing high FPR.

Problem: Bimodal distribution in normal samples (some normal windows predicted with very high probability)
Goal: Identify what's causing these false alarms to improve model or features

Author: Himanshu's WATCHTOWER Project
Date: 2026-02-06
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import GroupKFold
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FalseAlarmInvestigator:
    """
    Investigates high-probability normal samples (false alarms).

    The probability distribution showed bimodal behavior in normal samples:
    - Most normals have low probability (median=0.040)
    - Some normals have very high probability (up to 0.999)

    This class analyzes WHAT makes those high-probability normals different.
    """

    def __init__(self, config_path: str = 'configs/xgboost_config.yaml'):
        self.config = self._load_config(config_path)
        self.X = None
        self.y = None
        self.groups = None
        self.feature_names = None
        self.df_features = None

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path) as f:
            return yaml.safe_load(f)

    def load_data(self):
        """Load data and feature names."""
        logger.info("Loading data...")

        self.X = np.load(self.config['data']['X_path'])
        self.y = np.load(self.config['data']['y_path'])

        self.df_features = pd.read_parquet(self.config['data']['features_table_path'])
        self.groups = self.df_features[self.config['data']['group_column']].values

        # Get feature names from the parquet file (exclude metadata columns)
        exclude_cols = ['ts_start_ns', 'scenario_id', 'weak_label', 'pci_mode']
        self.feature_names = [c for c in self.df_features.columns if c not in exclude_cols]

        logger.info(f"Loaded: X={self.X.shape}, y={self.y.shape}")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Unique scenarios: {np.unique(self.groups)}")

        return self

    def run_cv_and_collect_predictions(self) -> pd.DataFrame:
        """
        Run CV and collect predictions with original feature values.

        Returns DataFrame with:
        - All original features
        - y_true (actual label)
        - y_prob (predicted probability)
        - fold (which CV fold)
        - scenario_id (for grouping)
        """
        logger.info("\nRunning GroupKFold CV to collect predictions...")

        n_splits = self.config['split']['n_splits']
        gkf = GroupKFold(n_splits=n_splits)

        all_rows = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(self.X, self.y, self.groups), 1):
            logger.info(f"Processing fold {fold}/{n_splits}...")

            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            # Compute scale_pos_weight
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            scale_pos_weight = n_neg / n_pos

            # Create model
            xgb_config = self.config['xgboost'].copy()
            if xgb_config['scale_pos_weight'] == 'auto':
                xgb_config['scale_pos_weight'] = scale_pos_weight

            model = xgb.XGBClassifier(**xgb_config)
            model.fit(X_train, y_train, verbose=False)

            # Get predictions
            y_prob = model.predict_proba(X_test)[:, 1]

            # Collect data for analysis
            for i, idx in enumerate(test_idx):
                row = {
                    'idx': idx,
                    'fold': fold,
                    'scenario_id': self.groups[idx],
                    'y_true': int(y_test[i]),
                    'y_prob': float(y_prob[i])
                }
                # Add all features
                for j, fname in enumerate(self.feature_names):
                    row[fname] = float(self.X[idx, j])

                all_rows.append(row)

        df = pd.DataFrame(all_rows)
        logger.info(f"Collected {len(df)} samples with predictions")

        return df

    def analyze_high_prob_normals(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Analyze normal samples that are predicted with high probability.

        These are the FALSE ALARMS at the given threshold.

        Args:
            df: DataFrame with predictions
            threshold: Probability threshold above which normals are "false alarms"

        Returns:
            DataFrame of high-probability normal samples
        """
        # Filter: Normal samples (y_true=0) with high probability (>threshold)
        high_prob_normals = df[(df['y_true'] == 0) & (df['y_prob'] >= threshold)].copy()
        low_prob_normals = df[(df['y_true'] == 0) & (df['y_prob'] < threshold)].copy()

        logger.info("\n" + "="*80)
        logger.info(f"ANALYSIS OF HIGH-PROBABILITY NORMAL SAMPLES (prob >= {threshold})")
        logger.info("="*80)

        n_normal = len(df[df['y_true'] == 0])
        n_high_prob = len(high_prob_normals)

        logger.info(f"\nTotal Normal Samples: {n_normal}")
        logger.info(f"High-Prob Normals (prob >= {threshold}): {n_high_prob} ({n_high_prob/n_normal*100:.1f}%)")
        logger.info(f"Low-Prob Normals (prob < {threshold}): {len(low_prob_normals)} ({len(low_prob_normals)/n_normal*100:.1f}%)")

        if n_high_prob == 0:
            logger.info("No high-probability normal samples found!")
            return high_prob_normals

        # Analyze by scenario
        logger.info("\n📊 SCENARIO DISTRIBUTION:")
        for scenario in df['scenario_id'].unique():
            scenario_df = df[(df['scenario_id'] == scenario) & (df['y_true'] == 0)]
            high_prob_scenario = high_prob_normals[high_prob_normals['scenario_id'] == scenario]

            total = len(scenario_df)
            high = len(high_prob_scenario)
            pct = high/total*100 if total > 0 else 0

            logger.info(f"  Scenario {scenario}: {high}/{total} high-prob normals ({pct:.1f}%)")

        # Compare feature distributions
        logger.info("\n📊 FEATURE DIFFERENCES (High-Prob Normals vs Low-Prob Normals):")
        logger.info("-"*80)

        feature_diffs = []
        for fname in self.feature_names:
            high_mean = high_prob_normals[fname].mean()
            low_mean = low_prob_normals[fname].mean()
            diff = high_mean - low_mean

            high_std = high_prob_normals[fname].std()
            low_std = low_prob_normals[fname].std()

            # Effect size (Cohen's d approximation)
            pooled_std = np.sqrt((high_std**2 + low_std**2) / 2)
            effect_size = diff / pooled_std if pooled_std > 0 else 0

            feature_diffs.append({
                'feature': fname,
                'high_prob_mean': high_mean,
                'low_prob_mean': low_mean,
                'diff': diff,
                'effect_size': abs(effect_size)
            })

        diff_df = pd.DataFrame(feature_diffs).sort_values('effect_size', ascending=False)

        logger.info("\nTop 10 Features with LARGEST Differences:")
        for _, row in diff_df.head(10).iterrows():
            direction = "↑" if row['diff'] > 0 else "↓"
            logger.info(f"  {row['feature']:25s}: {row['high_prob_mean']:+8.3f} vs {row['low_prob_mean']:+8.3f} "
                       f"(diff={row['diff']:+7.3f}, effect={row['effect_size']:.2f}) {direction}")

        return high_prob_normals, low_prob_normals, diff_df

    def compare_with_anomalies(self, df: pd.DataFrame, high_prob_normals: pd.DataFrame) -> pd.DataFrame:
        """
        Compare high-prob normals with actual anomalies.

        Are these "false alarms" actually similar to real anomalies?
        (Could indicate mislabeling or edge cases)
        """
        anomalies = df[df['y_true'] == 1].copy()

        logger.info("\n" + "="*80)
        logger.info("COMPARISON: HIGH-PROB NORMALS vs ACTUAL ANOMALIES")
        logger.info("="*80)

        logger.info(f"\nHigh-Prob Normals: n={len(high_prob_normals)}, Mean Prob={high_prob_normals['y_prob'].mean():.3f}")
        logger.info(f"Actual Anomalies:  n={len(anomalies)}, Mean Prob={anomalies['y_prob'].mean():.3f}")

        # Compare key features
        logger.info("\n📊 FEATURE COMPARISON (High-Prob Normals vs Anomalies):")
        logger.info("-"*80)

        comparisons = []
        for fname in self.feature_names:
            hp_mean = high_prob_normals[fname].mean()
            anom_mean = anomalies[fname].mean()
            diff = abs(hp_mean - anom_mean)

            comparisons.append({
                'feature': fname,
                'high_prob_normal_mean': hp_mean,
                'anomaly_mean': anom_mean,
                'absolute_diff': diff
            })

        comp_df = pd.DataFrame(comparisons).sort_values('absolute_diff', ascending=True)

        logger.info("\nFeatures where High-Prob Normals are MOST SIMILAR to Anomalies:")
        for _, row in comp_df.head(10).iterrows():
            logger.info(f"  {row['feature']:25s}: HP-Normal={row['high_prob_normal_mean']:+8.3f}, "
                       f"Anomaly={row['anomaly_mean']:+8.3f} (diff={row['absolute_diff']:.3f})")

        logger.info("\nFeatures where High-Prob Normals are MOST DIFFERENT from Anomalies:")
        for _, row in comp_df.tail(5).iterrows():
            logger.info(f"  {row['feature']:25s}: HP-Normal={row['high_prob_normal_mean']:+8.3f}, "
                       f"Anomaly={row['anomaly_mean']:+8.3f} (diff={row['absolute_diff']:.3f})")

        return comp_df

    def plot_feature_distributions(
        self,
        df: pd.DataFrame,
        high_prob_normals: pd.DataFrame,
        low_prob_normals: pd.DataFrame,
        top_features: List[str],
        save_path: str = None
    ):
        """
        Plot feature distributions for the most discriminative features.

        Shows:
        - Low-prob normals (correctly classified)
        - High-prob normals (false alarms)
        - Anomalies (for comparison)
        """
        anomalies = df[df['y_true'] == 1]

        n_features = len(top_features)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, fname in enumerate(top_features[:6]):
            ax = axes[i]

            ax.hist(low_prob_normals[fname], bins=30, alpha=0.5, color='green',
                   label=f'Low-Prob Normal (n={len(low_prob_normals)})', density=True)
            ax.hist(high_prob_normals[fname], bins=30, alpha=0.5, color='orange',
                   label=f'High-Prob Normal (n={len(high_prob_normals)})', density=True)
            ax.hist(anomalies[fname], bins=30, alpha=0.5, color='red',
                   label=f'Anomaly (n={len(anomalies)})', density=True)

            ax.set_xlabel(fname, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{fname}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.suptitle('Feature Distributions: Low-Prob Normals vs High-Prob Normals (False Alarms) vs Anomalies',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✅ Saved feature distribution plot: {save_path}")

        plt.close()

    def generate_report(self, save_path: str = 'reports/false_alarm_investigation.txt'):
        """
        Run complete investigation and save report.
        """
        logger.info("\n" + "🔍"*40)
        logger.info("FALSE ALARM INVESTIGATION REPORT")
        logger.info("🔍"*40)

        # Load data
        self.load_data()

        # Get CV predictions
        df = self.run_cv_and_collect_predictions()

        # Analyze at different thresholds
        for threshold in [0.3, 0.5, 0.7]:
            high_prob, low_prob, diff_df = self.analyze_high_prob_normals(df, threshold=threshold)

            if len(high_prob) > 0:
                self.compare_with_anomalies(df, high_prob)

                # Plot for threshold 0.5
                if threshold == 0.5:
                    top_features = diff_df.head(6)['feature'].tolist()
                    Path('reports/plots').mkdir(parents=True, exist_ok=True)
                    self.plot_feature_distributions(
                        df, high_prob, low_prob, top_features,
                        save_path='reports/plots/false_alarm_features.png'
                    )

        # Key insights
        logger.info("\n" + "="*80)
        logger.info("KEY INSIGHTS & RECOMMENDATIONS")
        logger.info("="*80)

        # Check for scenario bias
        normal_samples = df[df['y_true'] == 0]
        high_prob_50 = normal_samples[normal_samples['y_prob'] >= 0.5]

        scenario_stats = []
        for scenario in df['scenario_id'].unique():
            sc_normals = normal_samples[normal_samples['scenario_id'] == scenario]
            sc_high = high_prob_50[high_prob_50['scenario_id'] == scenario]
            rate = len(sc_high) / len(sc_normals) * 100 if len(sc_normals) > 0 else 0
            scenario_stats.append({'scenario': scenario, 'false_alarm_rate': rate, 'n': len(sc_normals)})

        sc_df = pd.DataFrame(scenario_stats).sort_values('false_alarm_rate', ascending=False)

        logger.info("\n📊 FALSE ALARM RATE BY SCENARIO:")
        for _, row in sc_df.iterrows():
            logger.info(f"  Scenario {row['scenario']}: {row['false_alarm_rate']:.1f}% false alarms (n={row['n']})")

        # Check if false alarms are concentrated in specific scenarios
        max_far = sc_df['false_alarm_rate'].max()
        min_far = sc_df['false_alarm_rate'].min()

        if max_far - min_far > 30:
            logger.info("\n⚠️  FINDING: False alarms are concentrated in specific scenarios!")
            logger.info("   This suggests scenario-specific characteristics causing false alarms.")
            logger.info("   Consider: Adding scenario-specific features or separate models.")

        # Overall recommendations
        logger.info("\n" + "="*80)
        logger.info("RECOMMENDED ACTIONS:")
        logger.info("="*80)

        if len(high_prob_50) / len(normal_samples) > 0.3:
            logger.info("1. ⚠️  HIGH FALSE ALARM RATE - Consider:")
            logger.info("   a) Investigate potential mislabeling in training data")
            logger.info("   b) Review labeling criteria for edge cases")
            logger.info("   c) Use higher threshold (0.50 or higher)")
        else:
            logger.info("1. ✅ False alarm rate is manageable")

        logger.info("\n2. Feature Engineering Suggestions:")
        logger.info("   - Add temporal features (changes over time)")
        logger.info("   - Add scenario-aware features")
        logger.info("   - Consider feature interactions")

        logger.info("\n3. Model Improvements:")
        logger.info("   - Try isotonic calibration")
        logger.info("   - Consider ensemble with different algorithms")
        logger.info("   - Add confidence estimation")

        logger.info("="*80)

        return df


def main():
    """Run the investigation."""
    investigator = FalseAlarmInvestigator()
    df = investigator.generate_report()

    # Save the full prediction data for further analysis
    df.to_parquet('reports/cv_predictions_with_features.parquet', index=False)
    logger.info("\n✅ Saved full CV predictions to: reports/cv_predictions_with_features.parquet")

    return df


if __name__ == "__main__":
    main()
