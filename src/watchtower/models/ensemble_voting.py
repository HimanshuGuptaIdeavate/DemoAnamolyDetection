"""
WATCHTOWER - Ensemble Parallel Voting
Combines XGBoost (M0) and LSTM (M1) predictions via weighted average.

Formula: prob_final = w * prob_xgb + (1-w) * prob_lstm
Weight w is optimized using CV predictions from both models.

Author: Himanshu's WATCHTOWER Project
Date: 2026-02-13
"""

import numpy as np
import pandas as pd
import yaml
import json
import mlflow
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import logging

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from watchtower.models.threshold_selection import (
    tune_threshold_from_cv,
    WINDOWS_PER_HOUR,
)

logger = logging.getLogger(__name__)


class EnsembleVoting:
    """
    Parallel Voting Ensemble: XGBoost + LSTM.

    Loads CV predictions from both models (same folds, same samples),
    finds optimal combination weight, and evaluates ensemble performance.
    """

    def __init__(self, config_path: str = 'configs/ensemble_config.yaml'):
        self.config = self._load_config(config_path)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.optimal_weight = 0.5
        self.ensemble_proba = None

        self._create_directories()
        logger.info("Ensemble Voting initialized")

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _create_directories(self):
        dirs = [
            self.config['artifacts']['report_dir'],
            self.config['artifacts']['plots_dir'],
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def load_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load CV predictions from both models.

        Returns:
            y_true: (871,) ground truth labels
            y_proba_xgb: (871,) XGBoost probabilities
            y_proba_lstm: (871,) LSTM probabilities
        """
        logger.info("="*80)
        logger.info("LOADING CV PREDICTIONS")
        logger.info("="*80)

        preds = self.config['predictions']

        y_true_xgb = np.load(preds['xgb_y_true_path'])
        y_proba_xgb = np.load(preds['xgb_y_proba_path'])
        y_true_lstm = np.load(preds['lstm_y_true_path'])
        y_proba_lstm = np.load(preds['lstm_y_proba_path'])

        logger.info(f"XGBoost:  y_true={y_true_xgb.shape}, y_proba={y_proba_xgb.shape}")
        logger.info(f"LSTM:     y_true={y_true_lstm.shape}, y_proba={y_proba_lstm.shape}")

        # Validate alignment
        if not np.array_equal(y_true_xgb, y_true_lstm):
            n_diff = np.sum(y_true_xgb != y_true_lstm)
            logger.error(f"LABEL MISMATCH: {n_diff} labels differ between XGBoost and LSTM!")
            raise ValueError("XGBoost and LSTM labels don't match. Ensure same GroupKFold splits.")

        logger.info(f"Label alignment: VERIFIED ({len(y_true_xgb)} samples match)")

        # Quick stats
        logger.info(f"\nXGBoost proba:  mean={y_proba_xgb.mean():.3f}, std={y_proba_xgb.std():.3f}")
        logger.info(f"LSTM proba:     mean={y_proba_lstm.mean():.3f}, std={y_proba_lstm.std():.3f}")

        return y_true_xgb, y_proba_xgb, y_proba_lstm

    def find_optimal_weight(
        self,
        y_true: np.ndarray,
        y_proba_xgb: np.ndarray,
        y_proba_lstm: np.ndarray
    ) -> Dict:
        """
        Find optimal weight w that maximizes F2-Score.

        Formula: prob_ensemble = w * prob_xgb + (1-w) * prob_lstm

        Args:
            y_true: Ground truth labels
            y_proba_xgb: XGBoost probabilities
            y_proba_lstm: LSTM probabilities

        Returns:
            Dict with optimal weight and metrics at all tested weights
        """
        logger.info("\n" + "="*80)
        logger.info("WEIGHT OPTIMIZATION")
        logger.info("="*80)
        logger.info("Formula: prob_ensemble = w * prob_xgb + (1-w) * prob_lstm")

        wt_config = self.config['weight_tuning']
        weights = np.arange(wt_config['w_min'], wt_config['w_max'] + wt_config['w_step'], wt_config['w_step'])
        optimize_metric = wt_config['optimize_metric']

        results = []
        best_score = -1
        best_weight = 0.5

        for w in weights:
            # Combine predictions
            prob_ensemble = w * y_proba_xgb + (1 - w) * y_proba_lstm

            # Compute metrics at a range of thresholds and pick best F2
            best_f2_for_w = 0
            best_thresh_for_w = 0.5

            for thresh in np.arange(0.05, 0.95, 0.05):
                y_pred = (prob_ensemble >= thresh).astype(int)
                f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
                if f2 > best_f2_for_w:
                    best_f2_for_w = f2
                    best_thresh_for_w = thresh

            # Get full metrics at best threshold for this weight
            y_pred_best = (prob_ensemble >= best_thresh_for_w).astype(int)

            roc_auc = roc_auc_score(y_true, prob_ensemble)
            pr_auc = average_precision_score(y_true, prob_ensemble)
            brier = brier_score_loss(y_true, prob_ensemble)
            precision = precision_score(y_true, y_pred_best, zero_division=0)
            recall = recall_score(y_true, y_pred_best, zero_division=0)
            f1 = f1_score(y_true, y_pred_best, zero_division=0)
            f2 = best_f2_for_w

            cm = confusion_matrix(y_true, y_pred_best)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0

            n_normal = fp + tn
            fpr = fp / n_normal if n_normal > 0 else 0
            normal_rate = n_normal / len(y_true) * WINDOWS_PER_HOUR
            far_per_hour = fpr * normal_rate

            results.append({
                'weight_xgb': round(w, 2),
                'weight_lstm': round(1 - w, 2),
                'best_threshold': best_thresh_for_w,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'brier': brier,
                'f2_score': f2,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'fpr': fpr,
                'far_per_hour': far_per_hour,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
            })

            if f2 > best_score:
                best_score = f2
                best_weight = w

        df = pd.DataFrame(results)

        self.optimal_weight = best_weight
        self.weight_results = df

        # Log top results
        logger.info(f"\nWeight Search Results (sorted by F2-Score):")
        logger.info("-"*80)

        top_10 = df.nlargest(10, 'f2_score')
        for _, row in top_10.iterrows():
            logger.info(
                f"  w={row['weight_xgb']:.2f} XGB + {row['weight_lstm']:.2f} LSTM | "
                f"F2={row['f2_score']:.4f} | ROC-AUC={row['roc_auc']:.4f} | "
                f"Recall={row['recall']:.3f} | Precision={row['precision']:.3f} | "
                f"FPR={row['fpr']:.3f} | FAR={row['far_per_hour']:.1f}/hr"
            )

        # Best result
        best_row = df[df['weight_xgb'] == round(best_weight, 2)].iloc[0]

        logger.info(f"\n{'='*80}")
        logger.info(f"OPTIMAL ENSEMBLE WEIGHT")
        logger.info(f"{'='*80}")
        logger.info(f"  w = {best_weight:.2f} XGBoost + {1-best_weight:.2f} LSTM")
        logger.info(f"  Best Threshold: {best_row['best_threshold']:.2f}")
        logger.info(f"  F2-Score:  {best_row['f2_score']:.4f}")
        logger.info(f"  ROC-AUC:   {best_row['roc_auc']:.4f}")
        logger.info(f"  PR-AUC:    {best_row['pr_auc']:.4f}")
        logger.info(f"  Brier:     {best_row['brier']:.4f}")
        logger.info(f"  Recall:    {best_row['recall']:.4f}")
        logger.info(f"  Precision: {best_row['precision']:.4f}")
        logger.info(f"  FPR:       {best_row['fpr']:.4f}")
        logger.info(f"  FAR:       {best_row['far_per_hour']:.1f}/hour")
        logger.info(f"  Confusion: TP={best_row['tp']}, FP={best_row['fp']}, FN={best_row['fn']}, TN={best_row['tn']}")

        return {
            'optimal_weight': best_weight,
            'results_df': df,
            'best_row': best_row.to_dict()
        }

    def compare_models(
        self,
        y_true: np.ndarray,
        y_proba_xgb: np.ndarray,
        y_proba_lstm: np.ndarray
    ):
        """
        Compare XGBoost, LSTM, and Ensemble side-by-side.
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON: XGBoost vs LSTM vs Ensemble")
        logger.info("="*80)

        prob_ensemble = self.optimal_weight * y_proba_xgb + (1 - self.optimal_weight) * y_proba_lstm

        models = {
            'XGBoost (M0)': y_proba_xgb,
            'LSTM (M1)': y_proba_lstm,
            f'Ensemble (w={self.optimal_weight:.2f})': prob_ensemble,
        }

        comparison = []
        for name, proba in models.items():
            # Find best F2 threshold
            best_f2, best_thresh = 0, 0.5
            for thresh in np.arange(0.05, 0.95, 0.05):
                y_pred = (proba >= thresh).astype(int)
                f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
                if f2 > best_f2:
                    best_f2 = f2
                    best_thresh = thresh

            y_pred = (proba >= best_thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0

            n_normal = fp + tn
            fpr = fp / n_normal if n_normal > 0 else 0
            far = fpr * (n_normal / len(y_true) * WINDOWS_PER_HOUR)

            comparison.append({
                'Model': name,
                'ROC-AUC': roc_auc_score(y_true, proba),
                'PR-AUC': average_precision_score(y_true, proba),
                'Brier': brier_score_loss(y_true, proba),
                'F2-Score': best_f2,
                'Threshold': best_thresh,
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'FPR': fpr,
                'FAR/hr': far,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn,
            })

        comp_df = pd.DataFrame(comparison)

        # Print comparison table
        logger.info(f"\n{'Model':<28} {'ROC-AUC':>8} {'PR-AUC':>8} {'Brier':>7} {'F2':>7} {'Thresh':>7} {'Recall':>7} {'Prec':>7} {'FPR':>7} {'FAR/hr':>8}")
        logger.info("-"*105)
        for _, row in comp_df.iterrows():
            logger.info(
                f"{row['Model']:<28} {row['ROC-AUC']:>8.4f} {row['PR-AUC']:>8.4f} "
                f"{row['Brier']:>7.4f} {row['F2-Score']:>7.4f} {row['Threshold']:>7.2f} "
                f"{row['Recall']:>7.4f} {row['Precision']:>7.4f} {row['FPR']:>7.4f} {row['FAR/hr']:>8.1f}"
            )

        # Check if ensemble improves
        logger.info(f"\n{'='*80}")
        logger.info("ENSEMBLE BENEFIT ANALYSIS")
        logger.info(f"{'='*80}")

        xgb_row = comp_df[comp_df['Model'].str.contains('XGBoost')].iloc[0]
        lstm_row = comp_df[comp_df['Model'].str.contains('LSTM')].iloc[0]
        ens_row = comp_df[comp_df['Model'].str.contains('Ensemble')].iloc[0]

        metrics_to_check = ['ROC-AUC', 'PR-AUC', 'F2-Score', 'Recall']
        for metric in metrics_to_check:
            best_individual = max(xgb_row[metric], lstm_row[metric])
            ens_val = ens_row[metric]
            diff = ens_val - best_individual
            direction = "IMPROVED" if diff > 0 else "NO IMPROVEMENT" if diff == 0 else "DECREASED"
            logger.info(f"  {metric}: XGB={xgb_row[metric]:.4f}, LSTM={lstm_row[metric]:.4f}, Ensemble={ens_val:.4f} -> {direction} ({diff:+.4f})")

        # FAR comparison (lower is better)
        logger.info(f"\n  FAR/hour: XGB={xgb_row['FAR/hr']:.1f}, LSTM={lstm_row['FAR/hr']:.1f}, Ensemble={ens_row['FAR/hr']:.1f}")
        if ens_row['FAR/hr'] < min(xgb_row['FAR/hr'], lstm_row['FAR/hr']):
            logger.info(f"  -> IMPROVED (lower false alarms)")
        else:
            logger.info(f"  -> No FAR improvement")

        logger.info(f"{'='*80}")

        return comp_df

    def plot_weight_optimization(self, save_path: str = None):
        """Plot F2-Score and ROC-AUC vs weight."""
        df = self.weight_results

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: F2-Score vs Weight
        axes[0].plot(df['weight_xgb'], df['f2_score'], 'b-', linewidth=2, marker='o', markersize=3)
        axes[0].axvline(x=self.optimal_weight, color='red', linestyle='--',
                       label=f'Optimal: w={self.optimal_weight:.2f}')
        axes[0].set_xlabel('XGBoost Weight (w)', fontsize=12)
        axes[0].set_ylabel('F2-Score', fontsize=12)
        axes[0].set_title('F2-Score vs Ensemble Weight', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Plot 2: ROC-AUC vs Weight
        axes[1].plot(df['weight_xgb'], df['roc_auc'], 'g-', linewidth=2, marker='o', markersize=3)
        axes[1].axvline(x=self.optimal_weight, color='red', linestyle='--',
                       label=f'Optimal: w={self.optimal_weight:.2f}')
        axes[1].set_xlabel('XGBoost Weight (w)', fontsize=12)
        axes[1].set_ylabel('ROC-AUC', fontsize=12)
        axes[1].set_title('ROC-AUC vs Ensemble Weight', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # Plot 3: FAR vs Weight
        axes[2].plot(df['weight_xgb'], df['far_per_hour'], 'r-', linewidth=2, marker='o', markersize=3)
        axes[2].axvline(x=self.optimal_weight, color='red', linestyle='--',
                       label=f'Optimal: w={self.optimal_weight:.2f}')
        axes[2].set_xlabel('XGBoost Weight (w)', fontsize=12)
        axes[2].set_ylabel('False Alarms / Hour', fontsize=12)
        axes[2].set_title('FAR vs Ensemble Weight', fontsize=13, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.suptitle(f'Ensemble Weight Optimization (w=0: LSTM only, w=1: XGBoost only)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved weight optimization plot: {save_path}")
        plt.close()

    def plot_roc_comparison(
        self, y_true, y_proba_xgb, y_proba_lstm, save_path: str = None
    ):
        """Plot ROC curves for all three models."""
        prob_ensemble = self.optimal_weight * y_proba_xgb + (1 - self.optimal_weight) * y_proba_lstm

        plt.figure(figsize=(8, 6))

        for name, proba, color in [
            ('XGBoost', y_proba_xgb, 'blue'),
            ('LSTM', y_proba_lstm, 'green'),
            (f'Ensemble (w={self.optimal_weight:.2f})', prob_ensemble, 'red'),
        ]:
            fpr, tpr, _ = roc_curve(y_true, proba)
            auc = roc_auc_score(y_true, proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', linewidth=2, color=color)

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve Comparison: XGBoost vs LSTM vs Ensemble', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC comparison: {save_path}")
        plt.close()

    def plot_probability_scatter(
        self, y_true, y_proba_xgb, y_proba_lstm, save_path: str = None
    ):
        """Scatter plot of XGBoost vs LSTM probabilities, colored by true label."""
        fig, ax = plt.subplots(figsize=(8, 8))

        normal_mask = y_true == 0
        anomaly_mask = y_true == 1

        ax.scatter(y_proba_xgb[normal_mask], y_proba_lstm[normal_mask],
                  c='green', alpha=0.4, s=20, label=f'Normal (n={normal_mask.sum()})')
        ax.scatter(y_proba_xgb[anomaly_mask], y_proba_lstm[anomaly_mask],
                  c='red', alpha=0.4, s=20, label=f'Anomaly (n={anomaly_mask.sum()})')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Agreement line')

        ax.set_xlabel('XGBoost Probability', fontsize=12)
        ax.set_ylabel('LSTM Probability', fontsize=12)
        ax.set_title('XGBoost vs LSTM Predictions', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved probability scatter: {save_path}")
        plt.close()

    def save_ensemble_config(self, weight_result: Dict):
        """Save optimal ensemble configuration."""
        ensemble_output = {
            'optimal_weight_xgb': self.optimal_weight,
            'optimal_weight_lstm': 1 - self.optimal_weight,
            'formula': f'prob = {self.optimal_weight:.2f} * xgb + {1-self.optimal_weight:.2f} * lstm',
            'best_threshold': weight_result['best_row']['best_threshold'],
            'metrics': {
                'f2_score': weight_result['best_row']['f2_score'],
                'roc_auc': weight_result['best_row']['roc_auc'],
                'pr_auc': weight_result['best_row']['pr_auc'],
                'brier': weight_result['best_row']['brier'],
                'recall': weight_result['best_row']['recall'],
                'precision': weight_result['best_row']['precision'],
                'fpr': weight_result['best_row']['fpr'],
                'far_per_hour': weight_result['best_row']['far_per_hour'],
            },
            'confusion_matrix': {
                'tp': int(weight_result['best_row']['tp']),
                'fp': int(weight_result['best_row']['fp']),
                'fn': int(weight_result['best_row']['fn']),
                'tn': int(weight_result['best_row']['tn']),
            },
            'timestamp': self.timestamp,
        }

        output_path = Path('configs') / f'ensemble_optimal_{self.timestamp}.json'
        with open(output_path, 'w') as f:
            json.dump(ensemble_output, f, indent=2)

        logger.info(f"\nSaved ensemble config: {output_path}")

    def log_to_mlflow(self, metrics: Dict, comparison_df: pd.DataFrame):
        """Log ensemble results to MLflow."""
        logger.info("\nLogging to MLflow...")

        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

        run_name = f"{self.config['mlflow']['run_name_prefix']}_{self.timestamp}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param('optimal_weight_xgb', self.optimal_weight)
            mlflow.log_param('optimal_weight_lstm', 1 - self.optimal_weight)

            # Log ensemble metrics
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f'ensemble_{key}', val)

            logger.info(f"Logged to MLflow: {self.config['mlflow']['experiment_name']}")

    def run(self):
        """Run complete ensemble pipeline."""
        logger.info("\n" + "="*80)
        logger.info("WATCHTOWER - ENSEMBLE PARALLEL VOTING")
        logger.info("XGBoost (M0) + LSTM (M1)")
        logger.info("="*80 + "\n")

        # Load predictions
        y_true, y_proba_xgb, y_proba_lstm = self.load_predictions()

        # Find optimal weight
        weight_result = self.find_optimal_weight(y_true, y_proba_xgb, y_proba_lstm)

        # Compare all models
        comparison_df = self.compare_models(y_true, y_proba_xgb, y_proba_lstm)

        # Generate plots
        plots_dir = self.config['artifacts']['plots_dir']

        self.plot_weight_optimization(
            save_path=f"{plots_dir}/ensemble_weight_optimization_{self.timestamp}.png"
        )

        self.plot_roc_comparison(
            y_true, y_proba_xgb, y_proba_lstm,
            save_path=f"{plots_dir}/ensemble_roc_comparison_{self.timestamp}.png"
        )

        self.plot_probability_scatter(
            y_true, y_proba_xgb, y_proba_lstm,
            save_path=f"{plots_dir}/ensemble_probability_scatter_{self.timestamp}.png"
        )

        # Save ensemble config
        self.save_ensemble_config(weight_result)

        # Log to MLflow
        self.log_to_mlflow(weight_result['best_row'], comparison_df)

        # Final summary
        logger.info("\n" + "="*80)
        logger.info("ENSEMBLE PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nOptimal Ensemble: {self.optimal_weight:.2f} * XGBoost + {1-self.optimal_weight:.2f} * LSTM")
        logger.info(f"Best F2-Score:    {weight_result['best_row']['f2_score']:.4f}")
        logger.info(f"Best ROC-AUC:     {weight_result['best_row']['roc_auc']:.4f}")
        logger.info(f"Best FAR:         {weight_result['best_row']['far_per_hour']:.1f}/hour")
        logger.info("="*80)

        return weight_result, comparison_df


def main():
    """Main execution."""
    ensemble = EnsembleVoting()
    weight_result, comparison_df = ensemble.run()
    return weight_result, comparison_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
