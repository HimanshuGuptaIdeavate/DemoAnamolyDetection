"""
WATCHTOWER - CV Threshold Tuning with F2-Score
Finds optimal classification threshold using cross-validation results.

Goal: Catch more anomalies (high recall) with fewer false alarms (good precision)
Method: F2-Score optimization across CV folds (avoids overfitting)

F2 = 5 × (Precision × Recall) / (4 × Precision + Recall)
- Weighs recall 2x more than precision
- Perfect for anomaly detection where missing anomalies is costly

Metrics:
- FPR (False Positive Rate): FP / (FP + TN) - statistical metric
- FAR (False Alarms/Hour): Operational metric based on window size

Author: Himanshu's WATCHTOWER Project
Date: 2026-01-30
"""

# Window size in seconds (from windowing_config.py)
WINDOW_SEC = 5
WINDOWS_PER_HOUR = 3600 / WINDOW_SEC  # 720 windows per hour

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    fbeta_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class CVThresholdTuner:
    """
    Cross-Validation Threshold Tuning with F2-Score.

    Uses pooled predictions from all CV folds to find the optimal
    threshold that generalizes well and avoids overfitting.
    """

    def __init__(self, cv_results: Dict):
        """
        Initialize with cross-validation results.

        Args:
            cv_results: Results from GroupKFold CV containing 'fold_predictions'
                       Each fold has: {'y_true', 'y_pred', 'y_pred_proba'}
        """
        self.cv_results = cv_results
        self.y_true_pooled = None
        self.y_proba_pooled = None
        self.optimal_threshold = 0.5
        self.threshold_results = {}

        # Pool predictions from all folds
        self._pool_cv_predictions()

    def _pool_cv_predictions(self):
        """Pool predictions from all CV folds."""
        y_true_all = []
        y_proba_all = []

        for fold_pred in self.cv_results['fold_predictions']:
            y_true_all.extend(fold_pred['y_true'])
            y_proba_all.extend(fold_pred['y_pred_proba'])

        self.y_true_pooled = np.array(y_true_all)
        self.y_proba_pooled = np.array(y_proba_all)

        logger.info(f"Pooled {len(self.y_true_pooled)} samples from {len(self.cv_results['fold_predictions'])} folds")

    def find_optimal_threshold_f2(self) -> Dict[str, float]:
        """
        Find optimal threshold using F2-Score.

        F2 = 5 × (Precision × Recall) / (4 × Precision + Recall)
        - Recall is weighted 2x more than precision
        - Better for anomaly detection (catching anomalies is priority)

        Returns:
            Dict with optimal threshold and metrics
        """
        logger.info("\n" + "="*60)
        logger.info("CV THRESHOLD TUNING WITH F2-SCORE")
        logger.info("="*60)

        # Test thresholds from 0.1 to 0.9
        thresholds = np.linspace(0.1, 0.9, 81)  # 0.1, 0.11, 0.12, ..., 0.9

        best_f2 = 0
        best_threshold = 0.5
        all_results = []

        for thresh in thresholds:
            y_pred = (self.y_proba_pooled >= thresh).astype(int)

            # Compute metrics
            f2 = fbeta_score(self.y_true_pooled, y_pred, beta=2, zero_division=0)
            precision = precision_score(self.y_true_pooled, y_pred, zero_division=0)
            recall = recall_score(self.y_true_pooled, y_pred, zero_division=0)
            f1 = f1_score(self.y_true_pooled, y_pred, zero_division=0)

            # Compute FAR (False Alarm Rate) = FP / (FP + TN)
            cm_temp = confusion_matrix(self.y_true_pooled, y_pred)
            if cm_temp.shape == (2, 2):
                tn_temp, fp_temp, _, _ = cm_temp.ravel()
                far = fp_temp / (fp_temp + tn_temp) if (fp_temp + tn_temp) > 0 else 0
            else:
                far = 0

            all_results.append({
                'threshold': thresh,
                'f2_score': f2,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'far': far
            })

            if f2 > best_f2:
                best_f2 = f2
                best_threshold = thresh

        self.optimal_threshold = best_threshold
        self.threshold_results = pd.DataFrame(all_results)

        # Get final metrics at optimal threshold
        y_pred_optimal = (self.y_proba_pooled >= best_threshold).astype(int)
        cm = confusion_matrix(self.y_true_pooled, y_pred_optimal)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        # Calculate FPR (statistical) and FAR (operational)
        n_normal = fp + tn
        fpr = fp / n_normal if n_normal > 0 else 0

        # Calculate operational FAR (false alarms per hour)
        # Based on the proportion of normal windows that would trigger false alarms
        # Assuming continuous monitoring: normal_windows_per_hour * FPR
        normal_rate_per_hour = n_normal / len(self.y_true_pooled) * WINDOWS_PER_HOUR
        far_per_hour = fpr * normal_rate_per_hour

        # Total alerts per hour
        total_alerts = (tp + fp)
        alert_rate = total_alerts / len(self.y_true_pooled) * WINDOWS_PER_HOUR

        result = {
            'optimal_threshold': best_threshold,
            'f2_score': best_f2,
            'f1_score': f1_score(self.y_true_pooled, y_pred_optimal, zero_division=0),
            'precision': precision_score(self.y_true_pooled, y_pred_optimal, zero_division=0),
            'recall': recall_score(self.y_true_pooled, y_pred_optimal, zero_division=0),
            'fpr': fpr,
            'far_per_hour': far_per_hour,
            'total_alerts_per_hour': alert_rate,
            'accuracy': (tp + tn) / len(self.y_true_pooled),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }

        # Log results
        logger.info(f"\nOptimal Threshold: {result['optimal_threshold']:.4f}")
        logger.info(f"F2-Score:          {result['f2_score']:.4f}")
        logger.info(f"Precision:         {result['precision']:.4f}")
        logger.info(f"Recall:            {result['recall']:.4f}")
        logger.info(f"F1-Score:          {result['f1_score']:.4f}")
        logger.info(f"Accuracy:          {result['accuracy']:.4f}")

        logger.info(f"\n📊 FALSE ALARM ANALYSIS:")
        logger.info(f"  FPR (statistical):    {result['fpr']*100:.1f}%")
        logger.info(f"  FAR (operational):    {result['far_per_hour']:.1f} false alarms/hour")
        logger.info(f"  Total Alerts:         {result['total_alerts_per_hour']:.1f} alerts/hour")
        if far_per_hour > 0:
            logger.info(f"  Time Between FA:      {60/far_per_hour:.1f} minutes")

        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TP: {result['tp']} | FP: {result['fp']}")
        logger.info(f"  FN: {result['fn']} | TN: {result['tn']}")
        logger.info(f"\nInterpretation:")
        logger.info(f"  Catches {result['recall']*100:.1f}% of anomalies")
        logger.info(f"  {result['precision']*100:.1f}% of alerts are real anomalies")

        # Operational status assessment
        if far_per_hour <= 10:
            status = "✅ EXCELLENT - Very low alarm fatigue risk"
        elif far_per_hour <= 30:
            status = "✅ GOOD - Operationally sustainable"
        elif far_per_hour <= 60:
            status = "⚠️ MODERATE - May cause some alarm fatigue"
        else:
            status = "❌ HIGH - Alarm fatigue risk"

        logger.info(f"\n🚨 Operational Status: {status}")
        logger.info("="*60)

        return result

    def analyze_far_at_thresholds(self, thresholds: List[float] = None) -> pd.DataFrame:
        """
        Analyze FPR and operational FAR at different thresholds.

        FPR = FP / (FP + TN) = Statistical false positive rate
        FAR = False alarms per hour = Operational metric

        Args:
            thresholds: List of thresholds to analyze

        Returns:
            DataFrame with metrics at each threshold
        """
        if thresholds is None:
            thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

        logger.info("\n" + "="*70)
        logger.info("FALSE ALARM ANALYSIS AT DIFFERENT THRESHOLDS")
        logger.info("="*70)
        logger.info("\nFPR = False Positive Rate = FP / (FP + TN) [Statistical]")
        logger.info("FAR = False Alarms per Hour [Operational - what operators experience]\n")

        results = []
        n_total = len(self.y_true_pooled)

        for thresh in thresholds:
            y_pred = (self.y_proba_pooled >= thresh).astype(int)
            cm = confusion_matrix(self.y_true_pooled, y_pred)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0

            n_normal = fp + tn
            fpr = fp / n_normal if n_normal > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f2 = fbeta_score(self.y_true_pooled, y_pred, beta=2, zero_division=0)

            # Calculate operational FAR (false alarms per hour)
            normal_rate_per_hour = n_normal / n_total * WINDOWS_PER_HOUR
            far_per_hour = fpr * normal_rate_per_hour

            results.append({
                'threshold': thresh,
                'recall': recall,
                'precision': precision,
                'fpr': fpr,
                'far_per_hour': far_per_hour,
                'f2_score': f2,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            })

            logger.info(f"Threshold {thresh:.2f}: Recall={recall*100:5.1f}%, Precision={precision*100:5.1f}%, FPR={fpr*100:5.1f}%, FAR={far_per_hour:6.1f}/hour")

        df = pd.DataFrame(results)

        # Find best trade-off suggestion based on operational FAR
        logger.info("\n" + "-"*70)
        logger.info("OPERATIONAL TRADE-OFF ANALYSIS:")
        logger.info("-"*70)

        # Acceptable FAR thresholds (false alarms per hour)
        for max_far_hour in [5, 10, 15, 30, 60]:
            acceptable = df[df['far_per_hour'] <= max_far_hour]
            if len(acceptable) > 0:
                best = acceptable.loc[acceptable['recall'].idxmax()]
                logger.info(f"  Max {max_far_hour:3d} FA/hour: Threshold={best['threshold']:.2f} → Recall={best['recall']*100:.1f}%, Precision={best['precision']*100:.1f}%, FAR={best['far_per_hour']:.1f}/hour")

        # Recommendation based on operational sustainability
        logger.info("\n" + "-"*70)
        logger.info("RECOMMENDATION:")
        logger.info("-"*70)

        # Find threshold with FAR <= 30/hour and max recall
        sustainable = df[df['far_per_hour'] <= 30]
        if len(sustainable) > 0:
            recommended = sustainable.loc[sustainable['recall'].idxmax()]
            logger.info(f"  ⭐ Recommended: Threshold={recommended['threshold']:.2f}")
            logger.info(f"     Recall:    {recommended['recall']*100:.1f}% (catches {recommended['recall']*100:.0f}% of anomalies)")
            logger.info(f"     Precision: {recommended['precision']*100:.1f}%")
            logger.info(f"     FAR:       {recommended['far_per_hour']:.1f} false alarms/hour")
            logger.info(f"     Status:    Operationally sustainable ✅")
        else:
            logger.info("  ⚠️ No threshold achieves <= 30 FA/hour. Consider improving model or features.")

        logger.info("="*70)

        return df

    def validate_threshold_per_fold(self) -> pd.DataFrame:
        """
        Validate the optimal threshold on each fold separately.

        This shows how stable the threshold is across different data splits.
        Low variance = more robust threshold.

        Returns:
            DataFrame with per-fold metrics at optimal threshold
        """
        logger.info("\n" + "="*60)
        logger.info("THRESHOLD VALIDATION PER FOLD")
        logger.info("="*60)

        fold_results = []

        for i, fold_pred in enumerate(self.cv_results['fold_predictions'], 1):
            y_true = np.array(fold_pred['y_true'])
            y_proba = np.array(fold_pred['y_pred_proba'])
            y_pred = (y_proba >= self.optimal_threshold).astype(int)

            f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)

            fold_results.append({
                'fold': i,
                'f2_score': f2,
                'precision': precision,
                'recall': recall,
                'n_samples': len(y_true)
            })

            logger.info(f"Fold {i}: F2={f2:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

        df = pd.DataFrame(fold_results)

        # Summary statistics
        logger.info(f"\nSummary (threshold={self.optimal_threshold:.4f}):")
        logger.info(f"  Mean F2:       {df['f2_score'].mean():.4f} ± {df['f2_score'].std():.4f}")
        logger.info(f"  Mean Precision: {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
        logger.info(f"  Mean Recall:    {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")

        # Stability assessment
        std = df['f2_score'].std()
        if std < 0.02:
            stability = "EXCELLENT (very stable)"
        elif std < 0.05:
            stability = "GOOD (stable)"
        elif std < 0.10:
            stability = "MODERATE (some variance)"
        else:
            stability = "LOW (high variance - consider more data)"

        logger.info(f"  Stability:      {stability}")
        logger.info("="*60)

        return df

    def plot_threshold_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create threshold analysis visualization.

        Shows:
        1. F2-Score vs Threshold (with optimal point)
        2. Precision & Recall vs Threshold

        Args:
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure
        """
        if self.threshold_results is None or len(self.threshold_results) == 0:
            logger.warning("No threshold results. Run find_optimal_threshold_f2() first.")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        df = self.threshold_results

        # Plot 1: F2-Score vs Threshold
        ax1 = axes[0]
        ax1.plot(df['threshold'], df['f2_score'], 'b-', linewidth=2, label='F2-Score')
        ax1.plot(df['threshold'], df['f1_score'], 'g--', linewidth=1.5, alpha=0.7, label='F1-Score')

        # Mark optimal threshold
        optimal_idx = df['f2_score'].idxmax()
        ax1.scatter([self.optimal_threshold], [df.loc[optimal_idx, 'f2_score']],
                   color='red', s=150, zorder=5, marker='*',
                   label=f'Optimal: {self.optimal_threshold:.3f}')
        ax1.axvline(x=self.optimal_threshold, color='red', linestyle='--', alpha=0.5)

        ax1.set_xlabel('Threshold', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('F2-Score vs Threshold\n(F2 favors Recall 2x over Precision)', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0.1, 0.9])
        ax1.set_ylim([0, 1.05])

        # Plot 2: Precision & Recall vs Threshold
        ax2 = axes[1]
        ax2.plot(df['threshold'], df['precision'], 'b-', linewidth=2, label='Precision')
        ax2.plot(df['threshold'], df['recall'], 'r-', linewidth=2, label='Recall')

        # Mark optimal threshold
        ax2.axvline(x=self.optimal_threshold, color='green', linestyle='--', alpha=0.7,
                   label=f'Optimal Threshold: {self.optimal_threshold:.3f}')

        # Add annotation
        opt_precision = df.loc[optimal_idx, 'precision']
        opt_recall = df.loc[optimal_idx, 'recall']
        ax2.scatter([self.optimal_threshold], [opt_precision], color='blue', s=100, zorder=5)
        ax2.scatter([self.optimal_threshold], [opt_recall], color='red', s=100, zorder=5)

        ax2.set_xlabel('Threshold', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title(f'Precision & Recall vs Threshold\n(At {self.optimal_threshold:.3f}: P={opt_precision:.2f}, R={opt_recall:.2f})', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0.1, 0.9])
        ax2.set_ylim([0, 1.05])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"✅ Saved threshold analysis plot: {save_path}")

        plt.close()

        return fig


def tune_threshold_from_cv(
    cv_results: Dict,
    save_plot: bool = True,
    plots_dir: str = 'reports/plots'
) -> Tuple[float, Dict]:
    """
    Main function to tune threshold from CV results.

    Args:
        cv_results: Results from GroupKFold CV
        save_plot: Whether to save the analysis plot
        plots_dir: Directory to save plots

    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    # Create tuner
    tuner = CVThresholdTuner(cv_results)

    # Find optimal threshold
    result = tuner.find_optimal_threshold_f2()

    # Analyze FAR at different thresholds
    far_analysis = tuner.analyze_far_at_thresholds()

    # Validate per fold
    fold_validation = tuner.validate_threshold_per_fold()

    # Save plot
    if save_plot:
        from pathlib import Path
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
        plot_path = f"{plots_dir}/threshold_analysis.png"
        tuner.plot_threshold_analysis(save_path=plot_path)

    return result['optimal_threshold'], result
