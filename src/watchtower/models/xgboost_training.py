"""
WATCHTOWER - XGBoost Training Module
Trains M0 XGBoost model with GroupKFold cross-validation

Author: Himanshu's WATCHTOWER Project
Date: 2026-01-12
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import yaml
import json
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, List
import logging

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# Import threshold selection
from watchtower.models.threshold_selection import tune_threshold_from_cv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """
    XGBoost training pipeline with GroupKFold validation.
    """
    
    def __init__(self, config_path: str = 'configs/xgboost_config.yaml'):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_names = None
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directories
        self._create_directories()
        
        logger.info("XGBoost Trainer initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    
    def _create_directories(self):
        """Create output directories."""
        dirs = [
            self.config['artifacts']['model_dir'],
            self.config['artifacts']['report_dir'],
            self.config['artifacts']['plots_dir'],
            'logs',
            'mlruns'
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load preprocessed data.
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            groups: Scenario IDs for GroupKFold (n_samples,)
        """
        logger.info("="*80)
        logger.info("LOADING DATA")
        logger.info("="*80)
        
        # Load X, y
        X_path = self.config['data']['X_path']
        y_path = self.config['data']['y_path']
        
        X = np.load(X_path)
        y = np.load(y_path)
        
        logger.info(f"✅ Loaded X: {X.shape}")
        logger.info(f"✅ Loaded y: {y.shape}")
        
        # Load scenario IDs for GroupKFold
        features_table_path = self.config['data']['features_table_path']
        df = pd.read_parquet(features_table_path)
        group_column = self.config['data']['group_column']
        groups = df[group_column].values
        
        logger.info(f"✅ Loaded groups: {groups.shape}")
        logger.info(f"   Unique scenarios: {np.unique(groups)}")
        logger.info(f"   Scenario counts: {dict(zip(*np.unique(groups, return_counts=True)))}")
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"\nClass distribution:")
        for cls, count in zip(unique, counts):
            logger.info(f"   Class {cls}: {count:,} samples ({count/len(y)*100:.1f}%)")
        
        return X, y, groups
    
    def compute_scale_pos_weight(self, y: np.ndarray) -> float:
        """
        Compute scale_pos_weight for class imbalance.
        
        Args:
            y: Labels
            
        Returns:
            scale_pos_weight: #negative / #positive
        """
        n_negative = (y == 0).sum()
        n_positive = (y == 1).sum()
        
        scale_pos_weight = n_negative / n_positive
        
        logger.info(f"\nClass imbalance:")
        logger.info(f"   Negative samples: {n_negative:,}")
        logger.info(f"   Positive samples: {n_positive:,}")
        logger.info(f"   scale_pos_weight: {scale_pos_weight:.3f}")
        
        return scale_pos_weight
    
    def create_model(self, scale_pos_weight: float) -> xgb.XGBClassifier:
        """
        Create XGBoost classifier with configured hyperparameters.
        
        Args:
            scale_pos_weight: Weight for positive class
            
        Returns:
            XGBoost classifier
        """
        xgb_config = self.config['xgboost'].copy()
        
        # Replace 'auto' with computed value
        if xgb_config['scale_pos_weight'] == 'auto':
            xgb_config['scale_pos_weight'] = scale_pos_weight
        
        model = xgb.XGBClassifier(**xgb_config)
        
        logger.info("\nXGBoost configuration:")
        for key, value in xgb_config.items():
            logger.info(f"   {key}: {value}")
        
        return model
    
    def train_with_groupkfold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train with GroupKFold cross-validation.
        
        Args:
            X: Features
            y: Labels
            groups: Scenario IDs
            
        Returns:
            Dictionary with training results
        """
        logger.info("\n" + "="*80)
        logger.info("GROUPKFOLD CROSS-VALIDATION")
        logger.info("="*80)
        
        n_splits = self.config['split']['n_splits']
        gkf = GroupKFold(n_splits=n_splits)
        
        cv_results = {
            'fold_scores': [],
            'fold_models': [],
            'fold_predictions': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"FOLD {fold}/{n_splits}")
            logger.info(f"{'='*80}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]
            groups_test = groups[test_idx]
            
            logger.info(f"Train scenarios: {np.unique(groups_train)}")
            logger.info(f"Test scenarios: {np.unique(groups_test)}")
            logger.info(f"Train size: {len(X_train):,} samples")
            logger.info(f"Test size: {len(X_test):,} samples")
            
            # Compute scale_pos_weight for this fold
            scale_pos_weight = self.compute_scale_pos_weight(y_train)
            
            # Create and train model
            model = self.create_model(scale_pos_weight)
            
            logger.info(f"\nTraining fold {fold}...")
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= self.config['evaluation']['probability_threshold']).astype(int)

            # Handle single-class fold (ROC-AUC and PR-AUC undefined)
            if len(np.unique(y_test)) < 2:
                roc_auc = np.nan
                pr_auc = np.nan
                logger.warning(f"⚠️  Fold {fold}: Only one class in test set, ROC-AUC/PR-AUC undefined")
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                pr_auc = average_precision_score(y_test, y_pred_proba)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            logger.info(f"\nFold {fold} Results:")
            logger.info(f"   ROC-AUC:   {roc_auc:.4f}" if not np.isnan(roc_auc) else f"   ROC-AUC:   N/A (single class)")
            logger.info(f"   PR-AUC:    {pr_auc:.4f}" if not np.isnan(pr_auc) else f"   PR-AUC:    N/A (single class)")
            logger.info(f"   Accuracy:  {accuracy:.4f}")
            logger.info(f"   Precision: {precision:.4f}")
            logger.info(f"   Recall:    {recall:.4f}")
            logger.info(f"   F1-Score:  {f1:.4f}")
            
            # Store results
            cv_results['fold_scores'].append({
                'fold': fold,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'test_scenarios': np.unique(groups_test).tolist()
            })
            cv_results['fold_models'].append(model)
            cv_results['fold_predictions'].append({
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            })
        
        # Aggregate results (use nanmean to handle NaN from single-class folds)
        scores_df = pd.DataFrame(cv_results['fold_scores'])

        logger.info("\n" + "="*80)
        logger.info("CROSS-VALIDATION SUMMARY")
        logger.info("="*80)
        roc_auc_mean = np.nanmean(scores_df['roc_auc'])
        roc_auc_std = np.nanstd(scores_df['roc_auc'])
        pr_auc_mean = np.nanmean(scores_df['pr_auc'])
        pr_auc_std = np.nanstd(scores_df['pr_auc'])
        valid_folds = scores_df['roc_auc'].notna().sum()
        logger.info(f"\nMean ROC-AUC:   {roc_auc_mean:.4f} ± {roc_auc_std:.4f} ({valid_folds}/{len(scores_df)} valid folds)")
        logger.info(f"Mean PR-AUC:    {pr_auc_mean:.4f} ± {pr_auc_std:.4f}")
        logger.info(f"Mean Accuracy:  {scores_df['accuracy'].mean():.4f} ± {scores_df['accuracy'].std():.4f}")
        logger.info(f"Mean Precision: {scores_df['precision'].mean():.4f} ± {scores_df['precision'].std():.4f}")
        logger.info(f"Mean Recall:    {scores_df['recall'].mean():.4f} ± {scores_df['recall'].std():.4f}")
        logger.info(f"Mean F1-Score:  {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f}")
        
        return cv_results
    
    def train_final_model(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> xgb.XGBClassifier:
        """
        Train final model on all data.
        
        Args:
            X: All features
            y: All labels
            
        Returns:
            Trained XGBoost model
        """
        logger.info("\n" + "="*80)
        logger.info("TRAINING FINAL MODEL ON ALL DATA")
        logger.info("="*80)
        
        # Compute scale_pos_weight
        scale_pos_weight = self.compute_scale_pos_weight(y)
        
        # Create model (disable early stopping for final training)
        xgb_config = self.config['xgboost'].copy()
        if xgb_config['scale_pos_weight'] == 'auto':
            xgb_config['scale_pos_weight'] = scale_pos_weight
        
        # Remove early stopping parameters for final model
        xgb_config.pop('early_stopping_rounds', None)
        xgb_config.pop('eval_metric', None)
        
        model = xgb.XGBClassifier(**xgb_config)
        
        logger.info("\nXGBoost configuration (final model):")
        for key, value in xgb_config.items():
            logger.info(f"   {key}: {value}")
        
        # Train on all data
        logger.info(f"\nTraining on {len(X):,} samples...")
        model.fit(X, y, verbose=False)
        
        logger.info("✅ Final model trained successfully")
        
        self.model = model
        return model
    
    def evaluate_model(
        self,
        model: xgb.XGBClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        threshold = self.config['evaluation']['probability_threshold']
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"\nTest Set Results:")
        logger.info(f"   ROC-AUC:   {roc_auc:.4f}")
        logger.info(f"   Accuracy:  {accuracy:.4f}")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall:    {recall:.4f}")
        logger.info(f"   F1-Score:  {f1:.4f}")
        
        # Classification report
        logger.info(f"\nClassification Report:")
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'])
        logger.info(f"\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"                 Predicted: Normal  |  Predicted: Anomaly")
        logger.info(f"Actual: Normal   {cm[0,0]:>8}         |  {cm[0,1]:>8}")
        logger.info(f"Actual: Anomaly  {cm[1,0]:>8}         |  {cm[1,1]:>8}")
        
        return {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: str = None
    ):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved confusion matrix: {save_path}")
        plt.close()
    
    def plot_roc_curve(
        self,
        y_test: np.ndarray,
        y_pred_proba: np.ndarray,
        roc_auc: float,
        save_path: str = None
    ):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved ROC curve: {save_path}")
        plt.close()
    
    def plot_feature_importance(
        self,
        model: xgb.XGBClassifier,
        top_n: int = 20,
        save_path: str = None
    ):
        """Plot feature importance."""
        # Get feature importance
        importance_type = self.config['feature_importance']['method']
        importance = model.get_booster().get_score(importance_type=importance_type)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        })
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel(f'Importance ({importance_type})')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved feature importance: {save_path}")
        plt.close()
        
        return importance_df
    
    def save_model(self, model: xgb.XGBClassifier) -> str:
        """Save trained model."""
        model_dir = self.config['artifacts']['model_dir']
        filename_template = self.config['artifacts']['model_filename']
        filename = filename_template.format(timestamp=self.timestamp)
        model_path = Path(model_dir) / filename
        
        joblib.dump(model, model_path)
        logger.info(f"✅ Model saved: {model_path}")
        
        return str(model_path)
    
    def log_to_mlflow(
        self,
        model: xgb.XGBClassifier,
        metrics: Dict[str, float],
        cv_results: Dict[str, Any] = None
    ):
        """Log training run to MLflow."""
        logger.info("\n" + "="*80)
        logger.info("LOGGING TO MLFLOW")
        logger.info("="*80)
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        
        # Set experiment
        experiment_name = self.config['mlflow']['experiment_name']
        mlflow.set_experiment(experiment_name)
        
        # Start run
        run_name = f"{self.config['mlflow']['run_name_prefix']}_{self.timestamp}"
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params(self.config['xgboost'])
            mlflow.log_param('split_method', self.config['split']['method'])
            mlflow.log_param('n_splits', self.config['split']['n_splits'])
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log CV results if available
            if cv_results:
                scores_df = pd.DataFrame(cv_results['fold_scores'])
                for metric in ['roc_auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1']:
                    mlflow.log_metric(f'cv_mean_{metric}', np.nanmean(scores_df[metric]))
                    mlflow.log_metric(f'cv_std_{metric}', np.nanstd(scores_df[metric]))
            
            # Log model
            if self.config['mlflow']['log_model']:
                mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"✅ Logged to MLflow experiment: {experiment_name}")
            logger.info(f"   Run name: {run_name}")
    
    def run(self):
        """Run complete training pipeline."""
        logger.info("\n" + "🎯 "*40)
        logger.info("WATCHTOWER - XGBOOST TRAINING PIPELINE")
        logger.info("🎯 "*40 + "\n")
        
        # Load data
        X, y, groups = self.load_data()
        
        # GroupKFold cross-validation
        cv_results = self.train_with_groupkfold(X, y, groups)
        
        # Train final model on all data
        final_model = self.train_final_model(X, y)
        
        # Save model
        model_path = self.save_model(final_model)
        
        # Generate plots
        if self.config['evaluation']['generate_plots']:
            plots_dir = self.config['artifacts']['plots_dir']

            # Clean up old plot files before generating new ones
            self._cleanup_old_plots(plots_dir)

            # Use last fold for test evaluation plots
            last_fold_preds = cv_results['fold_predictions'][-1]
            y_test = last_fold_preds['y_true']
            y_pred_proba = last_fold_preds['y_pred_proba']
            y_pred = last_fold_preds['y_pred']
            
            cm = confusion_matrix(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Plot confusion matrix
            self.plot_confusion_matrix(
                cm,
                save_path=f"{plots_dir}/confusion_matrix_{self.timestamp}.png"
            )
            
            # Plot ROC curve
            self.plot_roc_curve(
                y_test,
                y_pred_proba,
                roc_auc,
                save_path=f"{plots_dir}/roc_curve_{self.timestamp}.png"
            )
            
            # Plot feature importance
            top_n = self.config['feature_importance']['top_n']
            self.plot_feature_importance(
                final_model,
                top_n=top_n,
                save_path=f"{plots_dir}/feature_importance_{self.timestamp}.png"
            )
        
        # Aggregate metrics
        scores_df = pd.DataFrame(cv_results['fold_scores'])
        metrics = {
            'cv_mean_roc_auc': np.nanmean(scores_df['roc_auc']),
            'cv_mean_pr_auc': np.nanmean(scores_df['pr_auc']),
            'cv_mean_accuracy': scores_df['accuracy'].mean(),
            'cv_mean_precision': scores_df['precision'].mean(),
            'cv_mean_recall': scores_df['recall'].mean(),
            'cv_mean_f1': scores_df['f1'].mean()
        }

        # CV Threshold Tuning with F2-Score
        if self.config.get('threshold_tuning', {}).get('enabled', True):
            plots_dir = self.config['artifacts']['plots_dir']
            optimal_threshold, threshold_metrics = tune_threshold_from_cv(
                cv_results,
                save_plot=True,
                plots_dir=plots_dir
            )

            # Update metrics with threshold tuning results
            metrics['optimal_threshold'] = optimal_threshold
            metrics['threshold_f2_score'] = threshold_metrics['f2_score']
            metrics['threshold_precision'] = threshold_metrics['precision']
            metrics['threshold_recall'] = threshold_metrics['recall']

            # Save optimal threshold to config for future use
            self._save_optimal_threshold(optimal_threshold, threshold_metrics)

        # Log to MLflow
        self.log_to_mlflow(final_model, metrics, cv_results)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nModel saved: {model_path}")
        logger.info(f"Mean CV ROC-AUC: {metrics['cv_mean_roc_auc']:.4f}")
        logger.info(f"Mean CV PR-AUC:  {metrics['cv_mean_pr_auc']:.4f}")
        logger.info(f"Mean CV Accuracy: {metrics['cv_mean_accuracy']:.4f}")
        if 'optimal_threshold' in metrics:
            logger.info(f"\n⭐ OPTIMAL THRESHOLD (F2): {metrics['optimal_threshold']:.4f}")
            logger.info(f"   At this threshold:")
            logger.info(f"   - Recall: {metrics['threshold_recall']:.4f} (catches {metrics['threshold_recall']*100:.1f}% of anomalies)")
            logger.info(f"   - Precision: {metrics['threshold_precision']:.4f}")
            logger.info(f"   - F2-Score: {metrics['threshold_f2_score']:.4f}")
        logger.info("\n" + "="*80)

        return final_model, cv_results

    def _save_optimal_threshold(self, threshold: float, metrics: Dict):
        """Save optimal threshold to a JSON file for production use."""
        threshold_config = {
            'optimal_threshold': threshold,
            'method': 'cv_f2_score',
            'metrics': {
                'f2_score': metrics['f2_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'accuracy': metrics['accuracy']
            },
            'confusion_matrix': {
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'tn': metrics['tn'],
                'fn': metrics['fn']
            },
            'timestamp': self.timestamp
        }

        # Save to configs directory
        output_path = Path('configs') / f'optimal_threshold_{self.timestamp}.json'
        with open(output_path, 'w') as f:
            json.dump(threshold_config, f, indent=2)

        logger.info(f"✅ Saved optimal threshold config: {output_path}")

    def _cleanup_old_plots(self, plots_dir: str):
        """Remove old plot files before generating new ones."""
        patterns = [
            'confusion_matrix_*.png',
            'roc_curve_*.png',
            'feature_importance_*.png',
            'threshold_analysis.png'
        ]

        plots_path = Path(plots_dir)
        removed_count = 0

        for pattern in patterns:
            for old_file in plots_path.glob(pattern):
                try:
                    old_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove {old_file}: {e}")

        if removed_count > 0:
            logger.info(f"🧹 Cleaned up {removed_count} old plot file(s)")


def main():
    """Main execution."""
    trainer = XGBoostTrainer(config_path='configs/xgboost_config.yaml')
    model, cv_results = trainer.run()
    return model, cv_results


if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
