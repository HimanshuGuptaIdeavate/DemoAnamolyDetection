"""
WATCHTOWER - LSTM Training Module
Trains M1 LSTM model with GroupKFold cross-validation.

Same CV structure as XGBoost to ensure ensemble alignment.
Input: Raw 2Hz sequences (871, 10, 16) — 8 raw + 8 delta features
Output: Trained LSTM model + CV predictions (compatible with XGBoost format)

Author: Himanshu's WATCHTOWER Project
Date: 2026-02-11
"""

import numpy as np
import pandas as pd
import yaml
import json
import joblib
import mlflow
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any
import logging

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    fbeta_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Import threshold selection (shared with XGBoost)
from watchtower.models.threshold_selection import tune_threshold_from_cv, WINDOWS_PER_HOUR

logger = logging.getLogger(__name__)


class LSTMTrainer:
    """
    LSTM training pipeline with GroupKFold cross-validation.

    Uses the same fold structure as XGBoost to ensure predictions
    are aligned for parallel voting ensemble.
    """

    def __init__(self, config_path: str = 'configs/lstm_config.yaml'):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.model = None
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        self._create_directories()

        # Set random seeds for reproducibility
        seed = self.config['split']['random_state']
        np.random.seed(seed)
        tf.random.set_seed(seed)

        logger.info("LSTM Trainer initialized")

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
        Load LSTM data.

        Returns:
            X: (n_windows, 10, 16) sequences (8 raw + 8 delta features)
            y: (n_windows,) labels
            groups: (n_windows,) scenario IDs
        """
        logger.info("="*80)
        logger.info("LOADING LSTM DATA")
        logger.info("="*80)

        X = np.load(self.config['data']['X_lstm_path'])
        y = np.load(self.config['data']['y_path'])
        groups = np.load(self.config['data']['groups_path'], allow_pickle=True)

        logger.info(f"X_lstm: {X.shape} (windows, timesteps, features)")
        logger.info(f"y:      {y.shape}")
        logger.info(f"groups: {groups.shape}")
        logger.info(f"Unique scenarios: {np.unique(groups)}")

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            logger.info(f"  Class {cls}: {count:,} ({count/len(y)*100:.1f}%)")

        return X, y, groups

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build Bidirectional LSTM model.

        Architecture:
            Input (10, 16)
            -> Bidirectional(LSTM(64, return_sequences=True))
            -> BatchNormalization
            -> Dropout(0.4)
            -> Bidirectional(LSTM(32))
            -> BatchNormalization
            -> Dropout(0.4)
            -> Dense(32, relu)
            -> Dropout(0.3)
            -> Dense(1, sigmoid)

        Bidirectional reads sequences forward AND backward, capturing
        patterns like "SINR drops then recovers" from both directions.

        Args:
            input_shape: (timesteps, features) = (10, 16)

        Returns:
            Compiled Keras Sequential model
        """
        lstm_config = self.config['lstm']
        train_config = self.config['training']

        use_bidirectional = lstm_config.get('bidirectional', True)

        layers = [Input(shape=input_shape)]

        # First LSTM layer
        lstm_1 = LSTM(lstm_config['units_1'], return_sequences=True)
        if use_bidirectional:
            layers.append(Bidirectional(lstm_1))
        else:
            layers.append(lstm_1)
        layers.append(BatchNormalization())
        layers.append(Dropout(lstm_config['dropout']))

        # Second LSTM layer
        lstm_2 = LSTM(lstm_config['units_2'], return_sequences=False)
        if use_bidirectional:
            layers.append(Bidirectional(lstm_2))
        else:
            layers.append(lstm_2)
        layers.append(BatchNormalization())
        layers.append(Dropout(lstm_config['dropout']))

        # Dense layers
        layers.append(Dense(lstm_config['dense_units'], activation=lstm_config['activation']))
        layers.append(Dropout(lstm_config.get('dense_dropout', 0.3)))
        layers.append(Dense(1, activation='sigmoid'))

        model = Sequential(layers)

        model.compile(
            optimizer=Adam(learning_rate=train_config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _get_callbacks(self) -> list:
        """Create training callbacks."""
        train_config = self.config['training']

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=train_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=train_config['reduce_lr_factor'],
                patience=train_config['reduce_lr_patience'],
                min_lr=train_config['min_lr'],
                verbose=1
            )
        ]

        return callbacks

    def _compute_class_weight(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced data."""
        n_negative = (y == 0).sum()
        n_positive = (y == 1).sum()
        total = len(y)

        # Same approach as XGBoost's scale_pos_weight
        weight_0 = total / (2.0 * n_negative)
        weight_1 = total / (2.0 * n_positive)

        class_weight = {0: weight_0, 1: weight_1}

        logger.info(f"  Class weights: {class_weight}")
        return class_weight

    def _normalize_data(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Normalize data per-feature (fit on train, transform both).

        Input shape: (n_windows, 10, 4)
        Reshape to (n_windows*10, 4) -> scale -> reshape back
        """
        n_train, timesteps, n_features = X_train.shape
        n_test = X_test.shape[0]

        # Reshape: (n, 10, 4) -> (n*10, 4)
        X_train_flat = X_train.reshape(-1, n_features)
        X_test_flat = X_test.reshape(-1, n_features)

        # Fit scaler on TRAIN only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)

        # Reshape back: (n*10, 4) -> (n, 10, 4)
        X_train_scaled = X_train_scaled.reshape(n_train, timesteps, n_features)
        X_test_scaled = X_test_scaled.reshape(n_test, timesteps, n_features)

        return X_train_scaled, X_test_scaled, scaler

    def train_with_groupkfold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train with GroupKFold cross-validation.
        Same fold structure as XGBoost for ensemble alignment.

        Args:
            X: (n_windows, 10, 4) raw sequences
            y: (n_windows,) labels
            groups: (n_windows,) scenario IDs

        Returns:
            Dictionary with training results (same format as XGBoost)
        """
        logger.info("\n" + "="*80)
        logger.info("GROUPKFOLD CROSS-VALIDATION (LSTM)")
        logger.info("="*80)

        n_splits = self.config['split']['n_splits']
        gkf = GroupKFold(n_splits=n_splits)

        train_config = self.config['training']
        input_shape = (X.shape[1], X.shape[2])  # (10, 4)

        cv_results = {
            'fold_scores': [],
            'fold_models': [],
            'fold_predictions': [],
            'fold_histories': [],
            'fold_scalers': []
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
            logger.info(f"Test scenarios:  {np.unique(groups_test)}")
            logger.info(f"Train size: {len(X_train):,}")
            logger.info(f"Test size:  {len(X_test):,}")

            # Normalize (fit on train only)
            X_train_scaled, X_test_scaled, scaler = self._normalize_data(X_train, X_test)
            logger.info(f"Normalization: fit on train ({len(X_train)}), transform test ({len(X_test)})")

            # Compute class weights
            class_weight = self._compute_class_weight(y_train)

            # Build fresh model for each fold
            model = self.build_model(input_shape)

            if fold == 1:
                model.summary(print_fn=logger.info)

            # Train
            logger.info(f"\nTraining fold {fold}...")
            history = model.fit(
                X_train_scaled, y_train,
                epochs=train_config['epochs'],
                batch_size=train_config['batch_size'],
                validation_split=train_config['validation_split'],
                class_weight=class_weight,
                callbacks=self._get_callbacks(),
                verbose=0
            )

            # Get actual epochs trained
            actual_epochs = len(history.history['loss'])
            best_val_loss = min(history.history['val_loss'])
            logger.info(f"  Trained for {actual_epochs} epochs (best val_loss: {best_val_loss:.4f})")

            # Predict
            y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
            threshold = self.config['evaluation']['probability_threshold']
            y_pred = (y_pred_proba >= threshold).astype(int)

            # Metrics
            if len(np.unique(y_test)) < 2:
                roc_auc = np.nan
                pr_auc = np.nan
                logger.warning(f"Fold {fold}: Only one class in test set")
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                pr_auc = average_precision_score(y_test, y_pred_proba)

            brier = brier_score_loss(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            logger.info(f"\nFold {fold} Results:")
            logger.info(f"  ROC-AUC:   {roc_auc:.4f}" if not np.isnan(roc_auc) else f"  ROC-AUC:   N/A")
            logger.info(f"  PR-AUC:    {pr_auc:.4f}" if not np.isnan(pr_auc) else f"  PR-AUC:    N/A")
            logger.info(f"  Brier:     {brier:.4f}")
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1-Score:  {f1:.4f}")

            # Store results (SAME format as XGBoost cv_results)
            cv_results['fold_scores'].append({
                'fold': fold,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'brier': brier,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'epochs_trained': actual_epochs,
                'best_val_loss': best_val_loss,
                'test_scenarios': np.unique(groups_test).tolist()
            })
            cv_results['fold_models'].append(model)
            cv_results['fold_predictions'].append({
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            })
            cv_results['fold_histories'].append(history.history)
            cv_results['fold_scalers'].append(scaler)

        # CV Summary
        scores_df = pd.DataFrame(cv_results['fold_scores'])

        logger.info("\n" + "="*80)
        logger.info("CROSS-VALIDATION SUMMARY (LSTM)")
        logger.info("="*80)

        roc_auc_mean = np.nanmean(scores_df['roc_auc'])
        roc_auc_std = np.nanstd(scores_df['roc_auc'])
        pr_auc_mean = np.nanmean(scores_df['pr_auc'])
        pr_auc_std = np.nanstd(scores_df['pr_auc'])
        brier_mean = scores_df['brier'].mean()
        brier_std = scores_df['brier'].std()

        logger.info(f"\nMean ROC-AUC:   {roc_auc_mean:.4f} +/- {roc_auc_std:.4f}")
        logger.info(f"Mean PR-AUC:    {pr_auc_mean:.4f} +/- {pr_auc_std:.4f}")
        logger.info(f"Mean Brier:     {brier_mean:.4f} +/- {brier_std:.4f} (lower is better)")
        logger.info(f"Mean Accuracy:  {scores_df['accuracy'].mean():.4f} +/- {scores_df['accuracy'].std():.4f}")
        logger.info(f"Mean Precision: {scores_df['precision'].mean():.4f} +/- {scores_df['precision'].std():.4f}")
        logger.info(f"Mean Recall:    {scores_df['recall'].mean():.4f} +/- {scores_df['recall'].std():.4f}")
        logger.info(f"Mean F1-Score:  {scores_df['f1'].mean():.4f} +/- {scores_df['f1'].std():.4f}")
        logger.info(f"Mean Epochs:    {scores_df['epochs_trained'].mean():.0f}")

        return cv_results

    def train_final_model(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[keras.Model, StandardScaler]:
        """
        Train final LSTM model on all data.

        Args:
            X: All raw sequences
            y: All labels

        Returns:
            Trained model and scaler
        """
        logger.info("\n" + "="*80)
        logger.info("TRAINING FINAL LSTM MODEL ON ALL DATA")
        logger.info("="*80)

        train_config = self.config['training']
        input_shape = (X.shape[1], X.shape[2])

        # Normalize all data
        n_samples, timesteps, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flat).reshape(n_samples, timesteps, n_features)

        # Class weight
        class_weight = self._compute_class_weight(y)

        # Build model
        model = self.build_model(input_shape)

        # Train (no validation split for final model, use early stopping epochs from CV)
        logger.info(f"Training on {len(X):,} samples...")
        model.fit(
            X_scaled, y,
            epochs=train_config['epochs'],
            batch_size=train_config['batch_size'],
            class_weight=class_weight,
            verbose=0
        )

        logger.info("Final LSTM model trained successfully")

        self.model = model
        return model, scaler

    def plot_training_history(self, cv_results: Dict, save_path: str = None):
        """Plot training loss curves for all folds."""
        n_folds = len(cv_results['fold_histories'])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for fold_idx, history in enumerate(cv_results['fold_histories'], 1):
            axes[0].plot(history['loss'], label=f'Fold {fold_idx}', alpha=0.8)
            axes[1].plot(history['val_loss'], label=f'Fold {fold_idx}', alpha=0.8)

        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.suptitle('LSTM Training History (All Folds)', fontsize=15, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training history: {save_path}")
        plt.close()

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        """Plot confusion matrix."""
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly']
        )
        plt.title('LSTM Confusion Matrix (Pooled CV)', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix: {save_path}")
        plt.close()

    def plot_roc_curve(self, y_true, y_proba, roc_auc, save_path: str = None):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'LSTM (AUC = {roc_auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('LSTM ROC Curve (Pooled CV)', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve: {save_path}")
        plt.close()

    def save_model(self, model: keras.Model, scaler: StandardScaler) -> str:
        """Save trained LSTM model and scaler."""
        model_dir = self.config['artifacts']['model_dir']
        filename_template = self.config['artifacts']['model_filename']
        filename = filename_template.format(timestamp=self.timestamp)
        model_path = Path(model_dir) / filename

        model.save(model_path)
        logger.info(f"Model saved: {model_path}")

        # Save scaler alongside model
        scaler_path = Path(model_dir) / f'lstm_scaler_{self.timestamp}.joblib'
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved: {scaler_path}")

        return str(model_path)

    def _cleanup_old_plots(self, plots_dir: str):
        """Remove old LSTM plot files."""
        patterns = [
            'lstm_confusion_matrix_*.png',
            'lstm_roc_curve_*.png',
            'lstm_training_history_*.png',
            'lstm_probability_distribution_*.png',
        ]

        plots_path = Path(plots_dir)
        removed = 0
        for pattern in patterns:
            for old_file in plots_path.glob(pattern):
                try:
                    old_file.unlink()
                    removed += 1
                except Exception as e:
                    logger.warning(f"Could not remove {old_file}: {e}")

        if removed > 0:
            logger.info(f"Cleaned up {removed} old LSTM plot(s)")

    def log_to_mlflow(self, model, metrics: Dict, cv_results: Dict):
        """Log training run to MLflow."""
        logger.info("\n" + "="*80)
        logger.info("LOGGING TO MLFLOW")
        logger.info("="*80)

        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

        run_name = f"{self.config['mlflow']['run_name_prefix']}_{self.timestamp}"

        with mlflow.start_run(run_name=run_name):
            # Log LSTM architecture params
            mlflow.log_params({
                'lstm_units_1': self.config['lstm']['units_1'],
                'lstm_units_2': self.config['lstm']['units_2'],
                'bidirectional': self.config['lstm'].get('bidirectional', True),
                'dropout': self.config['lstm']['dropout'],
                'dense_dropout': self.config['lstm'].get('dense_dropout', 0.3),
                'dense_units': self.config['lstm']['dense_units'],
                'learning_rate': self.config['training']['learning_rate'],
                'batch_size': self.config['training']['batch_size'],
                'n_splits': self.config['split']['n_splits'],
            })

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log CV results
            if cv_results:
                scores_df = pd.DataFrame(cv_results['fold_scores'])
                for metric in ['roc_auc', 'pr_auc', 'brier', 'accuracy', 'precision', 'recall', 'f1']:
                    mlflow.log_metric(f'cv_mean_{metric}', np.nanmean(scores_df[metric]))
                    mlflow.log_metric(f'cv_std_{metric}', np.nanstd(scores_df[metric]))

            logger.info(f"Logged to MLflow experiment: {self.config['mlflow']['experiment_name']}")
            logger.info(f"  Run name: {run_name}")

    def run(self):
        """Run complete LSTM training pipeline."""
        logger.info("\n" + "="*80)
        logger.info("WATCHTOWER - LSTM TRAINING PIPELINE (M1)")
        logger.info("="*80 + "\n")

        # Load data
        X, y, groups = self.load_data()

        # GroupKFold cross-validation
        cv_results = self.train_with_groupkfold(X, y, groups)

        # Train final model on all data
        final_model, final_scaler = self.train_final_model(X, y)

        # Save model
        model_path = self.save_model(final_model, final_scaler)

        # Generate plots
        if self.config['evaluation']['generate_plots']:
            plots_dir = self.config['artifacts']['plots_dir']
            self._cleanup_old_plots(plots_dir)

            # Pool all CV predictions
            y_all = np.concatenate([f['y_true'] for f in cv_results['fold_predictions']])
            y_proba_all = np.concatenate([f['y_pred_proba'] for f in cv_results['fold_predictions']])
            threshold = self.config['evaluation']['probability_threshold']
            y_pred_all = (y_proba_all >= threshold).astype(int)

            logger.info(f"\nPlots use pooled CV predictions: {len(y_all)} samples, threshold={threshold}")

            # Training history
            self.plot_training_history(
                cv_results,
                save_path=f"{plots_dir}/lstm_training_history_{self.timestamp}.png"
            )

            # Confusion matrix
            cm = confusion_matrix(y_all, y_pred_all)
            self.plot_confusion_matrix(
                cm,
                save_path=f"{plots_dir}/lstm_confusion_matrix_{self.timestamp}.png"
            )

            # ROC curve
            roc_auc = roc_auc_score(y_all, y_proba_all)
            self.plot_roc_curve(
                y_all, y_proba_all, roc_auc,
                save_path=f"{plots_dir}/lstm_roc_curve_{self.timestamp}.png"
            )

        # Aggregate metrics
        scores_df = pd.DataFrame(cv_results['fold_scores'])
        metrics = {
            'cv_mean_roc_auc': np.nanmean(scores_df['roc_auc']),
            'cv_mean_pr_auc': np.nanmean(scores_df['pr_auc']),
            'cv_mean_brier': scores_df['brier'].mean(),
            'cv_mean_accuracy': scores_df['accuracy'].mean(),
            'cv_mean_precision': scores_df['precision'].mean(),
            'cv_mean_recall': scores_df['recall'].mean(),
            'cv_mean_f1': scores_df['f1'].mean()
        }

        # Threshold tuning or manual FAR calculation
        if self.config.get('threshold_tuning', {}).get('enabled', True):
            plots_dir = self.config['artifacts']['plots_dir']
            optimal_threshold, threshold_metrics = tune_threshold_from_cv(
                cv_results,
                save_plot=True,
                plots_dir=plots_dir
            )

            metrics['optimal_threshold'] = optimal_threshold
            metrics['threshold_f2_score'] = threshold_metrics['f2_score']
            metrics['threshold_precision'] = threshold_metrics['precision']
            metrics['threshold_recall'] = threshold_metrics['recall']
            metrics['threshold_fpr'] = threshold_metrics['fpr']
            metrics['threshold_far_per_hour'] = threshold_metrics['far_per_hour']
            metrics['threshold_tp'] = threshold_metrics['tp']
            metrics['threshold_fp'] = threshold_metrics['fp']
            metrics['threshold_fn'] = threshold_metrics['fn']
            metrics['threshold_tn'] = threshold_metrics['tn']
        else:
            # Manual threshold FAR calculation
            threshold = self.config['evaluation']['probability_threshold']
            all_y_true = np.concatenate([f['y_true'] for f in cv_results['fold_predictions']])
            all_y_proba = np.concatenate([f['y_pred_proba'] for f in cv_results['fold_predictions']])
            all_y_pred = (all_y_proba >= threshold).astype(int)

            cm_pooled = confusion_matrix(all_y_true, all_y_pred)
            if cm_pooled.shape == (2, 2):
                tn, fp, fn, tp = cm_pooled.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0

            n_normal = fp + tn
            fpr = fp / n_normal if n_normal > 0 else 0
            normal_rate_per_hour = n_normal / len(all_y_true) * WINDOWS_PER_HOUR
            far_per_hour = fpr * normal_rate_per_hour

            metrics['threshold'] = threshold
            metrics['threshold_precision'] = precision_score(all_y_true, all_y_pred, zero_division=0)
            metrics['threshold_recall'] = recall_score(all_y_true, all_y_pred, zero_division=0)
            metrics['threshold_f2_score'] = fbeta_score(all_y_true, all_y_pred, beta=2, zero_division=0)
            metrics['threshold_fpr'] = fpr
            metrics['threshold_far_per_hour'] = far_per_hour
            metrics['threshold_tp'] = int(tp)
            metrics['threshold_fp'] = int(fp)
            metrics['threshold_fn'] = int(fn)
            metrics['threshold_tn'] = int(tn)

        # Log to MLflow
        self.log_to_mlflow(final_model, metrics, cv_results)

        # Save CV predictions for ensemble (Phase 3)
        self._save_cv_predictions(cv_results)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("LSTM TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nModel saved: {model_path}")
        logger.info(f"Mean CV ROC-AUC: {metrics['cv_mean_roc_auc']:.4f}")
        logger.info(f"Mean CV PR-AUC:  {metrics['cv_mean_pr_auc']:.4f}")
        logger.info(f"Mean CV Brier:   {metrics['cv_mean_brier']:.4f} (lower is better)")
        logger.info(f"Mean CV Accuracy: {metrics['cv_mean_accuracy']:.4f}")

        if 'optimal_threshold' in metrics:
            used_threshold = metrics['optimal_threshold']
            logger.info(f"\nOPTIMAL THRESHOLD (F2): {used_threshold:.4f}")
        else:
            used_threshold = metrics['threshold']
            logger.info(f"\nMANUAL THRESHOLD: {used_threshold}")

        logger.info(f"  Confusion Matrix: TP={metrics.get('threshold_tp', 'N/A')}, FP={metrics.get('threshold_fp', 'N/A')}, FN={metrics.get('threshold_fn', 'N/A')}, TN={metrics.get('threshold_tn', 'N/A')}")
        logger.info(f"  Recall:    {metrics['threshold_recall']:.4f} (catches {metrics['threshold_recall']*100:.1f}% of anomalies)")
        logger.info(f"  Precision: {metrics['threshold_precision']:.4f} ({metrics['threshold_precision']*100:.1f}% of alerts are real)")
        logger.info(f"  F2-Score:  {metrics['threshold_f2_score']:.4f}")
        logger.info(f"  FPR:       {metrics['threshold_fpr']:.4f} ({metrics['threshold_fpr']*100:.1f}% false positive rate)")
        logger.info(f"  FAR:       {metrics['threshold_far_per_hour']:.1f} false alarms/hour")

        if metrics['threshold_far_per_hour'] <= 10:
            logger.info(f"  Status:    EXCELLENT - Very low alarm fatigue risk")
        elif metrics['threshold_far_per_hour'] <= 30:
            logger.info(f"  Status:    GOOD - Operationally sustainable")
        elif metrics['threshold_far_per_hour'] <= 60:
            logger.info(f"  Status:    MODERATE - May cause some alarm fatigue")
        else:
            logger.info(f"  Status:    HIGH - Alarm fatigue risk")

        logger.info("\n" + "="*80)

        return final_model, cv_results

    def _save_cv_predictions(self, cv_results: Dict):
        """Save CV predictions for ensemble (Phase 3 needs these)."""
        y_true_all = np.concatenate([f['y_true'] for f in cv_results['fold_predictions']])
        y_proba_all = np.concatenate([f['y_pred_proba'] for f in cv_results['fold_predictions']])

        output_dir = Path(self.config['artifacts']['report_dir'])
        np.save(output_dir / 'lstm_cv_y_true.npy', y_true_all)
        np.save(output_dir / 'lstm_cv_y_proba.npy', y_proba_all)

        logger.info(f"\nSaved LSTM CV predictions for ensemble:")
        logger.info(f"  reports/lstm_cv_y_true.npy  -> {y_true_all.shape}")
        logger.info(f"  reports/lstm_cv_y_proba.npy -> {y_proba_all.shape}")


def main():
    """Main execution."""
    trainer = LSTMTrainer(config_path='configs/lstm_config.yaml')
    model, cv_results = trainer.run()
    return model, cv_results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
