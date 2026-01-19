#!/usr/bin/env python3
"""
WATCHTOWER - Optuna Hyperparameter Tuning with GroupKFold
Uses Bayesian optimization to find best XGBoost parameters.

Optuna is SMARTER than GridSearch:
- Learns from previous trials
- Focuses on promising parameter regions
- Much faster to find optimal parameters

Usage:
    python scripts/run_optuna_tuning.py
    python scripts/run_optuna_tuning.py --n_trials 50
    python scripts/run_optuna_tuning.py --n_trials 100 --timeout 3600

Author: Himanshu's WATCHTOWER Project
Date: 2026-01-13
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import joblib
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class OptunaXGBoostTuner:
    """
    Optuna-based hyperparameter tuning for XGBoost with GroupKFold.
    
    Uses Tree-structured Parzen Estimator (TPE) for smart search.
    """
    
    def __init__(self, n_splits=4, random_state=42):
        """
        Initialize tuner.
        
        Args:
            n_splits: Number of GroupKFold splits
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.X = None
        self.y = None
        self.groups = None
        self.scale_pos_weight = None
        self.best_params = None
        self.best_score = None
        
    def load_data(self):
        """Load preprocessed data."""
        logger.info("="*80)
        logger.info("LOADING DATA")
        logger.info("="*80)

        self.X = np.load('data/parquet/X.npy')
        self.y = np.load('data/parquet/y.npy')
        df = pd.read_parquet('data/parquet/features_table.parquet')
        self.groups = df['scenario_id'].values

        # Check and handle NaN values
        nan_count = np.isnan(self.X).sum()
        if nan_count > 0:
            logger.warning(f"⚠️  Found {nan_count} NaN values in X, replacing with 0")
            self.X = np.nan_to_num(self.X, nan=0.0)

        # Compute scale_pos_weight
        n_negative = (self.y == 0).sum()
        n_positive = (self.y == 1).sum()
        self.scale_pos_weight = n_negative / n_positive

        logger.info(f"✅ Loaded X: {self.X.shape}")
        logger.info(f"✅ Loaded y: {self.y.shape}")
        logger.info(f"   Unique scenarios: {len(np.unique(self.groups))}")
        logger.info(f"   Class distribution: {n_negative} negative, {n_positive} positive")
        logger.info(f"   scale_pos_weight: {self.scale_pos_weight:.3f}")
        
    def objective(self, trial):
        """
        Optuna objective function.

        Optuna will call this function many times, each time with different
        parameter suggestions, trying to maximize ROC-AUC.

        Args:
            trial: Optuna trial object

        Returns:
            Mean ROC-AUC across GroupKFold splits
        """
        try:
            return self._run_trial(trial)
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float('nan')

    def _run_trial(self, trial):
        """Run a single trial with suggested parameters."""
        # Suggest hyperparameters
        params = {
            # Number of boosting rounds
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            
            # Tree structure
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            
            # Learning rate
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            
            # Sampling
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            
            # Regularization
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),  # L1
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0),  # L2
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),  # Min loss reduction
            
            # Fixed parameters
            'scale_pos_weight': self.scale_pos_weight,
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist',
        }
        
        # GroupKFold cross-validation
        gkf = GroupKFold(n_splits=self.n_splits)
        fold_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(self.X, self.y, self.groups), 1):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            # Skip fold if only one class present
            if len(np.unique(y_test)) < 2:
                continue

            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)

            # Evaluate
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            fold_scores.append(roc_auc)

        # Return nan if no valid folds
        if len(fold_scores) == 0:
            return float('nan')
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        # Store fold scores as trial user attributes
        trial.set_user_attr('fold_scores', fold_scores)
        trial.set_user_attr('std_score', std_score)
        
        return mean_score
    
    def run_optimization(self, n_trials=100, timeout=None):
        """
        Run Optuna optimization.
        
        Args:
            n_trials: Number of trials to run
            timeout: Optional timeout in seconds
            
        Returns:
            Optuna study object
        """
        logger.info("\n" + "="*80)
        logger.info("OPTUNA HYPERPARAMETER OPTIMIZATION")
        logger.info("="*80)
        logger.info(f"Optimization algorithm: Tree-structured Parzen Estimator (TPE)")
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"CV folds: {self.n_splits}")
        logger.info(f"Timeout: {timeout if timeout else 'None'}")
        logger.info("="*80)
        
        # Create study
        study = optuna.create_study(
            direction='maximize',  # Maximize ROC-AUC
            sampler=TPESampler(seed=self.random_state),  # Bayesian optimization
            pruner=MedianPruner(n_startup_trials=10),  # Prune unpromising trials
            study_name='xgboost_groupkfold_optuna'
        )
        
        # Add callback for progress tracking
        def progress_callback(study, trial):
            if trial.value is None:
                logger.info(f"Trial {trial.number:3d}: FAILED")
                return
            if trial.number % 5 == 0:
                best_value = study.best_value if study.best_value is not None else 0.0
                logger.info(f"Trial {trial.number:3d}: ROC-AUC = {trial.value:.4f} (Best so far: {best_value:.4f})")
        
        # Optimize
        logger.info("\nStarting optimization...")
        logger.info("Each trial trains 4 models (one per fold)")
        logger.info("This will take a while - grab a coffee! ☕\n")
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[progress_callback],
            show_progress_bar=True,
            n_jobs=1  # Sequential (GroupKFold already parallel)
        )
        
        logger.info("\n✅ Optimization complete!")
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return study
    
    def analyze_results(self, study):
        """
        Analyze and display optimization results.
        
        Args:
            study: Optuna study object
        """
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("="*80)
        
        # Best trial
        best_trial = study.best_trial
        
        logger.info(f"\n🎯 BEST TRIAL")
        logger.info("-"*80)
        logger.info(f"Trial number: #{best_trial.number}")
        logger.info(f"ROC-AUC: {best_trial.value:.4f}")
        logger.info(f"Std: {best_trial.user_attrs.get('std_score', 0):.4f}")
        
        # Improvement over current
        current_score = 0.8090  # Your current performance
        improvement = (best_trial.value - current_score) * 100
        logger.info(f"\n📈 IMPROVEMENT")
        logger.info("-"*80)
        logger.info(f"Current model: {current_score:.4f}")
        logger.info(f"Best model:    {best_trial.value:.4f}")
        logger.info(f"Improvement:   {improvement:+.2f}%")
        
        # Best parameters
        logger.info(f"\n⚙️  BEST PARAMETERS")
        logger.info("-"*80)
        for key, value in sorted(best_trial.params.items()):
            logger.info(f"  {key:25s}: {value}")
        
        # Fold-by-fold scores
        fold_scores = best_trial.user_attrs.get('fold_scores', [])
        if fold_scores:
            logger.info(f"\n📊 FOLD-BY-FOLD PERFORMANCE")
            logger.info("-"*80)
            for i, score in enumerate(fold_scores, 1):
                logger.info(f"  Fold {i}: {score:.4f}")
        
        # Top 10 trials
        logger.info("\n" + "="*80)
        logger.info("TOP 10 TRIALS")
        logger.info("="*80)
        
        trials_df = study.trials_dataframe()
        trials_df = trials_df.sort_values('value', ascending=False)
        
        display_cols = ['number', 'value', 'params_n_estimators', 'params_max_depth', 
                       'params_learning_rate', 'params_subsample']
        available_cols = [col for col in display_cols if col in trials_df.columns]
        
        print("\n", trials_df[available_cols].head(10).to_string(index=False))
        
        return trials_df
    
    def train_final_model(self):
        """Train final model with best parameters on all data."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING FINAL MODEL WITH BEST PARAMETERS")
        logger.info("="*80)
        
        # Prepare final parameters
        final_params = self.best_params.copy()
        final_params.update({
            'scale_pos_weight': self.scale_pos_weight,
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist',
        })
        
        logger.info(f"\nTraining on all {len(self.X):,} samples...")
        
        # Train model
        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(self.X, self.y, verbose=False)
        
        logger.info("✅ Final model trained successfully!")
        
        return final_model
    
    def save_results(self, study, trials_df, final_model):
        """
        Save all optimization results.
        
        Args:
            study: Optuna study
            trials_df: DataFrame with all trials
            final_model: Trained final model
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create directories
        Path('reports').mkdir(exist_ok=True)
        Path('models').mkdir(exist_ok=True)
        Path('configs').mkdir(exist_ok=True)
        
        logger.info("\n" + "="*80)
        logger.info("SAVING RESULTS")
        logger.info("="*80)
        
        # 1. Save best parameters JSON
        best_params_path = f'configs/optuna_best_params_{timestamp}.json'
        best_config = {
            'best_score': float(self.best_score),
            'best_params': self.best_params,
            'scale_pos_weight': float(self.scale_pos_weight),
            'n_trials': len(study.trials),
            'n_splits': self.n_splits,
            'improvement_over_baseline': float((self.best_score - 0.8090) * 100),
            'timestamp': timestamp
        }
        
        with open(best_params_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        logger.info(f"✅ Best params JSON: {best_params_path}")
        
        # 2. Save best parameters YAML (for config file)
        yaml_path = f'configs/optuna_best_params_{timestamp}.yaml'
        yaml_config = {
            'xgboost': {
                k: (int(v) if isinstance(v, (np.integer, np.int64)) else 
                    float(v) if isinstance(v, (np.floating, np.float64)) else v)
                for k, v in self.best_params.items()
            }
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"✅ Best params YAML: {yaml_path}")
        
        # 3. Save all trials CSV
        trials_path = f'reports/optuna_trials_{timestamp}.csv'
        trials_df.to_csv(trials_path, index=False)
        logger.info(f"✅ All trials CSV: {trials_path}")
        
        # 4. Save final model
        model_path = f'models/xgboost_optuna_best_{timestamp}.joblib'
        joblib.dump(final_model, model_path)
        logger.info(f"✅ Best model: {model_path}")
        
        # 5. Save optimization summary
        summary_path = f'reports/optuna_summary_{timestamp}.txt'
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OPTUNA HYPERPARAMETER OPTIMIZATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Optimization Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Trials: {len(study.trials)}\n")
            f.write(f"CV Folds: {self.n_splits}\n\n")
            f.write(f"Best ROC-AUC: {self.best_score:.4f}\n")
            f.write(f"Baseline ROC-AUC: 0.8090\n")
            f.write(f"Improvement: {(self.best_score - 0.8090)*100:+.2f}%\n\n")
            f.write("Best Parameters:\n")
            f.write("-"*80 + "\n")
            for k, v in sorted(self.best_params.items()):
                f.write(f"  {k:25s}: {v}\n")
        
        logger.info(f"✅ Summary report: {summary_path}")
        
        return {
            'params_json': best_params_path,
            'params_yaml': yaml_path,
            'trials_csv': trials_path,
            'model': model_path,
            'summary': summary_path
        }
    
    def plot_optimization_history(self, study):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            
            logger.info("\n" + "="*80)
            logger.info("GENERATING PLOTS")
            logger.info("="*80)
            
            Path('reports/plots').mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Plot 1: Optimization history
            fig, ax = plt.subplots(figsize=(12, 6))
            
            trials = study.trials
            trial_numbers = [t.number for t in trials]
            values = [t.value for t in trials]
            best_values = [study.best_trials[0].value if i == 0 
                          else max([t.value for t in trials[:i+1]]) 
                          for i in range(len(trials))]
            
            ax.plot(trial_numbers, values, 'o-', alpha=0.5, label='Trial ROC-AUC')
            ax.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best ROC-AUC')
            ax.axhline(y=0.8090, color='g', linestyle='--', label='Baseline (0.8090)')
            ax.set_xlabel('Trial Number', fontsize=12)
            ax.set_ylabel('ROC-AUC', fontsize=12)
            ax.set_title('Optuna Optimization History', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            
            plot_path = f'reports/plots/optuna_history_{timestamp}.png'
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Optimization history: {plot_path}")
            
            # Plot 2: Parameter importances
            fig, ax = plt.subplots(figsize=(10, 6))
            
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())[:10]  # Top 10
            values = [importance[p] for p in params]
            
            ax.barh(params, values)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title('Hyperparameter Importance (Top 10)', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            importance_path = f'reports/plots/optuna_importance_{timestamp}.png'
            plt.tight_layout()
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Parameter importance: {importance_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Optuna Hyperparameter Tuning for XGBoost with GroupKFold',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (20 trials, ~10 minutes)
    python scripts/run_optuna_tuning.py --n_trials 20
    
    # Standard optimization (50 trials, ~25 minutes)
    python scripts/run_optuna_tuning.py --n_trials 50
    
    # Comprehensive search (100 trials, ~50 minutes)
    python scripts/run_optuna_tuning.py --n_trials 100
    
    # With timeout (stop after 1 hour)
    python scripts/run_optuna_tuning.py --n_trials 100 --timeout 3600
        """
    )
    
    parser.add_argument(
        '--n_trials',
        type=int,
        default=50,
        help='Number of optimization trials (default: 50)'
    )
    
    parser.add_argument(
        '--n_splits',
        type=int,
        default=4,
        help='Number of GroupKFold splits (default: 4)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds (default: None)'
    )
    
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()
    
    print("\n" + "🎯 "*40)
    print("WATCHTOWER - OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("Bayesian Optimization with GroupKFold")
    print("🎯 "*40 + "\n")
    
    # Initialize tuner
    tuner = OptunaXGBoostTuner(n_splits=args.n_splits)
    
    # Load data
    tuner.load_data()
    
    # Run optimization
    study = tuner.run_optimization(n_trials=args.n_trials, timeout=args.timeout)
    
    # Analyze results
    trials_df = tuner.analyze_results(study)
    
    # Train final model
    final_model = tuner.train_final_model()
    
    # Save results
    outputs = tuner.save_results(study, trials_df, final_model)
    
    # Plot results
    tuner.plot_optimization_history(study)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ OPTUNA OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"\n📊 Results:")
    print(f"   Baseline ROC-AUC:  0.8090")
    print(f"   Best ROC-AUC:      {tuner.best_score:.4f}")
    print(f"   Improvement:       {(tuner.best_score - 0.8090)*100:+.2f}%")
    print(f"\n📁 Outputs:")
    for key, path in outputs.items():
        print(f"   {key:15s}: {path}")
    print("\n" + "="*80)
    print("\n✨ Next Steps:")
    print("   1. Review best parameters in configs/")
    print("   2. Update configs/xgboost_config.yaml with best params")
    print("   3. Retrain with: python scripts/run_xgboost_training.py")
    print("   4. Deploy the best model from models/")
    print("="*80 + "\n")
    
    return study


if __name__ == "__main__":
    import sys
    try:
        study = main()
    except KeyboardInterrupt:
        logger.info("\n\nOptimization interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        sys.exit(1)
