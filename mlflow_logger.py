"""
WATCHTOWER - MLflow Integration for Data Validation
Comprehensive experiment tracking and artifact logging

This module logs validation results, data profiles, and visualizations to MLflow.
"""

import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json
import warnings
warnings.filterwarnings('ignore')

from validation_config import MLFLOW_CONFIG, get_critical_features


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class MLflowLogger:
    """
    MLflow logging utility for WATCHTOWER data validation and experiments.
    
    Features:
    - Automatic experiment tracking
    - Validation results logging
    - Data profiling
    - Visualization generation and logging
    - Parameter and metric tracking
    - Artifact management
    """
    
    def __init__(
        self, 
        experiment_name: str = "watchtower_data_validation",
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize MLflow logger.
        
        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking URI (defaults to config)
        """
        # Set tracking URI
        if tracking_uri is None:
            tracking_uri = MLFLOW_CONFIG['tracking_uri']
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id
        except:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        
        self.experiment_name = experiment_name
        self.plots_dir = Path("reports/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 MLflow initialized: experiment='{experiment_name}'")
        print(f"   Tracking URI: {tracking_uri}")
        print(f"   Experiment ID: {self.experiment_id}")
    
    def log_data_validation(
        self,
        df: pd.DataFrame,
        validation_results: Dict[str, Any],
        sutd_commit: Optional[str] = None,
        run_name: Optional[str] = None
    ):
        """
        Log complete data validation run to MLflow.
        
        Args:
            df: Validated DataFrame
            validation_results: Validation results from WatchtowerValidator
            sutd_commit: SUTD dataset Git commit hash
            run_name: Custom run name
        """
        if run_name is None:
            run_name = f"data_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name) as run:
            print(f"\n🚀 Starting MLflow run: {run_name}")
            print(f"   Run ID: {run.info.run_id}")
            
            # Log dataset parameters
            self._log_dataset_params(df, sutd_commit)
            
            # Log validation metrics
            self._log_validation_metrics(validation_results)
            
            # Log summary statistics
            self._log_summary_stats(validation_results.get('summary_stats', {}))
            
            # Generate and log visualizations
            if MLFLOW_CONFIG.get('log_plots', True):
                self._log_visualizations(df)
            
            # Log validation report as artifact
            if MLFLOW_CONFIG.get('log_validation_results', True):
                self._log_validation_artifact(validation_results)
            
            # Log data profile
            if MLFLOW_CONFIG.get('log_data_profile', True):
                self._log_data_profile(df)
            
            print(f"✅ MLflow run completed: {run.info.run_id}")
            print(f"   View at: mlflow ui --backend-store-uri mlruns")
            
            return run.info.run_id
    
    def _log_dataset_params(self, df: pd.DataFrame, sutd_commit: Optional[str]):
        """Log dataset parameters."""
        print("  📝 Logging dataset parameters...")
        
        params = {
            "dataset_rows": len(df),
            "dataset_cols": len(df.columns),
            "dataset_size_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "validation_timestamp": datetime.now().isoformat(),
        }
        
        if sutd_commit:
            params["sutd_dataset_commit"] = sutd_commit
        
        if 'source_file' in df.columns:
            params["num_source_files"] = df['source_file'].nunique()
        
        mlflow.log_params(params)
    
    def _log_validation_metrics(self, validation_results: Dict[str, Any]):
        """Log validation metrics."""
        print("  📊 Logging validation metrics...")
        
        metrics = {
            "validation_success": int(validation_results['success']),
            "total_checks": validation_results['total_checks'],
            "passed_checks": validation_results['passed_checks'],
            "failed_checks": validation_results['failed_checks'],
            "validation_score": validation_results['passed_checks'] / validation_results['total_checks'],
        }
        
        mlflow.log_metrics(metrics)
        
        # Log category-wise success rates
        for category, results in validation_results['results_by_category'].items():
            if results:
                passed = sum(results.values())
                total = len(results)
                mlflow.log_metric(f"{category}_pass_rate", passed / total)
    
    def _log_summary_stats(self, summary_stats: Dict[str, Any]):
        """Log summary statistics as metrics."""
        print("  📈 Logging summary statistics...")
        
        # Log basic stats
        for key, value in summary_stats.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                mlflow.log_metric(key, value)
    
    def _log_visualizations(self, df: pd.DataFrame):
        """Generate and log visualization artifacts."""
        print("  📊 Generating and logging visualizations...")
        
        critical_features = get_critical_features()
        
        # 1. Signal quality distribution
        self._plot_signal_distributions(df, critical_features)
        
        # 2. Anomaly comparison
        if 'lab_anom' in df.columns:
            self._plot_anomaly_comparison(df, critical_features)
        
        # 3. Correlation heatmap
        self._plot_correlation_heatmap(df, critical_features)
        
        # 4. Time series sample (if Time column exists)
        if 'Time' in df.columns:
            self._plot_time_series_sample(df)
    
    def _plot_signal_distributions(self, df: pd.DataFrame, features: list):
        """Plot distributions of critical signal metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, feat in enumerate(features[:6]):
            if feat in df.columns:
                ax = axes[idx]
                df[feat].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(f'{feat} Distribution', fontsize=12, fontweight='bold')
                ax.set_xlabel(feat)
                ax.set_ylabel('Frequency')
                ax.grid(alpha=0.3)
                
                # Add stats
                mean = df[feat].mean()
                std = df[feat].std()
                ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
                ax.legend()
        
        plt.tight_layout()
        plot_path = self.plots_dir / "signal_distributions.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(str(plot_path))
        plt.close()
    
    def _plot_anomaly_comparison(self, df: pd.DataFrame, features: list):
        """Plot comparison of features between normal and anomaly samples."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        normal = df[df['lab_anom'] == 0]
        anomaly = df[df['lab_anom'] == 1]
        
        for idx, feat in enumerate(features[:4]):
            if feat in df.columns:
                ax = axes[idx]
                
                # Box plot
                data_to_plot = [normal[feat].dropna(), anomaly[feat].dropna()]
                bp = ax.boxplot(data_to_plot, labels=['Normal', 'Anomaly'], patch_artist=True)
                
                # Color boxes
                bp['boxes'][0].set_facecolor('lightgreen')
                bp['boxes'][1].set_facecolor('lightcoral')
                
                ax.set_title(f'{feat}: Normal vs Anomaly', fontsize=12, fontweight='bold')
                ax.set_ylabel(feat)
                ax.grid(alpha=0.3)
                
                # Add sample sizes
                ax.text(1, ax.get_ylim()[1], f'n={len(normal)}', ha='center', fontsize=9)
                ax.text(2, ax.get_ylim()[1], f'n={len(anomaly)}', ha='center', fontsize=9)
        
        plt.suptitle('Feature Comparison: Normal vs Anomaly', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = self.plots_dir / "anomaly_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(str(plot_path))
        plt.close()
    
    def _plot_correlation_heatmap(self, df: pd.DataFrame, features: list):
        """Plot correlation heatmap of critical features."""
        # Select only numeric columns
        numeric_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        
        if len(numeric_features) < 2:
            return
        
        corr_matrix = df[numeric_features].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = self.plots_dir / "correlation_heatmap.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(str(plot_path))
        plt.close()
    
    def _plot_time_series_sample(self, df: pd.DataFrame):
        """Plot time series sample of key metrics."""
        # Sample 1000 consecutive points for visualization
        sample_size = min(1000, len(df))
        df_sample = df.head(sample_size).copy()
        
        if 'Time' in df_sample.columns:
            try:
                df_sample['Time'] = pd.to_datetime(df_sample['Time'])
                df_sample = df_sample.sort_values('Time')
            except:
                pass
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot SINR
        if 'SINR' in df_sample.columns:
            axes[0].plot(df_sample.index, df_sample['SINR'], linewidth=0.8, alpha=0.7)
            axes[0].set_ylabel('SINR (dB)', fontweight='bold')
            axes[0].set_title('SINR Time Series', fontweight='bold')
            axes[0].grid(alpha=0.3)
            axes[0].axhline(10, color='red', linestyle='--', label='Anomaly threshold')
            axes[0].legend()
        
        # Plot RSRP
        if 'RSRP' in df_sample.columns:
            axes[1].plot(df_sample.index, df_sample['RSRP'], linewidth=0.8, alpha=0.7, color='orange')
            axes[1].set_ylabel('RSRP (dBm)', fontweight='bold')
            axes[1].set_title('RSRP Time Series', fontweight='bold')
            axes[1].grid(alpha=0.3)
        
        # Plot throughput
        if 'throughput_DL' in df_sample.columns:
            axes[2].plot(df_sample.index, df_sample['throughput_DL'], linewidth=0.8, alpha=0.7, color='green')
            axes[2].set_ylabel('Throughput (Mbps)', fontweight='bold')
            axes[2].set_xlabel('Sample Index', fontweight='bold')
            axes[2].set_title('Throughput Time Series', fontweight='bold')
            axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.plots_dir / "time_series_sample.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(str(plot_path))
        plt.close()
    
    def _log_validation_artifact(self, validation_results: Dict[str, Any]):
        """Log validation results as JSON artifact."""
        artifact_path = self.plots_dir / "validation_results.json"

        with open(artifact_path, 'w') as f:
            json.dump(validation_results, f, indent=2, cls=NumpyEncoder)

        mlflow.log_artifact(str(artifact_path))
    
    def _log_data_profile(self, df: pd.DataFrame):
        """Log data profile summary."""
        profile = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            profile[f'{col}_stats'] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'null_pct': float(df[col].isnull().sum() / len(df) * 100)
            }
        
        profile_path = self.plots_dir / "data_profile.json"
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2, cls=NumpyEncoder)

        mlflow.log_artifact(str(profile_path))


def main():
    """Main execution function."""
    import sys
    from validate_data import WatchtowerValidator
    
    # Default data path
    data_path = "data/raw/sutd"
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    # Run validation
    validator = WatchtowerValidator()
    success, validation_results = validator.run_validation(data_path, save_report=True)
    
    # Load data for MLflow logging
    df = validator.load_data(data_path)
    
    # Get SUTD commit if available
    sutd_commit = None
    version_file = Path("data/SUTD_VERSION.txt")
    if version_file.exists():
        sutd_commit = version_file.read_text().strip()
    
    # Log to MLflow
    logger = MLflowLogger()
    run_id = logger.log_data_validation(df, validation_results, sutd_commit)
    
    print("\n" + "="*80)
    print("✅ Validation and MLflow logging complete!")
    print(f"   Run ID: {run_id}")
    print(f"   Start MLflow UI: mlflow ui --backend-store-uri mlruns")
    print("="*80)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
