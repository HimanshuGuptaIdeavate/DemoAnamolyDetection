"""
WATCHTOWER - Windowing and Feature Engineering Pipeline
Transforms clean 2Hz telemetry data into 5-second aggregated windows with derived features

This module implements:
1. Time-based windowing (5-second non-overlapping windows)
2. Statistical aggregations (mean, std, min, max, mode)
3. Derived features (deltas, ratios)
4. Quality metrics (gap ratio, window samples)
5. MLflow logging and validation

Usage:
    from windowing_pipeline import WindowingPipeline
    
    pipeline = WindowingPipeline()
    windows_df, stats = pipeline.run_full_pipeline("data/parquet/clean_data.parquet")
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# Import configurations
from watchtower.data.windowing_config import (
    WINDOW_SEC,
    SAMPLE_RATE_HZ,
    SAMPLES_PER_WINDOW,
    MIN_SAMPLES_THRESHOLD,
    WINDOW_AGGREGATIONS,
    AGG_FUNCTION_MAP,
    DELTA_FEATURES,
    DELTA_PREFIX,
    RATIO_FEATURES,
    LABEL_COLUMNS,
    WEAK_LABEL_NAME,
    OUTPUT_PATHS,
    OUTPUT_COLUMNS_ORDER,
    MLFLOW_CONFIG,
    MLFLOW_PARAMETERS,
    VALIDATION_THRESHOLDS,
    get_output_path,
)


class WindowingPipeline:
    """
    Time-series windowing and feature engineering pipeline.
    
    Transforms raw 2Hz samples into 5-second aggregated windows with:
    - Statistical aggregations (mean, std, min, max, mode)
    - Temporal features (deltas between consecutive windows)
    - Ratio features (signal quality indices)
    - Quality metrics (gap ratio, sample counts)
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize windowing pipeline.
        
        Args:
            verbose: Print detailed progress information
        """
        self.verbose = verbose
        self.stats = {}
        
        # Create output directories
        for path in OUTPUT_PATHS.values():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """Print log message if verbose"""
        if self.verbose:
            prefix = {
                "INFO": "ℹ️ ",
                "SUCCESS": "✅",
                "WARNING": "⚠️ ",
                "ERROR": "❌",
            }.get(level, "")
            print(f"{prefix} {message}")
    
    # ==========================================================================
    # STEP 1: Load Clean Data
    # ==========================================================================
    
    def load_clean_data(self, input_path: str) -> pd.DataFrame:
        """
        Load cleaned data from parquet file.
        
        Args:
            input_path: Path to clean_data.parquet
            
        Returns:
            DataFrame with clean telemetry data
        """
        self.log("\n" + "="*80)
        self.log("STEP 1: Load Clean Data", "INFO")
        self.log("="*80)
        
        df = pd.read_parquet(input_path)
        
        self.log(f"Loaded: {len(df):,} rows × {len(df.columns)} columns")
        self.log(f"Time range: {df['ts_ns'].min()} to {df['ts_ns'].max()}")
        self.log(f"Scenarios: {df['scenario_id'].nunique()}")
        
        # Verify required columns exist
        required_cols = ['ts_ns', 'scenario_id', 'sinr_db', 'rsrp_dbm']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by scenario and timestamp (critical for windowing!)
        df = df.sort_values(['scenario_id', 'ts_ns']).reset_index(drop=True)
        
        self.stats['input_rows'] = len(df)
        self.stats['input_scenarios'] = df['scenario_id'].nunique()
        
        return df
    
    # ==========================================================================
    # STEP 2: Create Time Windows
    # ==========================================================================
    
    def create_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 5-second non-overlapping time windows.
        
        Args:
            df: Clean telemetry data sorted by scenario_id, ts_ns
            
        Returns:
            DataFrame with one row per window
        """
        self.log("\n" + "="*80)
        self.log("STEP 2: Create Time Windows", "INFO")
        self.log("="*80)
        self.log(f"Window size: {WINDOW_SEC} seconds")
        self.log(f"Expected samples per window: {SAMPLES_PER_WINDOW}")
        
        def process_scenario_windows(scenario_df):
            """Process windows for a single scenario"""
            n_samples = len(scenario_df)
            n_windows = n_samples // SAMPLES_PER_WINDOW
            
            windows = []
            for i in range(n_windows):
                # Extract window segment
                start_idx = i * SAMPLES_PER_WINDOW
                end_idx = start_idx + SAMPLES_PER_WINDOW
                segment = scenario_df.iloc[start_idx:end_idx]
                
                # Create window dict
                window = self._aggregate_window(segment)
                windows.append(window)
            
            return pd.DataFrame(windows)
        
        # Process each scenario separately
        scenario_windows = []
        for scenario_id, scenario_df in df.groupby('scenario_id', sort=False):
            scenario_result = process_scenario_windows(scenario_df)
            
            if len(scenario_result) > 0:
                scenario_windows.append(scenario_result)
                self.log(f"  ✓ {scenario_id}: {len(scenario_result)} windows created")
            else:
                self.log(f"  ⚠️  {scenario_id}: No complete windows (insufficient data)", "WARNING")
        
        # Concatenate all scenarios
        windows_df = pd.concat(scenario_windows, ignore_index=True)
        
        # Sort by scenario and timestamp
        windows_df = windows_df.sort_values(['scenario_id', 'ts_start_ns']).reset_index(drop=True)
        
        self.log(f"\n✅ Created {len(windows_df):,} windows from {self.stats['input_rows']:,} samples")
        self.log(f"   Window creation rate: {len(windows_df) / (self.stats['input_rows'] / SAMPLES_PER_WINDOW) * 100:.1f}%")
        
        self.stats['output_windows'] = len(windows_df)
        self.stats['window_creation_rate'] = len(windows_df) / (self.stats['input_rows'] / SAMPLES_PER_WINDOW)
        
        return windows_df
    
    def _aggregate_window(self, segment: pd.DataFrame) -> Dict[str, Any]:
        """
        Aggregate a single window segment.
        
        Args:
            segment: DataFrame slice for one window (typically 10 rows)
            
        Returns:
            Dictionary with aggregated window features
        """
        window = {}
        
        # Window metadata
        window['ts_start_ns'] = int(segment['ts_ns'].iloc[0])
        window['scenario_id'] = segment['scenario_id'].iloc[0]
        window['window_samples'] = len(segment)
        window['gap_ratio'] = float(1.0 - len(segment) / SAMPLES_PER_WINDOW)
        
        # Apply aggregations
        for col, agg_funcs in WINDOW_AGGREGATIONS.items():
            if col not in segment.columns:
                continue
            
            for agg_func_name in agg_funcs:
                agg_func = AGG_FUNCTION_MAP.get(agg_func_name)
                if agg_func:
                    col_name = f"{col.replace('_dbm', '').replace('_db', '').replace('_mbps', '')}_{agg_func_name}"
                    window[col_name] = agg_func(segment[col].dropna())
        
        # Aggregate labels (weak label = any label in window)
        if all(col in segment.columns for col in LABEL_COLUMNS):
            window[WEAK_LABEL_NAME] = int(segment[LABEL_COLUMNS].any(axis=1).any())
        else:
            window[WEAK_LABEL_NAME] = 0
        
        return window
    
    # ==========================================================================
    # STEP 3: Add Derived Features
    # ==========================================================================
    
    def add_derived_features(self, windows_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features: deltas and ratios.
        
        Args:
            windows_df: DataFrame with window aggregations
            
        Returns:
            DataFrame with additional derived features
        """
        self.log("\n" + "="*80)
        self.log("STEP 3: Add Derived Features", "INFO")
        self.log("="*80)
        
        df = windows_df.copy()
        
        # Add delta features (first differences)
        self.log("\nAdding delta features (temporal changes):")
        for feature in DELTA_FEATURES:
            if feature in df.columns:
                delta_col = f"{DELTA_PREFIX}{feature}"
                df[delta_col] = df.groupby('scenario_id')[feature].diff().fillna(0.0)
                self.log(f"  ✓ {delta_col}: Change in {feature}")
        
        # Add ratio features
        self.log("\nAdding ratio features (signal quality indices):")
        for ratio_name, ratio_spec in RATIO_FEATURES.items():
            if 'formula' in ratio_spec:
                # Custom formula
                df[ratio_name] = ratio_spec['formula'](df)
                self.log(f"  ✓ {ratio_name}: {ratio_spec['description']}")
            else:
                # Simple ratio: numerator / denominator
                num_col = ratio_spec['numerator']
                denom_col = ratio_spec['denominator']
                
                if num_col in df.columns and denom_col in df.columns:
                    # Avoid division by zero
                    df[ratio_name] = df[num_col] / (df[denom_col].abs() + 1e-6)
                    df[ratio_name] = df[ratio_name].replace([np.inf, -np.inf], 0.0)
                    self.log(f"  ✓ {ratio_name}: {ratio_spec['description']}")
        
        self.log(f"\n✅ Added {len(DELTA_FEATURES)} delta features + {len(RATIO_FEATURES)} ratio features")
        
        self.stats['delta_features'] = len(DELTA_FEATURES)
        self.stats['ratio_features'] = len(RATIO_FEATURES)
        
        return df
    
    # ==========================================================================
    # STEP 4: Validation and Quality Checks
    # ==========================================================================
    
    def validate_windows(self, windows_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate windowing output quality.
        
        Args:
            windows_df: DataFrame with windowed features
            
        Returns:
            Dictionary with validation results
        """
        self.log("\n" + "="*80)
        self.log("STEP 4: Validation and Quality Checks", "INFO")
        self.log("="*80)
        
        validation = {}
        
        # Check 1: Minimum windows
        n_windows = len(windows_df)
        validation['n_windows'] = n_windows
        if n_windows < VALIDATION_THRESHOLDS['min_windows']:
            self.log(f"⚠️  Warning: Only {n_windows} windows (threshold: {VALIDATION_THRESHOLDS['min_windows']})", "WARNING")
            validation['min_windows_check'] = False
        else:
            self.log(f"✓ Window count: {n_windows:,} windows")
            validation['min_windows_check'] = True
        
        # Check 2: Gap ratio (missing data)
        gap_ratio_mean = windows_df['gap_ratio'].mean()
        gap_ratio_max = windows_df['gap_ratio'].max()
        validation['gap_ratio_mean'] = float(gap_ratio_mean)
        validation['gap_ratio_max'] = float(gap_ratio_max)
        
        if gap_ratio_mean > VALIDATION_THRESHOLDS['max_gap_ratio']:
            self.log(f"⚠️  Warning: High gap ratio {gap_ratio_mean:.2%} (threshold: {VALIDATION_THRESHOLDS['max_gap_ratio']:.0%})", "WARNING")
            validation['gap_ratio_check'] = False
        else:
            self.log(f"✓ Gap ratio: {gap_ratio_mean:.2%} (avg), {gap_ratio_max:.2%} (max)")
            validation['gap_ratio_check'] = True
        
        # Check 3: Label distribution
        weak_label_counts = windows_df[WEAK_LABEL_NAME].value_counts()
        weak_label_rate = weak_label_counts.get(1, 0) / len(windows_df) if len(windows_df) > 0 else 0
        validation['weak_label_rate'] = float(weak_label_rate)
        validation['weak_label_counts'] = {int(k): int(v) for k, v in weak_label_counts.items()}
        
        if weak_label_rate < VALIDATION_THRESHOLDS['min_weak_label_rate']:
            self.log(f"⚠️  Warning: Low anomaly rate {weak_label_rate:.2%} (threshold: {VALIDATION_THRESHOLDS['min_weak_label_rate']:.0%})", "WARNING")
        elif weak_label_rate > VALIDATION_THRESHOLDS['max_weak_label_rate']:
            self.log(f"⚠️  Warning: High anomaly rate {weak_label_rate:.2%} (threshold: {VALIDATION_THRESHOLDS['max_weak_label_rate']:.0%})", "WARNING")
        else:
            self.log(f"✓ Anomaly rate: {weak_label_rate:.2%}")
        
        # Check 4: Feature statistics
        self.log(f"\nFeature Statistics:")
        self.log(f"  SINR mean: {windows_df['sinr_mean'].mean():.2f} dB")
        self.log(f"  SINR std: {windows_df['sinr_std'].mean():.2f} dB")
        self.log(f"  Throughput mean: {windows_df['app_dl_mean'].mean():.2f} Mbps")
        
        validation['feature_stats'] = {
            'sinr_mean_avg': float(windows_df['sinr_mean'].mean()),
            'sinr_std_avg': float(windows_df['sinr_std'].mean()),
            'throughput_mean_avg': float(windows_df['app_dl_mean'].mean()),
        }
        
        self.log(f"\n✅ Validation complete")
        
        return validation
    
    # ==========================================================================
    # STEP 5: Export and Logging
    # ==========================================================================
    
    def export_windows(self, windows_df: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """
        Export windows to Parquet format.
        
        Args:
            windows_df: DataFrame with windowed features
            output_path: Optional custom output path
            
        Returns:
            Path to exported file
        """
        self.log("\n" + "="*80)
        self.log("STEP 5: Export Windows to Parquet", "INFO")
        self.log("="*80)

        if output_path is None:
            output_path = get_output_path('windows_parquet')

        # Generate CSV preview before exporting to parquet
        csv_output_path = output_path.replace('.parquet', '.csv')
        windows_df.to_csv(csv_output_path, index=False)
        self.log(f"  ✓ CSV preview saved to: {csv_output_path}")
        
        # Select columns in preferred order (only those that exist)
        available_cols = [col for col in OUTPUT_COLUMNS_ORDER if col in windows_df.columns]
        other_cols = [col for col in windows_df.columns if col not in available_cols]
        final_cols = available_cols + other_cols
        
        windows_df[final_cols].to_parquet(output_path, index=False, compression='snappy')
        
        file_size_mb = Path(output_path).stat().st_size / 1024**2
        
        self.log(f"  ✓ Saved to: {output_path}")
        self.log(f"  ✓ File size: {file_size_mb:.2f} MB")
        self.log(f"  ✓ Rows: {len(windows_df):,}")
        self.log(f"  ✓ Columns: {len(final_cols)}")
        
        self.stats['output_file'] = output_path
        self.stats['output_size_mb'] = file_size_mb
        self.stats['output_columns'] = len(final_cols)
        
        return output_path
    
    def save_metrics(self, windows_df: pd.DataFrame, validation: Dict[str, Any]):
        """Save windowing metrics to JSON."""
        metrics_path = get_output_path('metrics_json')
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'input_rows': self.stats['input_rows'],
            'output_windows': self.stats['output_windows'],
            'window_sec': WINDOW_SEC,
            'samples_per_window': SAMPLES_PER_WINDOW,
            'scenarios': self.stats['input_scenarios'],
            'gap_ratio_mean': float(windows_df['gap_ratio'].mean()),
            'gap_ratio_max': float(windows_df['gap_ratio'].max()),
            'weak_label_counts': validation['weak_label_counts'],
            'weak_label_rate': validation['weak_label_rate'],
            'feature_stats': validation['feature_stats'],
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.log(f"  ✓ Metrics saved: {metrics_path}")
    
    def create_visualizations(self, windows_df: pd.DataFrame):
        """Create preview and distribution plots."""
        self.log("\nCreating visualizations:")
        
        # Plot 1: SINR time series
        preview_path = get_output_path('preview_plot')
        
        plt.figure(figsize=(12, 5))
        
        # Normalize timestamp to seconds from start
        ts_seconds = (windows_df['ts_start_ns'] - windows_df['ts_start_ns'].iloc[0]) / 1e9
        
        plt.plot(ts_seconds, windows_df['sinr_mean'], alpha=0.8, linewidth=1, label='SINR mean')
        plt.fill_between(
            ts_seconds,
            windows_df['sinr_mean'] - windows_df['sinr_std'],
            windows_df['sinr_mean'] + windows_df['sinr_std'],
            alpha=0.3,
            label='±1 std'
        )
        
        plt.title('SINR Mean - 5-Second Windows', fontsize=14, fontweight='bold')
        plt.xlabel('Seconds from Start', fontsize=12)
        plt.ylabel('SINR (dB)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(preview_path, dpi=140)
        plt.close()
        
        self.log(f"  ✓ Preview plot: {preview_path}")
        
        # Plot 2: Feature distributions
        dist_path = get_output_path('distribution_plot')
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # SINR mean
        axes[0, 0].hist(windows_df['sinr_mean'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('SINR Mean Distribution')
        axes[0, 0].set_xlabel('SINR (dB)')
        
        # SINR std
        axes[0, 1].hist(windows_df['sinr_std'], bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('SINR Std Distribution')
        axes[0, 1].set_xlabel('SINR Std (dB)')
        
        # Throughput mean
        axes[0, 2].hist(windows_df['app_dl_mean'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 2].set_title('Throughput Mean Distribution')
        axes[0, 2].set_xlabel('Throughput (Mbps)')
        
        # SINR delta
        if 'd_sinr_mean' in windows_df.columns:
            axes[1, 0].hist(windows_df['d_sinr_mean'], bins=50, alpha=0.7, color='red', edgecolor='black')
            axes[1, 0].set_title('SINR Delta Distribution')
            axes[1, 0].set_xlabel('SINR Change (dB)')
        
        # Gap ratio
        axes[1, 1].hist(windows_df['gap_ratio'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_title('Gap Ratio Distribution')
        axes[1, 1].set_xlabel('Gap Ratio')
        
        # Weak label
        label_counts = windows_df[WEAK_LABEL_NAME].value_counts()
        axes[1, 2].bar(label_counts.index, label_counts.values, alpha=0.7, color=['green', 'red'])
        axes[1, 2].set_title('Weak Label Distribution')
        axes[1, 2].set_xlabel('Weak Label')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_xticks([0, 1])
        axes[1, 2].set_xticklabels(['Normal', 'Anomaly'])
        
        plt.tight_layout()
        plt.savefig(dist_path, dpi=140)
        plt.close()
        
        self.log(f"  ✓ Distribution plot: {dist_path}")
    
    def log_to_mlflow(self, windows_df: pd.DataFrame, validation: Dict[str, Any]):
        """Log windowing run to MLflow."""
        try:
            import mlflow
            
            mlflow.set_tracking_uri(MLFLOW_CONFIG['tracking_uri'])
            mlflow.set_experiment(MLFLOW_CONFIG['experiment_name'])
            
            run_name = f"{MLFLOW_CONFIG['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name):
                # Log parameters
                mlflow.log_params(MLFLOW_PARAMETERS)
                
                # Log metrics
                mlflow.log_metrics({
                    'input_rows': self.stats['input_rows'],
                    'output_windows': self.stats['output_windows'],
                    'gap_ratio_mean': validation['gap_ratio_mean'],
                    'weak_label_rate': validation['weak_label_rate'],
                    'output_size_mb': self.stats['output_size_mb'],
                })
                
                # Log artifacts
                mlflow.log_artifact(self.stats['output_file'])
                mlflow.log_artifact(get_output_path('metrics_json'))
                mlflow.log_artifact(get_output_path('preview_plot'))
                mlflow.log_artifact(get_output_path('distribution_plot'))
                
                self.log("✅ Logged to MLflow", "SUCCESS")
                
        except Exception as e:
            self.log(f"⚠️  MLflow logging failed: {e}", "WARNING")
    
    # ==========================================================================
    # MAIN PIPELINE
    # ==========================================================================
    
    def run_full_pipeline(
        self,
        input_path: str,
        export: bool = True,
        visualize: bool = True,
        log_mlflow: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run complete windowing and feature engineering pipeline.
        
        Args:
            input_path: Path to clean_data.parquet
            export: Export to parquet
            visualize: Create visualization plots
            log_mlflow: Log to MLflow
            
        Returns:
            Tuple of (windows_dataframe, statistics)
        """
        self.log("\n" + "="*80)
        self.log("WATCHTOWER WINDOWING PIPELINE")
        self.log("5-Second Windows + Derived Features")
        self.log("="*80)
        self.log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = datetime.now()
        
        # Step 1: Load
        df = self.load_clean_data(input_path)
        
        # Step 2: Window
        windows_df = self.create_windows(df)
        
        # Step 3: Derive
        windows_df = self.add_derived_features(windows_df)
        
        # Step 4: Validate
        validation = self.validate_windows(windows_df)
        
        # Step 5: Export
        if export:
            output_path = self.export_windows(windows_df)
            self.save_metrics(windows_df, validation)
        
        # Visualize
        if visualize:
            self.create_visualizations(windows_df)
        
        # MLflow
        if log_mlflow:
            self.log_to_mlflow(windows_df, validation)
        
        # Track duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.stats['duration_seconds'] = duration
        
        # Final summary
        self.log("\n" + "="*80)
        self.log("WINDOWING PIPELINE COMPLETE", "SUCCESS")
        self.log("="*80)
        self.log(f"Duration: {duration:.2f} seconds")
        self.log(f"Input:  {self.stats['input_rows']:,} samples")
        self.log(f"Output: {self.stats['output_windows']:,} windows")
        self.log(f"Anomaly rate: {validation['weak_label_rate']:.1%}")
        if export:
            self.log(f"Saved to: {self.stats['output_file']}")
        self.log("="*80)
        
        return windows_df, {**self.stats, **validation}


def main():
    """Main execution function for standalone use."""
    import sys
    
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/parquet/clean_data.parquet"
    
    pipeline = WindowingPipeline(verbose=True)
    windows_df, stats = pipeline.run_full_pipeline(input_path)
    
    print(f"\n✅ Windows shape: {windows_df.shape}")
    print(f"✅ Output file: {stats['output_file']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
