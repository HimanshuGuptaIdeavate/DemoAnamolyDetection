"""
WATCHTOWER - Data Curation and Cleaning Pipeline (ETL)
Production-grade ETL implementation for 5G telemetry data

This module implements the complete ETL workflow:
1. Extract: Load raw CSV files
2. Transform: Standardize columns, fix types, add timestamps, clip ranges
3. Validate: GX validation on cleaned data
4. Load: Export to Parquet with MLflow logging

Usage:
    from etl_pipeline import ETLPipeline
    
    etl = ETLPipeline()
    clean_df, stats = etl.run_full_pipeline("data/raw/sutd")
"""

import glob
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)


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

# Import configurations
from watchtower.data.etl_config import (
    COLUMN_RENAME_MAP,
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS_DEFAULTS,
    NUMERIC_COLUMNS,
    INTEGER_COLUMNS,
    FLOAT_COLUMNS,
    START_EPOCH_NS,
    TIMESTAMP_COLUMN,
    CLIPPING_RANGES,
    DROP_NULL_COLUMNS,
    FINAL_COLUMN_ORDER,
    OUTPUT_DIRS,
    OUTPUT_FILES,
    ETL_METRICS,
    VALIDATION_THRESHOLDS,
    get_output_path,
)


class ETLPipeline:
    """
    Complete ETL pipeline for 5G telemetry data curation.
    
    Features:
    - Column standardization (consistent snake_case names)
    - Type coercion (float32, Int64 nullable integers)
    - Absolute timestamp generation (nanosecond precision)
    - Range clipping (remove broken measurements)
    - Null handling (drop critical nulls, fill optionals)
    - Duplicate detection and removal
    - Comprehensive statistics tracking
    - MLflow logging integration
    - GX validation integration
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize ETL pipeline.
        
        Args:
            verbose: Print detailed progress information
        """
        self.verbose = verbose
        self.stats = {}
        self.clipping_stats = {}
        
        # Create output directories
        for dir_name, dir_path in OUTPUT_DIRS.items():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
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
    # STEP 1: EXTRACT - Load Raw CSV Files
    # ==========================================================================
    
    def extract_raw_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and concatenate all CSV files from data directory.
        
        Args:
            data_path: Path to directory containing CSV files
            
        Returns:
            Concatenated DataFrame with 'scenario_id' column added
        """
        self.log("\n" + "="*80)
        self.log("STEP 1: EXTRACT - Loading Raw CSV Files", "INFO")
        self.log("="*80)
        
        data_path = Path(data_path)
        
        # Find all CSV files
        if data_path.is_file():
            csv_files = [data_path]
        else:
            csv_files = sorted(data_path.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        
        self.log(f"Found {len(csv_files)} CSV file(s)")
        
        # Load each file
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, on_bad_lines='skip', low_memory=False)
                
                # Add scenario identifier from filename
                df['scenario_id'] = csv_file.stem
                
                dataframes.append(df)
                self.log(f"  ✓ Loaded {csv_file.name}: {len(df):,} rows")
                
            except Exception as e:
                self.log(f"  ✗ Failed to load {csv_file.name}: {e}", "ERROR")
                continue
        
        if not dataframes:
            raise ValueError("No data loaded successfully")
        
        # Concatenate all dataframes
        raw_df = pd.concat(dataframes, ignore_index=True)
        
        self.log(f"\n✅ Loaded {len(raw_df):,} total rows from {len(csv_files)} files")
        
        # Track statistics
        self.stats['input_rows'] = len(raw_df)
        self.stats['input_columns'] = len(raw_df.columns)
        self.stats['num_files'] = len(csv_files)
        
        return raw_df
    
    # ==========================================================================
    # STEP 2: TRANSFORM - Standardize, Clean, Normalize
    # ==========================================================================
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to consistent snake_case format.
        
        Args:
            df: Input DataFrame with raw column names
            
        Returns:
            DataFrame with renamed columns
        """
        self.log("\n" + "="*80)
        self.log("STEP 2A: TRANSFORM - Standardize Column Names", "INFO")
        self.log("="*80)
        
        df_renamed = df.copy()
        
        # Apply column renaming
        columns_renamed = []
        for old_name, new_name in COLUMN_RENAME_MAP.items():
            if old_name in df_renamed.columns:
                df_renamed.rename(columns={old_name: new_name}, inplace=True)
                columns_renamed.append(f"{old_name} → {new_name}")
        
        self.log(f"Renamed {len(columns_renamed)} columns:")
        for rename in columns_renamed[:10]:  # Show first 10
            self.log(f"  • {rename}")
        if len(columns_renamed) > 10:
            self.log(f"  ... and {len(columns_renamed) - 10} more")
        
        # Add missing optional columns with defaults
        for col, default_value in OPTIONAL_COLUMNS_DEFAULTS.items():
            if col not in df_renamed.columns:
                df_renamed[col] = default_value
                self.log(f"  ➕ Added missing column: {col} = {default_value}")
        
        # Check for required columns
        missing_required = [col for col in REQUIRED_COLUMNS if col not in df_renamed.columns]
        if missing_required:
            self.log(f"Missing required columns: {missing_required}", "ERROR")
            raise ValueError(f"Missing required columns: {missing_required}")
        
        self.log(f"\n✅ Column standardization complete")
        self.stats['columns_renamed'] = len(columns_renamed)
        
        return df_renamed
    
    def coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerce columns to appropriate data types.

        Args:
            df: DataFrame with standardized column names

        Returns:
            DataFrame with corrected types
        """
        self.log("\n" + "="*80)
        self.log("STEP 2B: TRANSFORM - Type Coercion", "INFO")
        self.log("="*80)

        df_typed = df.copy()

        # Special handling for time_s column - may contain datetime strings
        if 'time_s' in df_typed.columns:
            # Check if it looks like datetime strings (not numeric)
            sample_value = df_typed['time_s'].dropna().iloc[0] if len(df_typed['time_s'].dropna()) > 0 else None
            if sample_value is not None and isinstance(sample_value, str) and '-' in sample_value:
                self.log(f"  📅 Detected datetime format in time_s column, parsing...")
                # Parse datetime strings
                df_typed['time_s'] = pd.to_datetime(df_typed['time_s'], errors='coerce')
                # Convert to Unix timestamp in seconds
                df_typed['time_s'] = df_typed['time_s'].astype('int64') // 10**9
                self.log(f"  ✓ Converted datetime strings to Unix timestamps")

        # Convert numeric columns
        coerced_count = 0
        for col in NUMERIC_COLUMNS:
            if col in df_typed.columns:
                # Skip time_s if already handled above
                if col == 'time_s' and df_typed[col].dtype in ['int64', 'float64']:
                    coerced_count += 1
                    continue

                original_dtype = df_typed[col].dtype
                df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce')

                nulls_introduced = df_typed[col].isnull().sum() - df[col].isnull().sum()
                if nulls_introduced > 0:
                    self.log(f"  ⚠️  {col}: {nulls_introduced} invalid values → NaN")

                coerced_count += 1
        
        # Convert to appropriate dtypes for memory efficiency
        for col in INTEGER_COLUMNS:
            if col in df_typed.columns:
                # Round first (in case of float values like 27.9)
                df_typed[col] = df_typed[col].round()
                # Use Int64 (nullable integer type)
                df_typed[col] = df_typed[col].astype('Int64')
        
        for col in FLOAT_COLUMNS:
            if col in df_typed.columns:
                df_typed[col] = df_typed[col].astype('float32')
        
        self.log(f"✅ Coerced {coerced_count} numeric columns")
        self.stats['columns_type_coerced'] = coerced_count
        
        return df_typed
    
    def add_absolute_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add absolute timestamp in nanoseconds.
        
        The dataset uses relative time (seconds since start).
        We convert to absolute nanosecond timestamps for:
        - Consistent time representation
        - Monotonic ordering
        - Compatibility with production PM logs
        
        Args:
            df: DataFrame with 'time_s' column
            
        Returns:
            DataFrame with 'ts_ns' column added
        """
        self.log("\n" + "="*80)
        self.log("STEP 2C: TRANSFORM - Add Absolute Timestamp", "INFO")
        self.log("="*80)
        
        df_timestamped = df.copy()

        if 'time_s' not in df_timestamped.columns:
            self.log("No 'time_s' column found - skipping timestamp generation", "WARNING")
            return df_timestamped

        # Handle NA/NaN values in time_s before conversion
        na_count = df_timestamped['time_s'].isna().sum()
        if na_count > 0:
            self.log(f"Found {na_count} NA values in 'time_s' column - filling with 0", "WARNING")
            df_timestamped['time_s'] = df_timestamped['time_s'].fillna(0)

        # Convert relative time to absolute nanosecond timestamp
        df_timestamped[TIMESTAMP_COLUMN] = (
            START_EPOCH_NS + (df_timestamped['time_s'] * 1e9).astype('int64')
        )
        
        # Statistics
        time_range = df_timestamped['time_s'].max() - df_timestamped['time_s'].min()
        num_records = len(df_timestamped)
        
        self.log(f"Time range: {time_range:.2f} seconds")
        self.log(f"Number of records: {num_records:,}")
        self.log(f"Average sampling rate: {num_records / time_range:.2f} Hz")
        self.log(f"Timestamp range: {df_timestamped[TIMESTAMP_COLUMN].min()} to {df_timestamped[TIMESTAMP_COLUMN].max()}")
        
        self.log(f"\n✅ Absolute timestamps added ({TIMESTAMP_COLUMN})")
        
        return df_timestamped
    
    def clip_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip values to realistic 3GPP bounds.
        
        Purpose: Remove broken measurements like:
        - RSRP = +100 dBm (impossible - signal can't be stronger than transmitter)
        - SINR = -500 dB (hardware measurement floor ~-20 dB)
        - Throughput > 100 Mbps (corrupt iPerf3 logs in testbed)
        
        Args:
            df: DataFrame with numeric columns
            
        Returns:
            DataFrame with clipped values
        """
        self.log("\n" + "="*80)
        self.log("STEP 2D: TRANSFORM - Range Clipping (Sanity Checks)", "INFO")
        self.log("="*80)
        
        df_clipped = df.copy()
        total_clipped = 0
        
        for col, range_spec in CLIPPING_RANGES.items():
            if col not in df_clipped.columns:
                continue
            
            clip_min = range_spec['min']
            clip_max = range_spec['max']
            reason = range_spec['reason']
            
            # Count values outside range
            below_min = (df_clipped[col] < clip_min).sum()
            above_max = (df_clipped[col] > clip_max).sum()
            total_col_clipped = below_min + above_max
            
            if total_col_clipped > 0:
                # Special handling for MCS (round before clipping)
                if 'mcs' in col:
                    df_clipped[col] = df_clipped[col].round()
                
                # Perform clipping
                df_clipped[col] = df_clipped[col].clip(lower=clip_min, upper=clip_max)
                
                # Track statistics
                self.clipping_stats[col] = {
                    'below_min': int(below_min),
                    'above_max': int(above_max),
                    'total_clipped': int(total_col_clipped),
                    'pct_clipped': (total_col_clipped / len(df_clipped)) * 100,
                    'clip_range': [clip_min, clip_max],
                    'reason': reason
                }
                
                total_clipped += total_col_clipped
                
                self.log(f"  ✂️  {col}: Clipped {total_col_clipped:,} values to [{clip_min}, {clip_max}]")
                self.log(f"      ({below_min:,} below, {above_max:,} above) - {reason}")
            else:
                self.log(f"  ✓ {col}: All values within [{clip_min}, {clip_max}]")
        
        self.log(f"\n✅ Clipped {total_clipped:,} values across {len(self.clipping_stats)} columns")
        self.stats['values_clipped'] = total_clipped
        self.stats['columns_clipped'] = len(self.clipping_stats)
        
        # Warning if too many values clipped
        pct_clipped = (total_clipped / len(df_clipped)) / len(CLIPPING_RANGES) * 100
        if pct_clipped > VALIDATION_THRESHOLDS['max_clipped_percentage']:
            self.log(f"⚠️  Warning: {pct_clipped:.2f}% of values clipped (threshold: {VALIDATION_THRESHOLDS['max_clipped_percentage']}%)", "WARNING")
        
        return df_clipped
    
    def handle_nulls_and_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle null values and remove duplicates using tiered strategy.
        
        Tiered Null Handling:
        - Tier 1: Drop rows with nulls in CRITICAL columns (~2% data loss)
        - Tier 2: Impute nulls in OPERATIONAL columns intelligently
        - Tier 3: Leave nulls in OPTIONAL columns (acceptable)
        
        Args:
            df: DataFrame with potential nulls and duplicates
            
        Returns:
            Cleaned DataFrame
        """
        self.log("\n" + "="*80)
        self.log("STEP 2E: TRANSFORM - Handle Nulls and Duplicates", "INFO")
        self.log("="*80)
        
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # TIER 1: Drop rows with nulls in critical columns
        self.log("\nTIER 1: Dropping rows with nulls in critical columns")
        null_counts_before = df_clean[DROP_NULL_COLUMNS].isnull().sum()
        df_clean.dropna(subset=DROP_NULL_COLUMNS, inplace=True)
        rows_dropped_nulls = initial_rows - len(df_clean)
        
        if rows_dropped_nulls > 0:
            pct_dropped = (rows_dropped_nulls / initial_rows) * 100
            self.log(f"  🗑️  Dropped {rows_dropped_nulls:,} rows ({pct_dropped:.2f}%) with critical nulls:")
            for col in DROP_NULL_COLUMNS:
                if null_counts_before[col] > 0:
                    self.log(f"      {col}: {null_counts_before[col]} nulls")
        else:
            self.log(f"  ✓ No rows dropped (no nulls in critical columns)")
        
        # TIER 2: Impute nulls in operational columns
        self.log("\nTIER 2: Imputing nulls in operational columns")
        
        from watchtower.data.etl_config import IMPUTE_STRATEGIES
        
        imputation_stats = {}
        for col, strategy in IMPUTE_STRATEGIES.items():
            if col not in df_clean.columns:
                continue
            
            nulls_before = df_clean[col].isnull().sum()
            if nulls_before == 0:
                continue
            
            method = strategy['method']
            reason = strategy['reason']
            
            # Apply imputation based on method
            if method == 'mode':
                fill_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 0
                df_clean[col].fillna(fill_value, inplace=True)
                
            elif method == 'median':
                fill_value = df_clean[col].median()
                df_clean[col].fillna(fill_value, inplace=True)
                
            elif method == 'ffill':
                # Forward fill requires data to be sorted by time
                # Group by scenario to avoid filling across different datasets
                if 'scenario_id' in df_clean.columns:
                    df_clean[col] = df_clean.groupby('scenario_id')[col].fillna(method='ffill')
                else:
                    df_clean[col].fillna(method='ffill', inplace=True)
                
                # If first rows still have nulls, back fill
                df_clean[col].fillna(method='bfill', inplace=True)
                fill_value = "forward fill"
            
            nulls_after = df_clean[col].isnull().sum()
            nulls_filled = nulls_before - nulls_after
            
            imputation_stats[col] = {
                'nulls_before': int(nulls_before),
                'nulls_filled': int(nulls_filled),
                'method': method,
                'fill_value': str(fill_value) if method != 'ffill' else 'forward fill',
                'reason': reason
            }
            
            self.log(f"  ✓ {col}: Filled {nulls_filled} nulls using {method}")
            self.log(f"      Reason: {reason}")
        
        # Track imputation statistics
        self.stats['imputation_stats'] = imputation_stats
        
        # TIER 3: Report on remaining nulls (acceptable columns)
        remaining_nulls = df_clean.isnull().sum()
        remaining_nulls = remaining_nulls[remaining_nulls > 0]
        
        if len(remaining_nulls) > 0:
            self.log("\nTIER 3: Remaining nulls (acceptable in optional columns)")
            for col, count in remaining_nulls.items():
                pct = (count / len(df_clean)) * 100
                self.log(f"  ℹ️  {col}: {count} nulls ({pct:.2f}%)")
        
        # Remove duplicates
        self.log("\nRemoving duplicate rows")
        rows_before_dedup = len(df_clean)
        df_clean.drop_duplicates(inplace=True)
        rows_dropped_duplicates = rows_before_dedup - len(df_clean)
        
        if rows_dropped_duplicates > 0:
            pct_duplicates = (rows_dropped_duplicates / rows_before_dedup) * 100
            self.log(f"  🗑️  Dropped {rows_dropped_duplicates:,} duplicate rows ({pct_duplicates:.2f}%)")
        else:
            self.log(f"  ✓ No duplicate rows found")
        
        # Final summary
        self.log(f"\n✅ Null handling complete:")
        self.log(f"   Rows: {initial_rows:,} → {len(df_clean):,} ({initial_rows - len(df_clean):,} dropped)")
        self.log(f"   Critical nulls: {rows_dropped_nulls} rows dropped")
        self.log(f"   Operational nulls: {sum(s['nulls_filled'] for s in imputation_stats.values())} values imputed")
        self.log(f"   Duplicates: {rows_dropped_duplicates} rows dropped")
        
        self.stats['rows_dropped_nulls'] = rows_dropped_nulls
        self.stats['rows_dropped_duplicates'] = rows_dropped_duplicates
        self.stats['values_imputed'] = sum(s['nulls_filled'] for s in imputation_stats.values())
        
        return df_clean
    
    def finalize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Finalize output: select columns, sort, reset index.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Final DataFrame ready for export
        """
        self.log("\n" + "="*80)
        self.log("STEP 2F: TRANSFORM - Finalize Output", "INFO")
        self.log("="*80)
        
        # Select only columns that exist
        available_columns = [col for col in FINAL_COLUMN_ORDER if col in df.columns]
        df_final = df[available_columns].copy()
        
        # Sort by scenario and timestamp
        if TIMESTAMP_COLUMN in df_final.columns and 'scenario_id' in df_final.columns:
            df_final.sort_values(['scenario_id', TIMESTAMP_COLUMN], inplace=True)
            self.log(f"  ✓ Sorted by scenario_id and {TIMESTAMP_COLUMN}")
        
        # Reset index
        df_final.reset_index(drop=True, inplace=True)
        
        self.log(f"  ✓ Selected {len(available_columns)} columns")
        self.log(f"  ✓ Final shape: {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")
        
        self.stats['output_rows'] = len(df_final)
        self.stats['output_columns'] = len(available_columns)
        
        return df_final
    
    # ==========================================================================
    # STEP 3: VALIDATE - GX Validation on Cleaned Data
    # ==========================================================================
    
    def validate_cleaned_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run Great Expectations validation on cleaned data.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Validation results dictionary
        """
        self.log("\n" + "="*80)
        self.log("STEP 3: VALIDATE - GX Validation on Cleaned Data", "INFO")
        self.log("="*80)
        
        try:
            from watchtower.data.validate_data import WatchtowerValidator

            # Save temporary CSV for validation
            temp_csv = get_output_path("temp_clean.csv", "interim")
            df.to_csv(temp_csv, index=False)
            
            # Run validation
            validator = WatchtowerValidator(output_dir="reports/validation")
            success, validation_results = validator.run_validation(temp_csv, save_report=True)
            
            if success:
                self.log("✅ All validation checks passed!", "SUCCESS")
            else:
                failed = validation_results['failed_checks']
                self.log(f"⚠️  {failed} validation checks failed", "WARNING")
            
            # Clean up temp file
            Path(temp_csv).unlink()
            
            return validation_results
            
        except ImportError:
            self.log("⚠️  Validation module not available - skipping GX validation", "WARNING")
            return {"success": None, "message": "Validation skipped"}
    
    # ==========================================================================
    # STEP 4: LOAD - Export to Parquet with MLflow Logging
    # ==========================================================================
    
    def export_to_parquet(self, df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        Export DataFrame to Parquet format.
        
        Args:
            df: DataFrame to export
            filename: Optional output filename
            
        Returns:
            Path to exported file
        """
        self.log("\n" + "="*80)
        self.log("STEP 4: LOAD - Export to Parquet", "INFO")
        self.log("="*80)
        
        if filename is None:
            filename = OUTPUT_FILES['clean_data']
        
        output_path = get_output_path(filename, 'parquet')
        
        # Export to parquet
        df.to_parquet(output_path, index=False, compression='snappy')
        
        file_size_mb = Path(output_path).stat().st_size / 1024**2
        
        self.log(f"  ✓ Saved to: {output_path}")
        self.log(f"  ✓ File size: {file_size_mb:.2f} MB")
        self.log(f"  ✓ Compression: snappy")
        
        self.stats['output_file'] = output_path
        self.stats['output_size_mb'] = file_size_mb
        
        return output_path
    
    def save_statistics(self):
        """Save ETL statistics to JSON file."""
        stats_path = get_output_path(OUTPUT_FILES['etl_report'], 'reports')
        
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2, cls=NumpyEncoder)

        self.log(f"  ✓ ETL statistics saved: {stats_path}")

        # Save clipping statistics
        clipping_path = get_output_path(OUTPUT_FILES['clipping_stats'], 'reports')
        with open(clipping_path, 'w') as f:
            json.dump(self.clipping_stats, f, indent=2, cls=NumpyEncoder)
        
        self.log(f"  ✓ Clipping statistics saved: {clipping_path}")
    
    def log_to_mlflow(self, df: pd.DataFrame):
        """
        Log ETL run to MLflow.
        
        Args:
            df: Final cleaned DataFrame
        """
        try:
            import mlflow
            
            mlflow.set_experiment("watchtower_etl")
            
            with mlflow.start_run(run_name=f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_params({
                    "start_epoch_ns": int(START_EPOCH_NS),
                    "num_files": self.stats.get('num_files', 0),
                })
                
                # Log metrics
                mlflow.log_metrics({
                    "input_rows": self.stats['input_rows'],
                    "output_rows": self.stats['output_rows'],
                    "rows_dropped": self.stats['input_rows'] - self.stats['output_rows'],
                    "columns_renamed": self.stats.get('columns_renamed', 0),
                    "values_clipped": self.stats.get('values_clipped', 0),
                    "output_size_mb": self.stats.get('output_size_mb', 0),
                })
                
                # Log artifacts
                mlflow.log_artifact(self.stats['output_file'])
                mlflow.log_artifact(get_output_path(OUTPUT_FILES['etl_report'], 'reports'))
                mlflow.log_artifact(get_output_path(OUTPUT_FILES['clipping_stats'], 'reports'))
                
                self.log("✅ Logged to MLflow", "SUCCESS")
                
        except Exception as e:
            self.log(f"⚠️  MLflow logging failed: {e}", "WARNING")
    
    # ==========================================================================
    # MAIN PIPELINE
    # ==========================================================================
    
    def run_full_pipeline(
        self,
        data_path: str,
        validate: bool = True,
        export_parquet: bool = True,
        log_mlflow: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run complete ETL pipeline.
        
        Args:
            data_path: Path to raw CSV files
            validate: Run GX validation on cleaned data
            export_parquet: Export to Parquet format
            log_mlflow: Log to MLflow
            
        Returns:
            Tuple of (cleaned_dataframe, statistics)
        """
        self.log("\n" + "="*80)
        self.log("WATCHTOWER ETL PIPELINE - DATA CURATION AND CLEANING")
        self.log("="*80)
        self.log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Track start time
        start_time = datetime.now()
        
        # Step 1: Extract
        df = self.extract_raw_data(data_path)
        
        # Step 2: Transform
        df = self.standardize_columns(df)
        df = self.coerce_types(df)
        df = self.add_absolute_timestamp(df)
        df = self.clip_ranges(df)
        df = self.handle_nulls_and_duplicates(df)
        df = self.finalize_output(df)
        
        # Step 3: Validate
        if validate:
            validation_results = self.validate_cleaned_data(df)
            self.stats['validation'] = validation_results
        
        # Step 4: Load
        if export_parquet:
            output_path = self.export_to_parquet(df)
        
        # Save statistics
        self.save_statistics()
        
        # Log to MLflow
        if log_mlflow:
            self.log_to_mlflow(df)
        
        # Track end time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.stats['duration_seconds'] = duration
        self.stats['start_time'] = start_time.isoformat()
        self.stats['end_time'] = end_time.isoformat()
        
        # Final summary
        self.log("\n" + "="*80)
        self.log("ETL PIPELINE COMPLETE", "SUCCESS")
        self.log("="*80)
        self.log(f"Duration: {duration:.2f} seconds")
        self.log(f"Input:  {self.stats['input_rows']:,} rows")
        self.log(f"Output: {self.stats['output_rows']:,} rows")
        self.log(f"Dropped: {self.stats['input_rows'] - self.stats['output_rows']:,} rows")
        self.log(f"Clipped: {self.stats.get('values_clipped', 0):,} values")
        if export_parquet:
            self.log(f"Saved to: {self.stats['output_file']}")
        self.log("="*80)
        
        return df, self.stats


def main():
    """Main execution function for standalone use."""
    import sys
    
    # Get data path from command line or use default
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sutd"
    
    # Run ETL pipeline
    etl = ETLPipeline(verbose=True)
    clean_df, stats = etl.run_full_pipeline(data_path)
    
    print(f"\n✅ Clean data shape: {clean_df.shape}")
    print(f"✅ Output file: {stats['output_file']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
