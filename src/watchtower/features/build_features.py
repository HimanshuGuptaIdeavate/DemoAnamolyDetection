"""
WATCHTOWER - Feature Table Builder
Build feature table from windows.parquet

This module ONLY does:
1. Load windows.parquet
2. Sort by scenario + timestamp (required for derivatives)
3. Create new derived features
4. Select features in fixed order
5. Save features_table.parquet

NO train-test split, NO scaling, NO encoding - just feature construction!
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd

# Import configuration
from watchtower.features.build_features_config import (
    PATHS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    METADATA_COLUMNS,
    TARGET,
    ALL_NUMERIC_FEATURES,
    NEW_DERIVED_FEATURES,
    FINAL_COLUMN_ORDER,
    VALIDATION_THRESHOLDS,
    get_new_derived_features,
    get_final_column_order,
)


class FeatureTableBuilder:
    """
    Simple feature table builder.
    
    Transforms windows.parquet into features_table.parquet with:
    - Proper sorting (for derivatives)
    - New derived features
    - Fixed column ordering
    
    Does NOT do: splitting, scaling, encoding
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize feature table builder.
        
        Args:
            verbose: Print progress information
        """
        self.verbose = verbose
        self.stats = {}
    
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
    # STEP 1: Load Windows
    # ==========================================================================
    
    def load_windows(self, input_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load windows from parquet.
        
        Args:
            input_path: Optional custom input path
            
        Returns:
            DataFrame with windows
        """
        self.log("\n" + "="*80)
        self.log("STEP 1: Load Windows", "INFO")
        self.log("="*80)
        
        if input_path is None:
            input_path = PATHS['input_windows']
        
        df = pd.read_parquet(input_path)
        
        self.log(f"Loaded: {len(df):,} windows")
        self.log(f"Columns: {len(df.columns)}")
        self.log(f"Scenarios: {df['scenario_id'].nunique()}")
        
        self.stats['input_windows'] = len(df)
        self.stats['input_columns'] = len(df.columns)
        
        return df
    
    # ==========================================================================
    # STEP 2: Sort Data (CRITICAL for Derivatives!)
    # ==========================================================================
    
    def sort_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort by scenario and timestamp.
        
        This is CRITICAL because derivatives (diff) require correct ordering!
        
        Args:
            df: Input DataFrame
            
        Returns:
            Sorted DataFrame
        """
        self.log("\n" + "="*80)
        self.log("STEP 2: Sort Data (Required for Derivatives!)", "INFO")
        self.log("="*80)
        
        df_sorted = df.sort_values(['scenario_id', 'ts_start_ns']).reset_index(drop=True)
        
        self.log(f"Sorted by: ['scenario_id', 'ts_start_ns']")
        self.log(f"✓ Order is now correct for derivative calculations")
        
        return df_sorted
    
    # ==========================================================================
    # STEP 3: Create Derived Features
    # ==========================================================================
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new derived features.
        
        Args:
            df: Sorted DataFrame
            
        Returns:
            DataFrame with new features added
        """
        self.log("\n" + "="*80)
        self.log("STEP 3: Create Derived Features", "INFO")
        self.log("="*80)
        
        df_feat = df.copy()
        
        # Create each new feature
        new_features = get_new_derived_features()
        
        for feat_name, feat_spec in new_features.items():
            try:
                # Call the feature creation function
                df_feat[feat_name] = feat_spec['function'](df_feat)
                
                self.log(f"✓ Created: {feat_name}")
                self.log(f"    {feat_spec['description']}")
                
            except Exception as e:
                self.log(f"⚠️  Failed to create {feat_name}: {e}", "WARNING")
        
        # Replace infinities with NaN
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        
        self.log(f"\n✅ Created {len(new_features)} new derived features")
        
        self.stats['derived_features_created'] = len(new_features)
        
        return df_feat
    
    # ==========================================================================
    # STEP 4: Select Features in Fixed Order
    # ==========================================================================
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select features in fixed order.
        
        This creates the train-serve contract: columns MUST be in this order
        for both training and production inference.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            DataFrame with selected features in fixed order
        """
        self.log("\n" + "="*80)
        self.log("STEP 4: Select Features (Fixed Order)", "INFO")
        self.log("="*80)
        
        # Get fixed column order
        final_columns = get_final_column_order()
        
        # Check which columns exist
        available_columns = [col for col in final_columns if col in df.columns]
        missing_columns = [col for col in final_columns if col not in df.columns]
        
        if missing_columns:
            self.log(f"⚠️  Missing columns: {missing_columns}", "WARNING")
        
        # Select available columns in fixed order
        df_feat = df[available_columns].copy()
        
        self.log(f"Selected: {len(available_columns)}/{len(final_columns)} columns")
        self.log(f"  Metadata: {len([c for c in available_columns if c in METADATA_COLUMNS])}")
        self.log(f"  Numeric: {len([c for c in available_columns if c in ALL_NUMERIC_FEATURES])}")
        self.log(f"  Categorical: {len([c for c in available_columns if c in CATEGORICAL_FEATURES])}")
        self.log(f"  Target: {1 if TARGET in available_columns else 0}")
        
        self.stats['selected_columns'] = len(available_columns)
        self.stats['output_features'] = len(available_columns) - len(METADATA_COLUMNS) - 1  # Exclude metadata + target
        
        return df_feat
    
    # ==========================================================================
    # STEP 5: Validate Data Quality
    # ==========================================================================
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate feature table quality.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Validation results dictionary
        """
        self.log("\n" + "="*80)
        self.log("STEP 5: Validate Data Quality", "INFO")
        self.log("="*80)
        
        validation = {}
        
        # Check 1: Minimum windows
        n_windows = len(df)
        validation['n_windows'] = n_windows
        
        if n_windows < VALIDATION_THRESHOLDS['min_windows']:
            self.log(f"⚠️  Warning: Only {n_windows} windows (expected ≥{VALIDATION_THRESHOLDS['min_windows']})", "WARNING")
            validation['min_windows_check'] = False
        else:
            self.log(f"✓ Window count: {n_windows:,}")
            validation['min_windows_check'] = True
        
        # Check 2: Missing values
        missing_counts = df.isnull().sum()
        missing_features = missing_counts[missing_counts > 0]
        
        if len(missing_features) > 0:
            self.log(f"\nMissing values found in {len(missing_features)} features:")
            for feat, count in missing_features.items():
                pct = (count / len(df)) * 100
                self.log(f"  {feat}: {count} ({pct:.2f}%)")
                
                if pct > VALIDATION_THRESHOLDS['max_missing_ratio'] * 100:
                    self.log(f"    ⚠️  Exceeds threshold ({VALIDATION_THRESHOLDS['max_missing_ratio']*100:.0f}%)", "WARNING")
        else:
            self.log(f"✓ No missing values")
        
        validation['missing_features'] = len(missing_features)
        validation['total_missing'] = int(missing_counts.sum())
        
        # Check 3: Feature statistics
        numeric_cols = [col for col in df.columns if col in ALL_NUMERIC_FEATURES]
        
        self.log(f"\nFeature Statistics:")
        self.log(f"  SINR mean: {df['sinr_mean'].mean():.2f} dB")
        self.log(f"  SINR range: {df['sinr_range'].mean():.2f} dB")
        self.log(f"  Throughput: {df['app_dl_mean'].mean():.2f} Mbps")
        
        validation['feature_stats'] = {
            'sinr_mean_avg': float(df['sinr_mean'].mean()),
            'sinr_range_avg': float(df['sinr_range'].mean()),
            'throughput_avg': float(df['app_dl_mean'].mean()),
        }
        
        # Check 4: Target distribution
        target_counts = df[TARGET].value_counts()
        anomaly_rate = target_counts.get(1, 0) / len(df)
        
        self.log(f"\nTarget Distribution:")
        self.log(f"  Normal (0): {target_counts.get(0, 0):,} ({(1-anomaly_rate)*100:.1f}%)")
        self.log(f"  Anomaly (1): {target_counts.get(1, 0):,} ({anomaly_rate*100:.1f}%)")
        
        validation['anomaly_rate'] = float(anomaly_rate)
        validation['target_counts'] = {int(k): int(v) for k, v in target_counts.items()}
        
        self.log(f"\n✅ Validation complete")
        
        return validation
    
    # ==========================================================================
    # STEP 6: Save Feature Table
    # ==========================================================================
    
    def save_feature_table(
        self,
        df: pd.DataFrame,
        validation: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """
        Save feature table to parquet.
        
        Args:
            df: Feature DataFrame
            validation: Validation results
            output_path: Optional custom output path
            
        Returns:
            Path to saved file
        """
        self.log("\n" + "="*80)
        self.log("STEP 6: Save Feature Table", "INFO")
        self.log("="*80)
        
        if output_path is None:
            output_path = PATHS['output_features']
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        df.to_parquet(output_path, index=False)
        
        file_size_mb = Path(output_path).stat().st_size / 1024**2
        
        self.log(f"✓ Saved to: {output_path}")
        self.log(f"  Rows: {len(df):,}")
        self.log(f"  Columns: {len(df.columns)}")
        self.log(f"  Size: {file_size_mb:.2f} MB")
        
        # Save statistics
        stats_path = Path(PATHS['reports']) / 'feature_table_stats.json'
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats_report = {
            'timestamp': datetime.now().isoformat(),
            'input_windows': self.stats['input_windows'],
            'output_rows': len(df),
            'output_columns': len(df.columns),
            'output_features': self.stats['output_features'],
            'derived_features_created': self.stats['derived_features_created'],
            'file_size_mb': file_size_mb,
            'validation': validation,
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_report, f, indent=2)
        
        self.log(f"✓ Saved statistics: {stats_path}")
        
        self.stats['output_file'] = output_path
        self.stats['output_size_mb'] = file_size_mb
        
        return output_path
    
    # ==========================================================================
    # MAIN PIPELINE
    # ==========================================================================
    
    def build_feature_table(
        self,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Build complete feature table.
        
        Args:
            input_path: Optional custom input path
            output_path: Optional custom output path
            
        Returns:
            Feature table DataFrame
        """
        self.log("\n" + "="*80)
        self.log("WATCHTOWER FEATURE TABLE BUILDER")
        self.log("="*80)
        self.log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = datetime.now()
        
        # Step 1: Load
        df = self.load_windows(input_path)
        
        # Step 2: Sort (CRITICAL!)
        df = self.sort_data(df)
        
        # Step 3: Derive
        df = self.create_derived_features(df)
        
        # Step 4: Select
        df_feat = self.select_features(df)
        
        # Step 5: Validate
        validation = self.validate_data(df_feat)
        
        # Step 6: Save
        output_file = self.save_feature_table(df_feat, validation, output_path)
        
        # Track duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.stats['duration_seconds'] = duration
        
        # Final summary
        self.log("\n" + "="*80)
        self.log("FEATURE TABLE BUILD COMPLETE", "SUCCESS")
        self.log("="*80)
        self.log(f"Duration: {duration:.2f} seconds")
        self.log(f"Input: {self.stats['input_windows']:,} windows")
        self.log(f"Output: {len(df_feat):,} rows × {len(df_feat.columns)} columns")
        self.log(f"Features: {self.stats['output_features']}")
        self.log(f"File: {output_file}")
        self.log("="*80)
        
        return df_feat


if __name__ == "__main__":
    builder = FeatureTableBuilder(verbose=True)
    df_feat = builder.build_feature_table()
    
    print(f"\n✅ Feature table built!")
    print(f"   Shape: {df_feat.shape}")
    print(f"   Saved to: {builder.stats['output_file']}")
