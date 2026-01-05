"""
WATCHTOWER - Great Expectations Validation Suite
Comprehensive data validation for 5G telemetry data

This module provides production-grade validation using Great Expectations.
All validation rules are configured in validation_config.py for easy maintenance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


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

# Great Expectations imports
try:
    from great_expectations.dataset import PandasDataset
    from great_expectations.core import ExpectationSuite, ExpectationConfiguration
except ImportError:
    print("⚠️ Great Expectations not installed. Run: pip install great-expectations")
    raise

# Import validation configuration
from watchtower.data.validation_config import (
    REQUIRED_COLUMNS,
    REQUIRED_COLUMNS_CLEAN,
    NON_NULL_COLUMNS,
    NON_NULL_COLUMNS_CLEAN,
    NULLABLE_COLUMNS,
    NULLABLE_COLUMNS_CLEAN,
    NUMERIC_RANGES,
    NUMERIC_RANGES_CLEAN,
    CATEGORICAL_VALUES,
    CATEGORICAL_VALUES_CLEAN,
    ANOMALY_SIGNATURE_RULES,
    ANOMALY_SIGNATURE_RULES_CLEAN,
    QUALITY_THRESHOLDS,
    SCENARIO_RULES,
    REPORT_CONFIG,
    get_critical_features,
)


class WatchtowerValidator:
    """
    Comprehensive validator for WATCHTOWER 5G telemetry data.
    
    Features:
    - Schema validation (required columns)
    - Null value checks
    - Numeric range validation (3GPP compliant)
    - Categorical value validation
    - Anomaly signature validation
    - Data quality metrics
    - Scenario-specific rules
    - HTML + JSON reporting
    """
    
    def __init__(self, output_dir: str = "reports/validation"):
        """
        Initialize validator.

        Args:
            output_dir: Directory for validation reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = Path(REPORT_CONFIG["plots_dir"])
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.validation_results = {}
        self.summary_stats = {}
        self.failure_details = {}  # Store detailed failure information

        # Data type detection (raw vs clean/ETL)
        self.is_clean_data = False
        self._required_columns = REQUIRED_COLUMNS
        self._non_null_columns = NON_NULL_COLUMNS
        self._nullable_columns = NULLABLE_COLUMNS
        self._numeric_ranges = NUMERIC_RANGES
        self._categorical_values = CATEGORICAL_VALUES
        self._anomaly_rules = ANOMALY_SIGNATURE_RULES

    def _detect_data_type(self, df: pd.DataFrame) -> bool:
        """
        Auto-detect whether data is raw or ETL-cleaned based on column names.

        Args:
            df: DataFrame to analyze

        Returns:
            True if data appears to be ETL-cleaned, False if raw
        """
        # Check for cleaned column names
        clean_indicators = ["rsrp_dbm", "rsrq_db", "sinr_db", "time_s", "app_dl_mbps"]
        raw_indicators = ["RSRP", "RSRQ", "SINR", "Time", "throughput_DL"]

        clean_count = sum(1 for col in clean_indicators if col in df.columns)
        raw_count = sum(1 for col in raw_indicators if col in df.columns)

        return clean_count > raw_count

    def _set_config_for_data_type(self, is_clean: bool):
        """Set the appropriate configuration based on data type."""
        self.is_clean_data = is_clean

        if is_clean:
            print("📋 Detected ETL-CLEANED data format (standardized column names)")
            self._required_columns = REQUIRED_COLUMNS_CLEAN
            self._non_null_columns = NON_NULL_COLUMNS_CLEAN
            self._nullable_columns = NULLABLE_COLUMNS_CLEAN
            self._numeric_ranges = NUMERIC_RANGES_CLEAN
            self._categorical_values = CATEGORICAL_VALUES_CLEAN
            self._anomaly_rules = ANOMALY_SIGNATURE_RULES_CLEAN
        else:
            print("📋 Detected RAW data format (original column names)")
            self._required_columns = REQUIRED_COLUMNS
            self._non_null_columns = NON_NULL_COLUMNS
            self._nullable_columns = NULLABLE_COLUMNS
            self._numeric_ranges = NUMERIC_RANGES
            self._categorical_values = CATEGORICAL_VALUES
            self._anomaly_rules = ANOMALY_SIGNATURE_RULES

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from CSV or parquet file(s).

        Args:
            data_path: Path to directory containing CSV/parquet files, single CSV, or parquet file

        Returns:
            Merged DataFrame
        """
        data_path = Path(data_path)

        if data_path.is_file():
            # Single file - check extension
            if data_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(data_path)
                print(f"✅ Loaded {len(df):,} samples from {data_path.name} (parquet)")
            else:
                df = pd.read_csv(data_path, on_bad_lines='skip', low_memory=False)
                print(f"✅ Loaded {len(df):,} samples from {data_path.name} (csv)")
        else:
            # Directory - check for parquet first, then CSV
            parquet_files = sorted(data_path.glob("*.parquet"))
            csv_files = sorted(data_path.glob("*.csv"))

            if parquet_files:
                dfs = []
                for pq_file in parquet_files:
                    df = pd.read_parquet(pq_file)
                    df['source_file'] = pq_file.stem
                    dfs.append(df)
                    print(f"  ✓ {pq_file.name}: {len(df):,} samples")

                df = pd.concat(dfs, ignore_index=True)
                print(f"✅ Merged {len(df):,} total samples from {len(parquet_files)} parquet files")
            elif csv_files:
                dfs = []
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file, on_bad_lines='skip', low_memory=False)
                    df['source_file'] = csv_file.stem
                    dfs.append(df)
                    print(f"  ✓ {csv_file.name}: {len(df):,} samples")

                df = pd.concat(dfs, ignore_index=True)
                print(f"✅ Merged {len(df):,} total samples from {len(csv_files)} CSV files")
            else:
                raise ValueError(f"No CSV or parquet files found in {data_path}")

        return df
    
    def validate_schema(self, gx_df: PandasDataset) -> Dict[str, Any]:
        """
        Validate dataset schema (required columns exist).

        Args:
            gx_df: Great Expectations PandasDataset

        Returns:
            Validation results
        """
        print("\n🔍 Validating Schema...")
        results = {}

        for col in self._required_columns:
            result = gx_df.expect_column_to_exist(col)
            results[f"column_exists_{col}"] = result['success']

            if result['success']:
                print(f"  ✓ {col}")
            else:
                print(f"  ✗ {col} - MISSING")
                self.failure_details[f"column_exists_{col}"] = {
                    'category': 'schema',
                    'column': col,
                    'error': f"Required column '{col}' not found in dataset",
                    'available_columns': list(gx_df.columns)[:10]  # Show first 10 columns
                }

        return results
    
    def validate_null_values(self, gx_df: PandasDataset) -> Dict[str, Any]:
        """
        Validate null value constraints.

        Args:
            gx_df: Great Expectations PandasDataset

        Returns:
            Validation results
        """
        print("\n🔍 Validating Null Values...")
        results = {}

        # Check non-null columns
        for col in self._non_null_columns:
            if col not in gx_df.columns:
                continue

            result = gx_df.expect_column_values_to_not_be_null(col)
            results[f"not_null_{col}"] = result['success']

            if result['success']:
                print(f"  ✓ {col}: No nulls")
            else:
                null_pct = result['result'].get('unexpected_percent', 0)
                print(f"  ✗ {col}: {null_pct:.2f}% nulls (should be 0%)")
                self.failure_details[f"not_null_{col}"] = {
                    'category': 'null_values',
                    'column': col,
                    'error': f"{null_pct:.2f}% null values found (expected 0%)",
                    'actual_value': f"{null_pct:.2f}%",
                    'expected': "0%"
                }

        # Check nullable columns (with thresholds)
        for col, max_null_pct in self._nullable_columns.items():
            if col not in gx_df.columns:
                continue

            result = gx_df.expect_column_values_to_not_be_null(
                col,
                mostly=1 - max_null_pct  # e.g., 0.95 for 5% allowed nulls
            )
            results[f"nullable_{col}"] = result['success']

            null_pct = result['result'].get('unexpected_percent', 0)
            if result['success']:
                print(f"  ✓ {col}: {null_pct:.2f}% nulls (≤{max_null_pct*100}% allowed)")
            else:
                print(f"  ⚠️ {col}: {null_pct:.2f}% nulls (>{max_null_pct*100}% allowed)")

        return results
    
    def validate_numeric_ranges(self, gx_df: PandasDataset) -> Dict[str, Any]:
        """
        Validate numeric columns are within expected ranges.

        Args:
            gx_df: Great Expectations PandasDataset

        Returns:
            Validation results
        """
        print("\n🔍 Validating Numeric Ranges (3GPP Standards)...")
        results = {}

        for col, range_spec in self._numeric_ranges.items():
            if col not in gx_df.columns:
                continue

            min_val = range_spec['min']
            max_val = range_spec['max']
            strict = range_spec['strict']

            # Strict: All values must be in range
            # Non-strict: Allow some outliers (mostly parameter)
            mostly = 1.0 if strict else 0.95

            result = gx_df.expect_column_values_to_be_between(
                col,
                min_value=min_val,
                max_value=max_val,
                mostly=mostly
            )

            results[f"range_{col}"] = result['success']

            if result['success']:
                print(f"  ✓ {col}: [{min_val}, {max_val}] {'(strict)' if strict else '(95%)'}")
            else:
                outliers = result['result'].get('unexpected_percent', 0)
                print(f"  ✗ {col}: {outliers:.2f}% outside [{min_val}, {max_val}]")
                self.failure_details[f"range_{col}"] = {
                    'category': 'numeric_ranges',
                    'column': col,
                    'error': f"{outliers:.2f}% of values outside expected range",
                    'expected_range': f"[{min_val}, {max_val}]",
                    'outlier_percent': f"{outliers:.2f}%",
                    'strict': strict
                }

        return results

    def validate_categorical_values(self, gx_df: PandasDataset) -> Dict[str, Any]:
        """
        Validate categorical columns have expected values.

        Args:
            gx_df: Great Expectations PandasDataset

        Returns:
            Validation results
        """
        print("\n🔍 Validating Categorical Values...")
        results = {}

        for col, spec in self._categorical_values.items():
            if col not in gx_df.columns:
                continue

            expected_values = spec['values']

            result = gx_df.expect_column_values_to_be_in_set(
                col,
                value_set=expected_values
            )

            results[f"categorical_{col}"] = result['success']

            if result['success']:
                print(f"  ✓ {col}: {expected_values}")
            else:
                unexpected = result['result'].get('unexpected_percent', 0)
                print(f"  ✗ {col}: {unexpected:.2f}% unexpected values")

        return results
    
    def validate_anomaly_signatures(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate anomaly signature patterns (based on your analysis).

        Args:
            df: Pandas DataFrame

        Returns:
            Validation results
        """
        print("\n🔍 Validating Anomaly Signatures...")
        results = {}

        if 'lab_anom' not in df.columns:
            print("  ⚠️ Skipping - lab_anom column not found")
            return results

        # Get correct column names based on data type
        sinr_col = 'sinr_db' if self.is_clean_data else 'SINR'
        tput_col = 'app_dl_mbps' if self.is_clean_data else 'throughput_DL'

        # Check SINR patterns for normal vs anomaly
        normal = df[df['lab_anom'] == 0]
        anomaly = df[df['lab_anom'] == 1]

        # Normal samples should have higher SINR
        if sinr_col in df.columns and len(normal) > 0 and len(anomaly) > 0:
            normal_sinr_mean = normal[sinr_col].mean()
            anomaly_sinr_mean = anomaly[sinr_col].mean()

            # Expect at least 10 dB difference (based on your analysis: 50-75% drop)
            sinr_diff = normal_sinr_mean - anomaly_sinr_mean
            results['anomaly_sinr_separation'] = sinr_diff >= 10

            if sinr_diff >= 10:
                print(f"  ✓ SINR separation: {sinr_diff:.1f} dB (≥10 dB)")
            else:
                print(f"  ⚠️ SINR separation: {sinr_diff:.1f} dB (<10 dB expected)")

            # Check if anomaly SINR is in expected range
            anomaly_sinr_low = (anomaly[sinr_col] < 15).sum() / len(anomaly)
            results['anomaly_sinr_low_rate'] = anomaly_sinr_low >= 0.6  # 60%+

            print(f"  ✓ Anomaly SINR < 15 dB: {anomaly_sinr_low*100:.1f}%")

        # Check throughput patterns
        if tput_col in df.columns and len(normal) > 0 and len(anomaly) > 0:
            normal_tput_mean = normal[tput_col].mean()
            anomaly_tput_mean = anomaly[tput_col].mean()

            if normal_tput_mean > 0:
                tput_drop_pct = (normal_tput_mean - anomaly_tput_mean) / normal_tput_mean
                results['anomaly_throughput_drop'] = tput_drop_pct >= 0.3  # 30%+ drop
                print(f"  ✓ Throughput drop: {tput_drop_pct*100:.1f}%")

        return results
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate overall data quality metrics.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            Validation results
        """
        print("\n🔍 Validating Data Quality...")
        results = {}
        
        # Check minimum samples
        min_samples = QUALITY_THRESHOLDS['min_samples']
        results['sufficient_samples'] = len(df) >= min_samples
        print(f"  ✓ Samples: {len(df):,} (≥{min_samples:,})")
        
        # Check duplicate rows
        duplicates = df.duplicated().sum() / len(df)
        max_dup = QUALITY_THRESHOLDS['max_duplicate_rows']
        results['low_duplicates'] = duplicates <= max_dup
        print(f"  ✓ Duplicates: {duplicates*100:.2f}% (≤{max_dup*100}%)")
        
        # Check anomaly rate
        if 'lab_anom' in df.columns:
            anomaly_rate = df['lab_anom'].sum() / len(df)
            min_anom = QUALITY_THRESHOLDS['min_anomaly_rate']
            max_anom = QUALITY_THRESHOLDS['max_anomaly_rate']
            
            results['anomaly_rate_reasonable'] = min_anom <= anomaly_rate <= max_anom
            print(f"  ✓ Anomaly rate: {anomaly_rate*100:.1f}% ({min_anom*100}-{max_anom*100}%)")
        
        # Overall null percentage
        total_nulls = df.isnull().sum().sum() / (len(df) * len(df.columns))
        max_null = QUALITY_THRESHOLDS['max_null_percentage_overall']
        results['low_overall_nulls'] = total_nulls <= max_null
        print(f"  ✓ Overall nulls: {total_nulls*100:.2f}% (≤{max_null*100}%)")
        
        return results
    
    def validate_scenario_specific(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate scenario-specific rules.

        Args:
            df: Pandas DataFrame

        Returns:
            Validation results
        """
        print("\n🔍 Validating Scenario-Specific Rules...")
        results = {}

        # Check for scenario column (source_file from raw data load, or scenario_id from ETL)
        scenario_col = None
        if 'source_file' in df.columns:
            scenario_col = 'source_file'
        elif 'scenario_id' in df.columns:
            scenario_col = 'scenario_id'

        if scenario_col is None:
            print("  ⚠️ Skipping - no source_file or scenario_id column")
            return results

        for scenario, rules in SCENARIO_RULES.items():
            scenario_df = df[df[scenario_col].str.contains(scenario, case=False, na=False)]
            
            if len(scenario_df) == 0:
                continue
            
            # Check sample count
            min_samples = rules['expected_samples_min']
            has_enough = len(scenario_df) >= min_samples
            results[f'scenario_{scenario}_samples'] = has_enough
            
            # Check anomaly rate
            if 'lab_anom' in scenario_df.columns:
                anom_rate = scenario_df['lab_anom'].sum() / len(scenario_df)
                min_rate, max_rate = rules['expected_anomaly_rate']
                in_range = min_rate <= anom_rate <= max_rate
                results[f'scenario_{scenario}_anomaly_rate'] = in_range

                status = "✓" if has_enough and in_range else "⚠️"
                print(f"  {status} {scenario}: {len(scenario_df):,} samples, "
                      f"{anom_rate*100:.1f}% anomalies")

                if not in_range:
                    self.failure_details[f'scenario_{scenario}_anomaly_rate'] = {
                        'category': 'scenario_specific',
                        'scenario': scenario,
                        'error': f"Anomaly rate {anom_rate*100:.1f}% outside expected range",
                        'actual_rate': f"{anom_rate*100:.1f}%",
                        'expected_range': f"{min_rate*100:.1f}% - {max_rate*100:.1f}%",
                        'sample_count': len(scenario_df)
                    }

        return results
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the dataset.

        Args:
            df: Pandas DataFrame

        Returns:
            Summary statistics dictionary
        """
        stats = {
            'total_samples': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'data_format': 'clean' if self.is_clean_data else 'raw',
        }

        # Critical features statistics (use correct column names)
        critical_features = get_critical_features(clean=self.is_clean_data)
        for feat in critical_features:
            if feat in df.columns:
                stats[f'{feat}_mean'] = float(df[feat].mean())
                stats[f'{feat}_std'] = float(df[feat].std())
                stats[f'{feat}_min'] = float(df[feat].min())
                stats[f'{feat}_max'] = float(df[feat].max())
                stats[f'{feat}_null_pct'] = float(df[feat].isnull().sum() / len(df) * 100)

        # Label statistics
        if 'lab_anom' in df.columns:
            stats['anomaly_count'] = int(df['lab_anom'].sum())
            stats['anomaly_rate'] = float(df['lab_anom'].sum() / len(df))

        if 'lab_inf' in df.columns:
            stats['interference_count'] = int(df['lab_inf'].sum())
            stats['interference_rate'] = float(df['lab_inf'].sum() / len(df))

        return stats
    
    def run_validation(self, data_path: str, save_report: bool = True) -> Tuple[bool, Dict]:
        """
        Run complete validation suite.

        Args:
            data_path: Path to data directory or CSV file
            save_report: Whether to save validation report

        Returns:
            Tuple of (overall_success, validation_results)
        """
        print("="*80)
        print("WATCHTOWER DATA VALIDATION")
        print("="*80)

        # Clear previous failure details
        self.failure_details = {}

        # Load data
        df = self.load_data(data_path)

        # Auto-detect data type and set appropriate config
        is_clean = self._detect_data_type(df)
        self._set_config_for_data_type(is_clean)

        # Create Great Expectations dataset
        gx_df = PandasDataset(df)

        # Run all validation checks
        all_results = {}
        
        all_results['schema'] = self.validate_schema(gx_df)
        all_results['null_values'] = self.validate_null_values(gx_df)
        all_results['numeric_ranges'] = self.validate_numeric_ranges(gx_df)
        all_results['categorical'] = self.validate_categorical_values(gx_df)
        all_results['anomaly_signatures'] = self.validate_anomaly_signatures(df)
        all_results['data_quality'] = self.validate_data_quality(df)
        all_results['scenario_specific'] = self.validate_scenario_specific(df)
        
        # Generate summary statistics
        self.summary_stats = self.generate_summary_stats(df)
        
        # Flatten results
        flat_results = {}
        for category, results in all_results.items():
            flat_results.update(results)
        
        # Calculate overall success
        total_checks = len(flat_results)
        passed_checks = sum(flat_results.values())
        overall_success = passed_checks == total_checks
        
        print("\n" + "="*80)
        print(f"VALIDATION SUMMARY: {passed_checks}/{total_checks} checks passed")
        print("="*80)
        
        if overall_success:
            print("✅ All validation checks passed!")
        else:
            failed = total_checks - passed_checks
            print(f"⚠️ {failed} validation checks failed")

            # Print detailed failure information
            print("\n" + "-"*80)
            print("FAILED CHECKS DETAILS:")
            print("-"*80)
            for category, results in all_results.items():
                failed_in_category = {k: v for k, v in results.items() if not v}
                if failed_in_category:
                    print(f"\n📁 Category: {category.upper()}")
                    for check_name, passed in failed_in_category.items():
                        print(f"   ❌ {check_name}: FAILED")
                        # Print detailed error if available
                        if check_name in self.failure_details:
                            details = self.failure_details[check_name]
                            print(f"      → Error: {details.get('error', 'Unknown error')}")
                            if 'column' in details:
                                print(f"      → Column: {details['column']}")
                            if 'expected_range' in details:
                                print(f"      → Expected: {details['expected_range']}")
                            if 'actual_value' in details:
                                print(f"      → Actual: {details['actual_value']}")
                            if 'outlier_percent' in details:
                                print(f"      → Outliers: {details['outlier_percent']}")
                            if 'scenario' in details:
                                print(f"      → Scenario: {details['scenario']}")
                            if 'actual_rate' in details:
                                print(f"      → Actual Rate: {details['actual_rate']}")
            print("-"*80)

        # Save report
        if save_report:
            self._save_report(all_results, flat_results, overall_success)

        self.validation_results = {
            'success': overall_success,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'results_by_category': all_results,
            'flat_results': flat_results,
            'failure_details': self.failure_details,
            'summary_stats': self.summary_stats,
            'timestamp': datetime.now().isoformat(),
        }
        
        return overall_success, self.validation_results
    
    def _save_report(self, results: Dict, flat_results: Dict, success: bool):
        """Save validation report as JSON."""
        report_path = self.output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'validation_success': success,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'flat_results': flat_results,
            'summary_stats': self.summary_stats,
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)

        print(f"\n📄 Validation report saved: {report_path}")

        # Also save as latest
        latest_path = self.output_dir / "validation_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)


def main():
    """Main execution function."""
    import sys
    
    # Default data path
    data_path = "data/raw/sutd"
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    # Run validation
    validator = WatchtowerValidator()
    success, results = validator.run_validation(data_path)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
