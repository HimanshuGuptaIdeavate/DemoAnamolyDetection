"""
WATCHTOWER - Feature Table Configuration
Configuration for building feature table from windows.parquet

This module defines:
- Which features to select from windows
- New derived features to create
- Fixed column ordering (train-serve contract)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

# ==============================================================================
# INPUT/OUTPUT PATHS
# ==============================================================================

PATHS = {
    'input_windows': 'data/parquet/windows.parquet',
    'output_features': 'data/parquet/features_table.parquet',
    'reports': 'reports',
}

# ==============================================================================
# CORE FEATURES FROM WINDOWS
# ==============================================================================

# Numeric features to keep from windows.parquet
NUMERIC_FEATURES = [
    # Signal quality
    'rsrp_mean',
    'rsrp_std',
    'rsrq_mean',
    'rsrq_std',
    'sinr_mean',
    'sinr_std',
    'sinr_min',
    'sinr_max',
    
    # Resource allocation
    'prb_dl_mean',
    'prb_ul_mean',
    'mcs_dl_mode',
    'mcs_ul_mode',
    
    # Throughput
    'app_dl_mean',
    'app_dl_std',
    
    # Existing derived features from windowing
    'd_rsrp_mean',
    'd_rsrq_mean',
    'd_sinr_mean',
    'd_sinr_std',
    'd_app_dl_mean',
    'rsrp_rsrq_ratio',
    'sinr_range',
    'throughput_per_prb',
]

# Categorical features
CATEGORICAL_FEATURES = [
    'pci_mode',     # Cell ID (mode per window)
    'scenario_id',  # Scenario identifier
]

# Metadata columns (keep for reference)
METADATA_COLUMNS = [
    'ts_start_ns',  # Timestamp
]

# Target variable
TARGET = 'weak_label'

# ==============================================================================
# NEW DERIVED FEATURES (Feature Engineering)
# ==============================================================================

def create_prb_util_ratio(df: pd.DataFrame) -> pd.Series:
    """
    PRB utilization ratio (downlink / total).
    Measures how much PRB allocation goes to downlink vs uplink.
    """
    return df['prb_dl_mean'] / (df['prb_dl_mean'] + df['prb_ul_mean'] + 1e-3)


def create_hour_sin(df: pd.DataFrame) -> pd.Series:
    """
    Hour of day encoded as sine (cyclical encoding).
    Captures time-of-day patterns.
    """
    utc_hour = pd.to_datetime(df['ts_start_ns'], unit='ns', utc=True).dt.hour
    return np.sin(2 * np.pi * utc_hour / 24)


def create_hour_cos(df: pd.DataFrame) -> pd.Series:
    """
    Hour of day encoded as cosine (cyclical encoding).
    Works with hour_sin for complete cyclical representation.
    """
    utc_hour = pd.to_datetime(df['ts_start_ns'], unit='ns', utc=True).dt.hour
    return np.cos(2 * np.pi * utc_hour / 24)


def create_sinr_cv(df: pd.DataFrame) -> pd.Series:
    """
    SINR coefficient of variation (std / mean).
    Measures SINR volatility relative to its magnitude.
    """
    return df['sinr_std'] / (df['sinr_mean'].abs() + 1e-6)


def create_throughput_efficiency(df: pd.DataFrame) -> pd.Series:
    """
    Throughput per unit SINR.
    Measures how efficiently throughput is achieved given signal quality.
    """
    return df['app_dl_mean'] / (df['sinr_mean'] + 1e-6)


# Dictionary of new features to create
NEW_DERIVED_FEATURES = {
    'prb_util_ratio': {
        'function': create_prb_util_ratio,
        'description': 'Downlink PRB utilization ratio',
    },
    'hour_sin': {
        'function': create_hour_sin,
        'description': 'Hour of day (sine component)',
    },
    'hour_cos': {
        'function': create_hour_cos,
        'description': 'Hour of day (cosine component)',
    },
    'sinr_cv': {
        'function': create_sinr_cv,
        'description': 'SINR coefficient of variation',
    },
    'throughput_efficiency': {
        'function': create_throughput_efficiency,
        'description': 'Throughput per unit SINR',
    },
}

# ==============================================================================
# FIXED COLUMN ORDER (Train-Serve Contract)
# ==============================================================================

# This order MUST be maintained in training and production
# Any changes here require retraining the model

# All numeric features in fixed order
ALL_NUMERIC_FEATURES = [
    # Signal quality (8)
    'rsrp_mean',
    'rsrp_std',
    'rsrq_mean',
    'rsrq_std',
    'sinr_mean',
    'sinr_std',
    'sinr_min',
    'sinr_max',
    
    # Resource allocation (4)
    'prb_dl_mean',
    'prb_ul_mean',
    'mcs_dl_mode',
    'mcs_ul_mode',
    
    # Throughput (2)
    'app_dl_mean',
    'app_dl_std',
    
    # Existing derived features (8)
    'd_rsrp_mean',
    'd_rsrq_mean',
    'd_sinr_mean',
    'd_sinr_std',
    'd_app_dl_mean',
    'rsrp_rsrq_ratio',
    'sinr_range',
    'throughput_per_prb',
    
    # New derived features (5)
    'prb_util_ratio',
    'sinr_cv',
    'throughput_efficiency',
    'hour_sin',
    'hour_cos',
]

# Final column order for features_table.parquet
FINAL_COLUMN_ORDER = (
    METADATA_COLUMNS +           # ts_start_ns
    ALL_NUMERIC_FEATURES +       # 27 numeric features
    CATEGORICAL_FEATURES +       # pci, scenario_id
    [TARGET]                     # weak_label
)

# ==============================================================================
# VALIDATION
# ==============================================================================

VALIDATION_THRESHOLDS = {
    'min_windows': 500,           # Minimum expected windows
    'max_missing_ratio': 0.05,    # Max 5% missing values
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_numeric_features() -> List[str]:
    """Get list of all numeric features"""
    return ALL_NUMERIC_FEATURES


def get_categorical_features() -> List[str]:
    """Get list of categorical features"""
    return CATEGORICAL_FEATURES


def get_new_derived_features() -> Dict[str, Any]:
    """Get new derived features to create"""
    return NEW_DERIVED_FEATURES


def get_final_column_order() -> List[str]:
    """Get fixed column order for output"""
    return FINAL_COLUMN_ORDER


def print_config_summary():
    """Print configuration summary"""
    print("="*80)
    print("WATCHTOWER FEATURE TABLE CONFIGURATION")
    print("="*80)
    print(f"\nInput: {PATHS['input_windows']}")
    print(f"Output: {PATHS['output_features']}")
    
    print(f"\nFeatures:")
    print(f"  Existing numeric: {len(NUMERIC_FEATURES)}")
    print(f"  New derived: {len(NEW_DERIVED_FEATURES)}")
    print(f"  Total numeric: {len(ALL_NUMERIC_FEATURES)}")
    print(f"  Categorical: {len(CATEGORICAL_FEATURES)}")
    print(f"  Total features: {len(ALL_NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)}")
    
    print(f"\nNew Derived Features:")
    for name, spec in NEW_DERIVED_FEATURES.items():
        print(f"  • {name}: {spec['description']}")
    
    print(f"\nOutput Columns: {len(FINAL_COLUMN_ORDER)}")
    print("="*80)


if __name__ == "__main__":
    print_config_summary()
    
    print("\n" + "="*80)
    print("FIXED COLUMN ORDER (Train-Serve Contract)")
    print("="*80)
    
    print("\nMetadata:")
    for col in METADATA_COLUMNS:
        print(f"  • {col}")
    
    print(f"\nNumeric Features ({len(ALL_NUMERIC_FEATURES)}):")
    for i, col in enumerate(ALL_NUMERIC_FEATURES, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nCategorical Features ({len(CATEGORICAL_FEATURES)}):")
    for i, col in enumerate(CATEGORICAL_FEATURES, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nTarget:")
    print(f"  • {TARGET}")
    
    print(f"\nTotal: {len(FINAL_COLUMN_ORDER)} columns")
    print("="*80)
