"""
WATCHTOWER - Windowing and Feature Engineering Configuration
Configuration for creating 5-second time windows and derived features

This module defines all windowing parameters, aggregation strategies,
and derived feature specifications.
"""

import numpy as np
from typing import Dict, List, Any

# ==============================================================================
# WINDOWING PARAMETERS
# ==============================================================================

# Window size in seconds
WINDOW_SEC = 5

# Sampling rate of raw data (Hz)
SAMPLE_RATE_HZ = 2

# Expected samples per window
SAMPLES_PER_WINDOW = WINDOW_SEC * SAMPLE_RATE_HZ  # 5s × 2Hz = 10 samples

# Minimum samples required for a valid window (70% threshold)
MIN_SAMPLES_THRESHOLD = 0.7 * SAMPLES_PER_WINDOW  # 7 samples minimum

# ==============================================================================
# WINDOW AGGREGATIONS
# ==============================================================================

# Define aggregation functions for each feature
# Format: {column_name: [agg_functions]}
WINDOW_AGGREGATIONS = {
    # Signal quality metrics - compute mean and std
    'rsrp_dbm': ['mean', 'std'],
    'rsrq_db': ['mean', 'std'],
    'sinr_db': ['mean', 'std', 'min', 'max'],  # SINR: primary indicator, more stats
    
    # Resource allocation - compute mean
    'prb_dl': ['mean'],
    'prb_ul': ['mean'],
    
    # Throughput - compute mean
    'app_dl_mbps': ['mean', 'std'],  # Throughput variability is important
    
    # Modulation and coding - compute mode (most common)
    'mcs_dl': ['mode'],
    'mcs_ul': ['mode'],
    
    # Cell ID - use mode (most frequent cell in window)
    'pci': ['mode'],
}

# Aggregation function mappings
AGG_FUNCTION_MAP = {
    'mean': lambda x: float(x.mean()) if len(x) > 0 else 0.0,
    'std': lambda x: float(x.std(ddof=0)) if len(x) > 1 else 0.0,
    'min': lambda x: float(x.min()) if len(x) > 0 else 0.0,
    'max': lambda x: float(x.max()) if len(x) > 0 else 0.0,
    'mode': lambda x: int(x.mode().iloc[0]) if len(x.mode()) > 0 else int(x.iloc[0]) if len(x) > 0 else 0,
    'sum': lambda x: float(x.sum()) if len(x) > 0 else 0.0,
}

# ==============================================================================
# DERIVED FEATURES (Delta Features)
# ==============================================================================

# Features to compute deltas (first differences) for
# These capture temporal changes between consecutive windows
DELTA_FEATURES = [
    'rsrp_mean',  # Change in signal strength
    'rsrq_mean',  # Change in signal quality
    'sinr_mean',  # Change in SINR (PRIMARY for drone "wiggle" detection!)
    'sinr_std',   # Change in SINR volatility
    'app_dl_mean', # Change in throughput
]

# Prefix for delta features
DELTA_PREFIX = 'd_'

# ==============================================================================
# RATIO FEATURES
# ==============================================================================

# Compute useful signal quality ratios
RATIO_FEATURES = {
    # RSRP/RSRQ ratio - signal strength vs quality trade-off
    'rsrp_rsrq_ratio': {
        'numerator': 'rsrp_mean',
        'denominator': 'rsrq_mean',
        'description': 'Signal strength to quality ratio'
    },
    
    # SINR range - volatility indicator
    'sinr_range': {
        'formula': lambda df: df['sinr_max'] - df['sinr_min'],
        'description': 'SINR volatility within window'
    },
    
    # Throughput per PRB - efficiency indicator
    'throughput_per_prb': {
        'numerator': 'app_dl_mean',
        'denominator': 'prb_dl_mean',
        'description': 'Throughput efficiency (Mbps per PRB)'
    },
}

# ==============================================================================
# LABEL AGGREGATION
# ==============================================================================

# Label columns to aggregate
LABEL_COLUMNS = ['lab_anom', 'lab_bs', 'lab_inf', 'lab_1rr']

# Weak label: 1 if ANY sample in window has ANY label flag set
# This is appropriate for anomaly detection (any anomaly in window = anomalous window)
WEAK_LABEL_NAME = 'weak_label'

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================

# Output file paths
OUTPUT_PATHS = {
    'windows_parquet': 'data/parquet/windows.parquet',
    'metrics_json': 'reports/windowing_metrics.json',
    'preview_plot': 'reports/plots/windowing_preview.png',
    'distribution_plot': 'reports/plots/windowing_distribution.png',
}

# Expected output columns (in order)
OUTPUT_COLUMNS_ORDER = [
    # Window metadata
    'ts_start_ns',
    'scenario_id',
    'pci_mode',
    
    # Signal quality aggregations
    'rsrp_mean', 'rsrp_std',
    'rsrq_mean', 'rsrq_std',
    'sinr_mean', 'sinr_std', 'sinr_min', 'sinr_max',
    
    # Resource allocation
    'prb_dl_mean', 'prb_ul_mean',
    
    # Throughput
    'app_dl_mean', 'app_dl_std',
    
    # MCS
    'mcs_dl_mode', 'mcs_ul_mode',
    
    # Window quality
    'gap_ratio',
    'window_samples',
    
    # Labels
    'weak_label',
    
    # Derived features (deltas)
    'd_rsrp_mean',
    'd_rsrq_mean',
    'd_sinr_mean',
    'd_sinr_std',
    'd_app_dl_mean',
    
    # Ratio features
    'rsrp_rsrq_ratio',
    'sinr_range',
    'throughput_per_prb',
]

# ==============================================================================
# MLFLOW CONFIGURATION
# ==============================================================================

MLFLOW_CONFIG = {
    'experiment_name': 'watchtower_windowing',
    'run_name_prefix': 'windowing',
    'tracking_uri': './mlruns',
}

# Parameters to log
MLFLOW_PARAMETERS = {
    'window_sec': WINDOW_SEC,
    'sample_rate_hz': SAMPLE_RATE_HZ,
    'samples_per_window': SAMPLES_PER_WINDOW,
    'min_samples_threshold': MIN_SAMPLES_THRESHOLD,
}

# ==============================================================================
# VALIDATION THRESHOLDS
# ==============================================================================

# Quality checks for windowing output
VALIDATION_THRESHOLDS = {
    'min_windows': 500,              # Minimum expected windows
    'max_gap_ratio': 0.20,           # Warn if >20% missing data on average
    'min_weak_label_rate': 0.05,     # Warn if <5% anomalies
    'max_weak_label_rate': 0.80,     # Warn if >80% anomalies
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_output_path(key: str) -> str:
    """Get output path for a given key"""
    from pathlib import Path
    path = OUTPUT_PATHS.get(key)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def print_config_summary():
    """Print windowing configuration summary"""
    print("="*80)
    print("WATCHTOWER WINDOWING CONFIGURATION")
    print("="*80)
    print(f"\nWindow Parameters:")
    print(f"  Window size: {WINDOW_SEC} seconds")
    print(f"  Sample rate: {SAMPLE_RATE_HZ} Hz")
    print(f"  Samples per window: {SAMPLES_PER_WINDOW}")
    print(f"  Min samples threshold: {MIN_SAMPLES_THRESHOLD}")
    
    print(f"\nAggregations:")
    for col, aggs in WINDOW_AGGREGATIONS.items():
        print(f"  {col}: {', '.join(aggs)}")
    
    print(f"\nDerived Features:")
    print(f"  Delta features: {len(DELTA_FEATURES)}")
    print(f"  Ratio features: {len(RATIO_FEATURES)}")
    
    print(f"\nOutput:")
    print(f"  Windows file: {OUTPUT_PATHS['windows_parquet']}")
    print(f"  Expected columns: {len(OUTPUT_COLUMNS_ORDER)}")
    print("="*80)


if __name__ == "__main__":
    print_config_summary()
