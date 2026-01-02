"""
WATCHTOWER - ETL Configuration
Configuration for data curation and cleaning pipeline

This file defines all column mappings, type coercions, and clipping ranges
for the ETL (Extract, Transform, Load) pipeline.
"""

import numpy as np
from typing import Dict, List, Any

# ==============================================================================
# COLUMN MAPPING (Raw → Standardized Snake-Case)
# ==============================================================================

# Map raw column names to consistent snake_case names
# This eliminates case-sensitivity bugs and makes downstream code predictable
COLUMN_RENAME_MAP = {
    # Time and identifiers
    'Time': 'time_s',
    'NR-ARFCN': 'nr_arfcn',
    'NR_ARFCN': 'nr_arfcn',
    'PCI': 'pci',
    'Cell ID': 'cell_id',
    'C-RNTI': 'c_rnti',
    'C_RNTI': 'c_rnti',
    
    # Signal quality metrics (handle multiple naming conventions)
    'SS_RSRP_dBm': 'rsrp_dbm',
    'RSRP': 'rsrp_dbm',
    'SS_RSRQ_dB': 'rsrq_db',
    'RSRQ': 'rsrq_db',
    'SS_SINR_dB': 'sinr_db',
    'SINR': 'sinr_db',
    
    # Modulation and Coding Scheme
    'PDSCH_MCS_CW0': 'mcs_dl',
    'PDSCH_MCS': 'mcs_dl',
    'PUSCH_MCS_CW0': 'mcs_ul',
    'PUSCH_MCS': 'mcs_ul',
    
    # Physical Resource Blocks
    'PDSCH_PRBs': 'prb_dl',
    'PDSCH PRBs': 'prb_dl',
    'PUSCH_PRBs': 'prb_ul',
    'PUSCH PRBs': 'prb_ul',
    
    # Throughput
    'APP_DL_Mbps': 'app_dl_mbps',
    'throughput_DL': 'app_dl_mbps',  # Alternative naming
    'APP_UL_Mbps': 'app_ul_mbps',
    'throughput_UL': 'app_ul_mbps',
    
    # Labels (ground truth)
    'lab_anom': 'lab_anom',
    'lab_bs': 'lab_bs',
    'lab_inf': 'lab_inf',
    'lab_1rr': 'lab_1rr',
}

# ==============================================================================
# REQUIRED COLUMNS (After Renaming)
# ==============================================================================

# Core columns that MUST exist after renaming
REQUIRED_COLUMNS = [
    'time_s',
    'rsrp_dbm',
    'rsrq_db',
    'sinr_db',
    'pci',
    'lab_anom',
]

# Optional columns (create with default values if missing)
OPTIONAL_COLUMNS_DEFAULTS = {
    'lab_bs': 0,
    'lab_inf': 0,
    'lab_1rr': 0,
    'nr_arfcn': 0,
    'c_rnti': 0,
}

# ==============================================================================
# TYPE COERCION
# ==============================================================================

# Numeric columns to convert with pd.to_numeric(errors='coerce')
NUMERIC_COLUMNS = [
    'time_s',
    'nr_arfcn',
    'pci',
    'rsrp_dbm',
    'rsrq_db',
    'sinr_db',
    'mcs_dl',
    'mcs_ul',
    'prb_dl',
    'prb_ul',
    'app_dl_mbps',
    'app_ul_mbps',
]

# Integer columns (convert to Int64 nullable integer type)
INTEGER_COLUMNS = [
    'pci',
    'mcs_dl',
    'mcs_ul',
    'lab_anom',
    'lab_bs',
    'lab_inf',
    'lab_1rr',
]

# Float columns (convert to float32 for memory efficiency)
FLOAT_COLUMNS = [
    'time_s',
    'rsrp_dbm',
    'rsrq_db',
    'sinr_db',
    'prb_dl',
    'prb_ul',
    'app_dl_mbps',
    'app_ul_mbps',
]

# ==============================================================================
# TIMESTAMP CONFIGURATION
# ==============================================================================

# Start epoch for absolute timestamp generation (nanoseconds)
# This creates monotonic timestamps similar to real PM logs
# Actual date doesn't matter - just ensures proper ordering
START_EPOCH_NS = np.int64(1_700_000_000_000_000_000)

# Timestamp column name (generated)
TIMESTAMP_COLUMN = 'ts_ns'

# ==============================================================================
# RANGE CLIPPING (Based on Client's Sanity Checks)
# ==============================================================================

# Clip values to realistic 3GPP bounds
# Purpose: Remove broken measurements (e.g., RSRP = +100 dBm, SINR = -500 dB)
CLIPPING_RANGES = {
    'rsrp_dbm': {
        'min': -156,
        'max': -31,
        'reason': 'Physical receive sensitivity to typical upper bound'
    },
    
    'rsrq_db': {
        'min': -20,
        'max': 3,
        'reason': 'Empirical radio quality envelope'
    },
    
    'sinr_db': {
        'min': -10,
        'max': 30,
        'reason': 'Indoor testbed limits'
    },
    
    'mcs_dl': {
        'min': 0,
        'max': 28,
        'reason': 'Valid NR MCS index range'
    },
    
    'mcs_ul': {
        'min': 0,
        'max': 28,
        'reason': 'Valid NR MCS index range'
    },
    
    'prb_dl': {
        'min': 0,
        'max': 273,
        'reason': 'Max RBs for 100 MHz bandwidth'
    },
    
    'prb_ul': {
        'min': 0,
        'max': 273,
        'reason': 'Max RBs for 100 MHz bandwidth'
    },
    
    'app_dl_mbps': {
        'min': 0,
        'max': 100,
        'reason': 'Avoid corrupt iPerf3 logs (testbed limit)'
    },
    
    'app_ul_mbps': {
        'min': 0,
        'max': 100,
        'reason': 'Testbed limit'
    },
}

# ==============================================================================
# NULL HANDLING (Tiered Strategy)
# ==============================================================================

# TIER 1: Critical columns - DROP rows with nulls
# These are ESSENTIAL for anomaly detection
# Null rate: ~1-2% → minimal data loss (~200 rows from 8,732)
DROP_NULL_COLUMNS = [
    'rsrp_dbm',    # Signal strength - critical for detection
    'rsrq_db',     # Signal quality - critical for detection  
    'sinr_db',     # PRIMARY anomaly indicator - must not be null!
    'pci',         # Cell ID - needed for tracking
]

# TIER 2: Operational columns - IMPUTE nulls intelligently
# These can be estimated without compromising anomaly detection
# Null rate: ~5-8% → imputation preserves data
IMPUTE_STRATEGIES = {
    # Modulation Coding Scheme - use MODE (most common)
    # MCS is categorical-like, mode is appropriate
    'mcs_dl': {
        'method': 'mode',
        'reason': 'MCS is categorical, use most common value'
    },
    'mcs_ul': {
        'method': 'mode', 
        'reason': 'MCS is categorical, use most common value'
    },
    
    # Resource Blocks - use MEDIAN (robust to outliers)
    # PRBs are continuous, median is robust
    'prb_dl': {
        'method': 'median',
        'reason': 'PRB usage is continuous, median is robust'
    },
    'prb_ul': {
        'method': 'median',
        'reason': 'PRB usage is continuous, median is robust'
    },
    
    # Throughput - use FORWARD FILL (temporal continuity)
    # Throughput changes slowly, use last known value
    'app_dl_mbps': {
        'method': 'ffill',
        'reason': 'Throughput has temporal continuity, use last value'
    },
    'app_ul_mbps': {
        'method': 'ffill',
        'reason': 'Throughput has temporal continuity, use last value'
    },
}

# Optional: Columns where nulls are ACCEPTABLE (keep as-is)
# These are not used in primary anomaly detection
NULLABLE_COLUMNS = [
    'nr_arfcn',    # Nice to have but not critical
    'c_rnti',      # Nice to have but not critical
]

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================

# Final column order (for clean parquet output)
FINAL_COLUMN_ORDER = [
    'ts_ns',
    'scenario_id',
    'time_s',
    'nr_arfcn',
    'pci',
    'rsrp_dbm',
    'rsrq_db',
    'sinr_db',
    'mcs_dl',
    'mcs_ul',
    'prb_dl',
    'prb_ul',
    'app_dl_mbps',
    'lab_anom',
    'lab_bs',
    'lab_inf',
    'lab_1rr',
]

# Output directories
OUTPUT_DIRS = {
    'parquet': 'data/parquet',
    'interim': 'data/interim',
    'reports': 'reports',
    'plots': 'reports/plots',
}

# Output filenames
OUTPUT_FILES = {
    'clean_data': 'clean_data.parquet',
    'etl_report': 'etl_report.json',
    'clipping_stats': 'clipping_statistics.json',
}

# ==============================================================================
# ETL STATISTICS TRACKING
# ==============================================================================

# Metrics to track during ETL
ETL_METRICS = [
    'input_rows',
    'input_columns',
    'output_rows',
    'output_columns',
    'rows_dropped_nulls',
    'rows_dropped_duplicates',
    'values_clipped',
    'columns_renamed',
    'columns_type_coerced',
]

# ==============================================================================
# MLFLOW CONFIGURATION
# ==============================================================================

MLFLOW_CONFIG = {
    'experiment_name': 'watchtower_etl',
    'run_name_prefix': 'data_curation',
    'log_artifacts': True,
    'log_stats': True,
    'log_plots': True,
}

# ==============================================================================
# VALIDATION THRESHOLDS
# ==============================================================================

# Warn if these thresholds are exceeded during ETL
VALIDATION_THRESHOLDS = {
    'max_null_percentage': 5.0,      # Warn if > 5% nulls
    'max_clipped_percentage': 10.0,   # Warn if > 10% values clipped
    'max_duplicate_percentage': 1.0,  # Warn if > 1% duplicates
    'min_output_rows': 5000,          # Warn if < 5000 rows output
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_output_path(filename: str, subdir: str = 'parquet') -> str:
    """Get full output path for a file"""
    from pathlib import Path
    output_dir = Path(OUTPUT_DIRS[subdir])
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / filename)


def print_config_summary():
    """Print ETL configuration summary"""
    print("="*80)
    print("WATCHTOWER ETL CONFIGURATION")
    print("="*80)
    print(f"\nColumn Mappings: {len(COLUMN_RENAME_MAP)} mappings")
    print(f"Required Columns: {len(REQUIRED_COLUMNS)}")
    print(f"Numeric Columns: {len(NUMERIC_COLUMNS)}")
    print(f"Clipping Ranges: {len(CLIPPING_RANGES)}")
    print(f"\nOutput Directory: {OUTPUT_DIRS['parquet']}")
    print(f"Clean Data File: {OUTPUT_FILES['clean_data']}")
    print("="*80)


if __name__ == "__main__":
    print_config_summary()
