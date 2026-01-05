"""
WATCHTOWER - Data Validation Configuration
Great Expectations validation rules for 5G telemetry data

This file defines all validation rules in a centralized, easily modifiable format.
To change column names or validation rules, simply edit this configuration.
"""

from typing import Dict, List, Any

# ==============================================================================
# COLUMN MAPPING
# ==============================================================================
# If column names change in future datasets, update only this section

# Raw column names (original data)
COLUMN_NAMES_RAW = {
    # Core identifiers
    "time": "Time",
    "pci": "PCI",
    "cell_id": "Cell ID",

    # Signal quality metrics (primary features for anomaly detection)
    "rsrp": "RSRP",
    "rsrq": "RSRQ",
    "sinr": "SINR",

    # Modulation and coding scheme
    "pdsch_mcs": "PDSCH_MCS",
    "pusch_mcs": "PUSCH_MCS",

    # Resource blocks
    "pdsch_prbs": "PDSCH PRBs",
    "pusch_prbs": "PUSCH PRBs",

    # Throughput
    "throughput_dl": "throughput_DL",
    "throughput_ul": "throughput_UL",

    # Labels (ground truth)
    "label_anomaly": "lab_anom",
    "label_interference": "lab_inf",
    "label_1rru": "lab_1rr",
}

# ETL-cleaned column names (after standardization)
COLUMN_NAMES_CLEAN = {
    "time": "time_s",
    "pci": "pci",
    "cell_id": "cell_id",
    "rsrp": "rsrp_dbm",
    "rsrq": "rsrq_db",
    "sinr": "sinr_db",
    "pdsch_mcs": "mcs_dl",
    "pusch_mcs": "mcs_ul",
    "pdsch_prbs": "prb_dl",
    "pusch_prbs": "prb_ul",
    "throughput_dl": "app_dl_mbps",
    "throughput_ul": "app_ul_mbps",
    "label_anomaly": "lab_anom",
    "label_interference": "lab_inf",
    "label_1rru": "lab_1rr",
}

# Default to raw column names for backward compatibility
COLUMN_NAMES = COLUMN_NAMES_RAW

# ==============================================================================
# REQUIRED COLUMNS
# ==============================================================================
# These columns MUST exist in the dataset

# Required columns for RAW data
REQUIRED_COLUMNS = [
    "Time",
    "RSRP",
    "RSRQ",
    "SINR",
    "PDSCH_MCS",
    "throughput_DL",
    "lab_anom",
]

# Required columns for CLEANED/ETL data
REQUIRED_COLUMNS_CLEAN = [
    "time_s",
    "rsrp_dbm",
    "rsrq_db",
    "sinr_db",
    "mcs_dl",
    "app_dl_mbps",
    "lab_anom",
]

# ==============================================================================
# VALIDATION RULES
# ==============================================================================

# 1. NULL VALUE CHECKS
# Columns that should never be null
NON_NULL_COLUMNS = [
    "Time",
    "RSRP",
    "RSRQ",
    "SINR",
    "lab_anom",
]

# Non-null columns for CLEANED/ETL data
NON_NULL_COLUMNS_CLEAN = [
    "time_s",
    "rsrp_dbm",
    "rsrq_db",
    "sinr_db",
    "lab_anom",
]

# Columns that can have some nulls (max percentage allowed)
NULLABLE_COLUMNS = {
    "PDSCH_MCS": 0.05,      # Allow up to 5% nulls
    "PUSCH_MCS": 0.05,
    "throughput_DL": 0.02,  # Allow up to 2% nulls
    "throughput_UL": 0.10,  # Allow up to 10% nulls
}

# Nullable columns for CLEANED/ETL data
NULLABLE_COLUMNS_CLEAN = {
    "mcs_dl": 0.05,         # Allow up to 5% nulls
    "mcs_ul": 0.05,
    "app_dl_mbps": 0.02,    # Allow up to 2% nulls
    "app_ul_mbps": 0.10,    # Allow up to 10% nulls
}

# 2. NUMERIC RANGE CHECKS
# Based on client's recommendations (Section 2.5.2) and 3GPP standards
# Purpose: Validate data quality using Great Expectations
# These ranges catch broken measurements (e.g., RSRP = +100 dBm, SINR = -500 dB)

NUMERIC_RANGES = {
    # Signal strength (dBm) - 3GPP TS 38.133
    "RSRP": {
        "min": -156,  # 3GPP TS 38.133 absolute minimum
        "max": -31,   # Physical receive sensitivity to typical upper bound
        "strict": False,  # Allow 5% outliers
        "description": "Reference Signal Received Power (3GPP + client bounds)"
    },
    
    # Signal quality (dB) - 3GPP TS 38.133  
    "RSRQ": {
        "min": -20,   # Empirical radio quality envelope
        "max": +3,    # Empirical radio quality envelope upper bound
        "strict": False,  # Allow 5% outliers
        "description": "Reference Signal Received Quality (empirical bounds)"
    },
    
    # Signal-to-Interference-plus-Noise Ratio (dB)
    "SINR": {
        "min": -10,   # Indoor testbed limit
        "max": 30,    # Indoor testbed limit
        "strict": False,  # Allow 5% outliers
        "description": "SINR - PRIMARY ANOMALY INDICATOR (indoor testbed limits)"
    },
    
    # Modulation and Coding Scheme (0-28 for 5G NR)
    "PDSCH_MCS": {
        "min": 0,
        "max": 28,
        "strict": True,  # Must be within range (no outliers allowed)
        "description": "Downlink MCS - Valid NR MCS index range (3GPP TS 38.214)"
    },
    
    "PUSCH_MCS": {
        "min": 0,
        "max": 28,
        "strict": True,  # Must be within range (no outliers allowed)
        "description": "Uplink MCS - Valid NR MCS index range (3GPP TS 38.214)"
    },
    
    # Physical Resource Blocks (0-273 for 100 MHz BW)
    "PDSCH PRBs": {
        "min": 0,
        "max": 273,
        "strict": True,  # Must be within range (no outliers allowed)
        "description": "Downlink PRBs - Max RBs for 100 MHz (3GPP TS 38.101)"
    },
    
    "PUSCH PRBs": {
        "min": 0,
        "max": 273,
        "strict": True,  # Must be within range (no outliers allowed)
        "description": "Uplink PRBs - Max RBs for 100 MHz (3GPP TS 38.101)"
    },
    
    # Throughput (Mbps) - Based on testbed capabilities
    "throughput_DL": {
        "min": 0,
        "max": 100000000,   # Avoid corrupt iPerf3 logs (client's recommendation)
        "strict": False,  # Allow 5% outliers
        "description": "Downlink throughput - Testbed practical limit"
    },
    
    "throughput_UL": {
        "min": 0,
        "max": 1000,   # Conservative practical limit
        "strict": False,  # Allow 5% outliers
        "description": "Uplink throughput - Testbed practical limit"
    },
}

# Numeric ranges for CLEANED/ETL data (same logic, different column names)
NUMERIC_RANGES_CLEAN = {
    "rsrp_dbm": {
        "min": -156,
        "max": -31,
        "strict": False,
        "description": "Reference Signal Received Power (3GPP + client bounds)"
    },

    "rsrq_db": {
        "min": -20,
        "max": +3,
        "strict": False,
        "description": "Reference Signal Received Quality (empirical bounds)"
    },

    "sinr_db": {
        "min": -10,
        "max": 30,
        "strict": False,
        "description": "SINR - PRIMARY ANOMALY INDICATOR (indoor testbed limits)"
    },

    "mcs_dl": {
        "min": 0,
        "max": 28,
        "strict": True,
        "description": "Downlink MCS - Valid NR MCS index range (3GPP TS 38.214)"
    },

    "mcs_ul": {
        "min": 0,
        "max": 28,
        "strict": True,
        "description": "Uplink MCS - Valid NR MCS index range (3GPP TS 38.214)"
    },

    "prb_dl": {
        "min": 0,
        "max": 273,
        "strict": True,
        "description": "Downlink PRBs - Max RBs for 100 MHz (3GPP TS 38.101)"
    },

    "prb_ul": {
        "min": 0,
        "max": 273,
        "strict": True,
        "description": "Uplink PRBs - Max RBs for 100 MHz (3GPP TS 38.101)"
    },

    "app_dl_mbps": {
        "min": 0,
        "max": 100000000,
        "strict": False,
        "description": "Downlink throughput - Testbed practical limit"
    },

    "app_ul_mbps": {
        "min": 0,
        "max": 1000,
        "strict": False,
        "description": "Uplink throughput - Testbed practical limit"
    },
}

# 3. CATEGORICAL VALUE CHECKS
CATEGORICAL_VALUES = {
    # Binary labels (0 or 1)
    "lab_anom": {
        "values": [0, 1],
        "description": "Anomaly label (0=normal, 1=anomaly)"
    },
    
    "lab_inf": {
        "values": [0, 1],
        "description": "Interference label (0=no interference, 1=interference)"
    },
    
    "lab_1rr": {
        "values": [0, 1],
        "description": "Single RRU scenario (0=multi-RRU, 1=single-RRU)"
    },
}

# Categorical values for CLEANED/ETL data (same as raw - labels don't change)
CATEGORICAL_VALUES_CLEAN = {
    "lab_anom": {
        "values": [0, 1],
        "description": "Anomaly label (0=normal, 1=anomaly)"
    },

    "lab_inf": {
        "values": [0, 1],
        "description": "Interference label (0=no interference, 1=interference)"
    },

    "lab_1rr": {
        "values": [0, 1],
        "description": "Single RRU scenario (0=multi-RRU, 1=single-RRU)"
    },
}

# 4. ANOMALY-SPECIFIC VALIDATION RULES
# Based on your analysis: anomalies show 50-75% SINR drops

ANOMALY_SIGNATURE_RULES = {
    "sinr_normal_range": {
        "metric": "SINR",
        "condition": "lab_anom == 0",
        "expected_min": 10,
        "expected_max": 35,
        "description": "Normal samples should have SINR 10-35 dB"
    },
    
    "sinr_anomaly_range": {
        "metric": "SINR", 
        "condition": "lab_anom == 1",
        "expected_min": -10,
        "expected_max": 15,
        "description": "Anomaly samples typically have SINR < 15 dB"
    },
    
    "throughput_normal_range": {
        "metric": "throughput_DL",
        "condition": "lab_anom == 0",
        "expected_min": 80,
        "expected_max": 400,
        "description": "Normal samples should have throughput > 80 Mbps"
    },
    
    "throughput_anomaly_range": {
        "metric": "throughput_DL",
        "condition": "lab_anom == 1",
        "expected_min": 0,
        "expected_max": 150,
        "description": "Anomaly samples typically have reduced throughput"
    },
}

# Anomaly signature rules for CLEANED/ETL data
ANOMALY_SIGNATURE_RULES_CLEAN = {
    "sinr_normal_range": {
        "metric": "sinr_db",
        "condition": "lab_anom == 0",
        "expected_min": 10,
        "expected_max": 35,
        "description": "Normal samples should have SINR 10-35 dB"
    },

    "sinr_anomaly_range": {
        "metric": "sinr_db",
        "condition": "lab_anom == 1",
        "expected_min": -10,
        "expected_max": 15,
        "description": "Anomaly samples typically have SINR < 15 dB"
    },

    "throughput_normal_range": {
        "metric": "app_dl_mbps",
        "condition": "lab_anom == 0",
        "expected_min": 80,
        "expected_max": 400,
        "description": "Normal samples should have throughput > 80 Mbps"
    },

    "throughput_anomaly_range": {
        "metric": "app_dl_mbps",
        "condition": "lab_anom == 1",
        "expected_min": 0,
        "expected_max": 150,
        "description": "Anomaly samples typically have reduced throughput"
    },
}

# 5. DATA QUALITY THRESHOLDS
QUALITY_THRESHOLDS = {
    "min_samples": 5000,               # Minimum samples required
    "max_duplicate_rows": 0.01,        # Max 1% duplicate rows
    "min_anomaly_rate": 0.20,          # Expect at least 20% anomalies
    "max_anomaly_rate": 0.60,          # Expect at most 60% anomalies
    "max_null_percentage_overall": 0.05,  # Max 5% nulls across dataset
}

# 6. SCENARIO-SPECIFIC RULES
# Different scenarios have different characteristics
SCENARIO_RULES = {
    "Lvl4_AllRRUOn": {
        "expected_samples_min": 2000,
        "expected_anomaly_rate": (0.30, 0.40),  # 30-40% (actual ~35.6%)
    },
    
    "Lvl5_AllRRUOn": {
        "expected_samples_min": 1800,
        "expected_anomaly_rate": (0.50, 0.65),  # 50-65% (high interference)
    },
    
    "Lvl6_1RRUOn": {
        "expected_samples_min": 2000,
        "expected_anomaly_rate": (0.35, 0.45),  # 35-45% (weak coverage)
    },
    
    "Lvl6_AllRRUOn": {
        "expected_samples_min": 2000,
        "expected_anomaly_rate": (0.45, 0.55),  # 45-55%
    },
}

# ==============================================================================
# REPORTING CONFIGURATION
# ==============================================================================

REPORT_CONFIG = {
    "output_dir": "reports/validation",
    "plots_dir": "reports/plots",
    "generate_html": True,
    "generate_json": True,
    "generate_plots": True,
    "log_to_mlflow": True,
}

# ==============================================================================
# MLFLOW CONFIGURATION
# ==============================================================================

MLFLOW_CONFIG = {
    "tracking_uri": "./mlruns",
    "experiment_name": "watchtower_data_validation",
    "artifact_location": "mlruns",
    "log_plots": True,
    "log_validation_results": True,
    "log_data_profile": True,
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_column_name(key: str) -> str:
    """Get actual column name from configuration key."""
    return COLUMN_NAMES.get(key, key)

def get_all_required_columns() -> List[str]:
    """Get list of all required columns."""
    return REQUIRED_COLUMNS

def get_validation_rules() -> Dict[str, Any]:
    """Get all validation rules in a structured format."""
    return {
        "required_columns": REQUIRED_COLUMNS,
        "non_null_columns": NON_NULL_COLUMNS,
        "nullable_columns": NULLABLE_COLUMNS,
        "numeric_ranges": NUMERIC_RANGES,
        "categorical_values": CATEGORICAL_VALUES,
        "anomaly_signatures": ANOMALY_SIGNATURE_RULES,
        "quality_thresholds": QUALITY_THRESHOLDS,
        "scenario_rules": SCENARIO_RULES,
    }

def get_critical_features(clean: bool = False) -> List[str]:
    """Get list of critical features for anomaly detection."""
    if clean:
        return ["sinr_db", "rsrp_dbm", "rsrq_db", "app_dl_mbps", "mcs_dl"]
    return ["SINR", "RSRP", "RSRQ", "throughput_DL", "PDSCH_MCS"]

# ==============================================================================
# VALIDATION SEVERITY LEVELS
# ==============================================================================

SEVERITY_LEVELS = {
    "CRITICAL": ["required_columns", "non_null_columns", "categorical_values"],
    "HIGH": ["numeric_ranges_strict", "quality_thresholds"],
    "MEDIUM": ["numeric_ranges_non_strict", "anomaly_signatures"],
    "LOW": ["nullable_columns", "scenario_rules"],
}
