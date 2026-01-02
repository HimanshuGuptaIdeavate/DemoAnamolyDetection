# 📊 WATCHTOWER Validation & MLflow Guide

Complete guide for data validation and experiment tracking.

## 🎯 What You Have

After setup, you now have:

1. **validation_config.py** - Centralized configuration for all validation rules
2. **validate_data.py** - Great Expectations validation suite
3. **mlflow_logger.py** - MLflow integration for experiment tracking
4. **run_validation_pipeline.py** - Master script to run everything

---

## 🚀 Quick Start

### One-Command Execution

```bash
# Activate environment
source .venv/bin/activate

# Copy validation files to project
cp validation_config.py src/watchtower/data/
cp validate_data.py src/watchtower/data/
cp mlflow_logger.py src/watchtower/utils/
cp run_validation_pipeline.py scripts/

# Run complete pipeline
python scripts/run_validation_pipeline.py data/raw/sutd
```

**This will:**
1. ✅ Validate schema (required columns)
2. ✅ Check null values
3. ✅ Validate numeric ranges (3GPP standards)
4. ✅ Validate categorical values
5. ✅ Check anomaly signatures
6. ✅ Assess data quality
7. ✅ Validate scenario-specific rules
8. ✅ Generate visualizations
9. ✅ Log everything to MLflow
10. ✅ Save JSON + HTML reports

---

## 📁 File Placement in Your Project

```
watchtower/
├── src/watchtower/
│   ├── data/
│   │   ├── validation_config.py    ← Config file
│   │   └── validate_data.py        ← Validator
│   └── utils/
│       └── mlflow_logger.py        ← MLflow logger
│
└── scripts/
    └── run_validation_pipeline.py  ← Master script
```

---

## 🔧 Configuration (validation_config.py)

All validation rules are in one place for easy modification:

### To Change Column Names

```python
# In validation_config.py
COLUMN_NAMES = {
    "sinr": "SINR",              # Change to "SS_SINR_dB" if needed
    "rsrp": "RSRP",              # Change to "SS_RSRP_dBm" if needed
    "throughput_dl": "throughput_DL",  # Update as needed
}
```

### To Modify Validation Rules

```python
# In validation_config.py

# 1. Adjust numeric ranges
NUMERIC_RANGES = {
    "SINR": {
        "min": -30,          # Change minimum
        "max": 40,           # Change maximum
        "strict": False,     # Set to True for 100% compliance
    }
}

# 2. Change null tolerance
NULLABLE_COLUMNS = {
    "PDSCH_MCS": 0.05,   # Allow 5% nulls → change to 0.10 for 10%
}

# 3. Modify quality thresholds
QUALITY_THRESHOLDS = {
    "min_samples": 5000,  # Change minimum required samples
    "min_anomaly_rate": 0.20,  # Adjust expected anomaly rate
}
```

### To Add New Validations

```python
# In validation_config.py

# Add new metric to validate
NUMERIC_RANGES["new_metric"] = {
    "min": 0,
    "max": 100,
    "strict": True,
    "description": "Description of new metric"
}

# Add to required columns
REQUIRED_COLUMNS.append("new_metric")
```

---

## 🔍 Validation Checks Performed

### 1. Schema Validation
- Required columns exist
- Column types are correct

### 2. Null Value Checks
- Critical columns have no nulls
- Other columns within null thresholds

### 3. Numeric Range Validation (3GPP Standards)
- **RSRP**: -140 to -44 dBm
- **RSRQ**: -20 to -3 dB
- **SINR**: -30 to 40 dB (PRIMARY INDICATOR)
- **MCS**: 0 to 28
- **PRBs**: 0 to 273
- **Throughput**: 0 to 1000 Mbps

### 4. Categorical Value Checks
- Labels are binary (0 or 1)
- No unexpected categories

### 5. Anomaly Signature Validation
- Normal samples: SINR > 10 dB
- Anomaly samples: SINR typically < 15 dB
- SINR separation ≥ 10 dB between normal/anomaly
- Throughput drop ≥ 30% for anomalies

### 6. Data Quality Metrics
- Minimum 5,000 samples
- Duplicate rows < 1%
- Anomaly rate 20-60%
- Overall nulls < 5%

### 7. Scenario-Specific Rules
- Each scenario has expected sample count
- Each scenario has expected anomaly rate

---

## 📊 MLflow Tracking

### What Gets Logged

**Parameters:**
- Dataset rows, columns, size
- SUTD dataset commit hash
- Number of source files
- Validation timestamp

**Metrics:**
- Validation success (0/1)
- Total/passed/failed checks
- Validation score (pass rate)
- Category-wise pass rates
- SINR, RSRP, RSRQ, throughput stats
- Anomaly rate, interference rate

**Artifacts:**
- Validation results (JSON)
- Data profile (JSON)
- Signal distributions plot
- Anomaly comparison plot
- Correlation heatmap
- Time series sample plot

### View MLflow UI

```bash
# Start MLflow server
mlflow ui --backend-store-uri mlruns

# Open in browser
# http://localhost:5000
```

---

## 🎨 Generated Visualizations

### 1. Signal Distributions
- Histograms of RSRP, RSRQ, SINR, MCS, PRBs, Throughput
- Mean lines and statistics

### 2. Anomaly Comparison
- Box plots: Normal vs Anomaly
- Shows feature separation
- Sample sizes included

### 3. Correlation Heatmap
- Feature correlations
- Identifies relationships

### 4. Time Series Sample
- SINR, RSRP, Throughput over time
- Anomaly threshold lines
- 1000-sample window

All saved to `reports/plots/` and logged to MLflow.

---

## 💻 Usage Examples

### Basic Usage

```bash
# Run validation on default data path
python scripts/run_validation_pipeline.py
```

### Custom Data Path

```bash
# Validate specific CSV
python scripts/run_validation_pipeline.py data/raw/sutd/Lvl5_AllRRUOn_Anomaly_label.csv

# Validate different directory
python scripts/run_validation_pipeline.py /path/to/other/data
```

### Skip MLflow (Validation Only)

```bash
# Just validate, don't log to MLflow
python scripts/run_validation_pipeline.py --skip-mlflow
```

### Custom Run Name

```bash
# Name your MLflow run
python scripts/run_validation_pipeline.py --run-name "production_data_check"
```

### Programmatic Usage

```python
from src.watchtower.data.validate_data import WatchtowerValidator
from src.watchtower.utils.mlflow_logger import MLflowLogger

# Run validation
validator = WatchtowerValidator()
success, results = validator.run_validation("data/raw/sutd")

# Log to MLflow
df = validator.load_data("data/raw/sutd")
logger = MLflowLogger()
run_id = logger.log_data_validation(df, results)

print(f"Validation: {'PASSED' if success else 'FAILED'}")
print(f"MLflow run: {run_id}")
```

---

## 📈 Expected Output

```
================================================================================
WATCHTOWER DATA VALIDATION & MLFLOW PIPELINE
================================================================================
Start time: 2024-12-22 18:30:00
Data path: data/raw/sutd
================================================================================

STEP 1: Running Great Expectations Validation

================================================================================
WATCHTOWER DATA VALIDATION
================================================================================
✅ Loaded 8,732 total samples from 4 files

🔍 Validating Schema...
  ✓ Time
  ✓ RSRP
  ✓ RSRQ
  ✓ SINR
  ✓ PDSCH_MCS
  ✓ throughput_DL
  ✓ lab_anom

🔍 Validating Null Values...
  ✓ Time: No nulls
  ✓ RSRP: No nulls
  ✓ RSRQ: No nulls
  ✓ SINR: No nulls
  ✓ lab_anom: No nulls
  ✓ PDSCH_MCS: 1.2% nulls (≤5% allowed)

🔍 Validating Numeric Ranges (3GPP Standards)...
  ✓ RSRP: [-140, -44] (95%)
  ✓ RSRQ: [-20, -3] (95%)
  ✓ SINR: [-30, 40] (95%)
  ✓ PDSCH_MCS: [0, 28] (strict)
  ✓ throughput_DL: [0, 1000] (95%)

🔍 Validating Categorical Values...
  ✓ lab_anom: [0, 1]
  ✓ lab_inf: [0, 1]

🔍 Validating Anomaly Signatures...
  ✓ SINR separation: 18.3 dB (≥10 dB)
  ✓ Anomaly SINR < 15 dB: 72.4%
  ✓ Throughput drop: 54.2%

🔍 Validating Data Quality...
  ✓ Samples: 8,732 (≥5,000)
  ✓ Duplicates: 0.1% (≤1%)
  ✓ Anomaly rate: 42.8% (20-60%)
  ✓ Overall nulls: 1.8% (≤5%)

🔍 Validating Scenario-Specific Rules...
  ✓ Lvl4_AllRRUOn: 2,323 samples, 24.9% anomalies
  ✓ Lvl5_AllRRUOn: 2,031 samples, 58.6% anomalies
  ✓ Lvl6_1RRUOn: 2,156 samples, 39.9% anomalies
  ✓ Lvl6_AllRRUOn: 2,222 samples, 50.0% anomalies

================================================================================
VALIDATION SUMMARY: 45/45 checks passed
================================================================================
✅ All validation checks passed!

📄 Validation report saved: reports/validation/validation_report_20241222_183000.json

================================================================================
STEP 2: Logging to MLflow

📊 MLflow initialized: experiment='watchtower_data_validation'
   Tracking URI: file://mlruns
   Experiment ID: 1

🚀 Starting MLflow run: data_validation_20241222_183000
   Run ID: abc123def456...
  📝 Logging dataset parameters...
  📊 Logging validation metrics...
  📈 Logging summary statistics...
  📊 Generating and logging visualizations...
  
✅ MLflow run completed: abc123def456...
   View at: mlflow ui --backend-store-uri mlruns

================================================================================
PIPELINE SUMMARY
================================================================================
✅ Validation: PASSED
   - Total checks: 45
   - Passed: 45
   - Failed: 0
   - Score: 100.0%

✅ MLflow: Logged successfully
   - View UI: mlflow ui --backend-store-uri mlruns
   - Then open: http://localhost:5000

📄 Reports saved to: reports/validation/
📊 Plots saved to: reports/plots/

End time: 2024-12-22 18:31:00
================================================================================
```

---

## 🔧 Troubleshooting

### Issue: Import errors

```bash
# Make sure files are in correct locations
# and watchtower is importable
cd /path/to/DemoAnamolyDetection
pip install -e .
```

### Issue: Great Expectations not installed

```bash
pip install great-expectations
```

### Issue: Matplotlib errors

```bash
pip install matplotlib seaborn
```

### Issue: MLflow UI won't start

```bash
# Check if port 5000 is already in use
mlflow ui --backend-store-uri mlruns --port 5001
```

### Issue: Validation fails

```bash
# Run with verbose output
python -u scripts/run_validation_pipeline.py data/raw/sutd 2>&1 | tee validation.log
```

---

## 📚 Next Steps

After successful validation:

1. ✅ **Data Validation Complete** ← You are here
2. ⏭️ **Data Preprocessing** (windowing, cleaning)
3. ⏭️ **Feature Engineering** (temporal features)
4. ⏭️ **Model Training** (XGBoost)
5. ⏭️ **Evaluation** (metrics, SHAP)
6. ⏭️ **Deployment** (FastAPI)

---

## 🎯 Key Takeaways

✅ **Centralized Configuration**: All rules in `validation_config.py`
✅ **Comprehensive Checks**: 7 types of validation
✅ **3GPP Compliant**: Follows industry standards
✅ **Anomaly-Aware**: Validates drone signature patterns
✅ **MLflow Integration**: Full experiment tracking
✅ **Visual Reports**: Plots + JSON + logs
✅ **Easy to Modify**: Change column names/rules in one place
✅ **Production-Ready**: Error handling, logging, artifacts

---

**Status**: ✅ Ready to validate!
**Time**: ~30 seconds for 8,732 samples
**Next Instruction**: Data preprocessing & windowing
