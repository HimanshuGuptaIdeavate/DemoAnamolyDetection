#!/bin/bash

################################################################################
# WATCHTOWER - Full Training Pipeline
# Runs complete ML pipeline from raw data to trained model
#
# Pipeline Steps:
#   1. ETL (Extract, Transform, Load)
#   2. Windowing (5-second windows at 2Hz)
#   3. Feature Engineering (27 numeric + 2 categorical features)
#   4. Preprocessing (StandardScaler + OneHotEncoder)
#   5. XGBoost Training (GroupKFold CV + Threshold Tuning)
#
# Usage:
#   bash scripts/run_full_training_pipeline.sh
#   OR
#   chmod +x scripts/run_full_training_pipeline.sh
#   ./scripts/run_full_training_pipeline.sh
#
# Author: Himanshu's WatchTower Project
# Date: 2026-02-06
################################################################################

set -e  # Exit immediately if any command fails
set -u  # Exit if undefined variable is used
set -o pipefail  # Exit if any command in a pipe fails

################################################################################
# CONFIGURATION
################################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Pipeline scripts
SCRIPT_DIR="scripts"
ETL_SCRIPT="${SCRIPT_DIR}/run_etl_pipeline.py"
WINDOWING_SCRIPT="${SCRIPT_DIR}/run_windowing_pipeline.py"
FEATURES_SCRIPT="${SCRIPT_DIR}/run_build_features.py"
PREPROCESSING_SCRIPT="${SCRIPT_DIR}/run_preprocessing_pipeline.py"
TRAINING_SCRIPT="${SCRIPT_DIR}/run_xgboost_training.py"

# Log directory
LOG_DIR="logs/pipeline_runs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/full_pipeline_${TIMESTAMP}.log"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

################################################################################
# HELPER FUNCTIONS
################################################################################

# Print colored banner
print_banner() {
    local color=$1
    local message=$2
    echo -e "${color}"
    echo "================================================================================"
    echo "  $message"
    echo "================================================================================"
    echo -e "${NC}"
}

# Print step header
print_step() {
    local step_num=$1
    local step_name=$2
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}▶ STEP ${step_num}/5: ${step_name}${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# Print success message
print_success() {
    local message=$1
    echo -e "${GREEN}✅ $message${NC}"
}

# Print error message
print_error() {
    local message=$1
    echo -e "${RED}❌ ERROR: $message${NC}"
}

# Print warning message
print_warning() {
    local message=$1
    echo -e "${YELLOW}⚠️  WARNING: $message${NC}"
}

# Print info message
print_info() {
    local message=$1
    echo -e "${BLUE}ℹ️  $message${NC}"
}

# Run Python script with error handling
run_python_script() {
    local script=$1
    local step_name=$2
    local start_time=$(date +%s)
    
    echo -e "${PURPLE}Running: python $script${NC}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Run script and capture output to both console and log file
    if python "$script" 2>&1 | tee -a "$LOG_FILE"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo ""
        print_success "$step_name completed successfully in ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo ""
        print_error "$step_name failed after ${duration}s"
        print_error "Check log file: $LOG_FILE"
        return 1
    fi
}

# Check if script exists
check_script_exists() {
    local script=$1
    local step_name=$2
    
    if [ ! -f "$script" ]; then
        print_error "Script not found: $script"
        print_error "Cannot run $step_name"
        exit 1
    fi
}

# Display pipeline summary
display_summary() {
    local total_duration=$1
    
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ PIPELINE COMPLETE!${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${BLUE}📊 Pipeline Summary:${NC}"
    echo -e "   Total Duration: ${total_duration}s ($(($total_duration / 60))m $(($total_duration % 60))s)"
    echo -e "   Timestamp: $TIMESTAMP"
    echo -e "   Log File: $LOG_FILE"
    echo ""
    echo -e "${BLUE}📁 Generated Artifacts:${NC}"
    echo -e "   ├─ data/clean_data.parquet"
    echo -e "   ├─ data/windows.parquet"
    echo -e "   ├─ data/features_table.parquet"
    echo -e "   ├─ data/X.npy, y.npy, groups.npy"
    echo -e "   ├─ artifacts/scaler_numeric.pkl"
    echo -e "   ├─ artifacts/encoder_categorical.pkl"
    echo -e "   └─ models/xgboost_model_*.joblib"
    echo ""
    echo -e "${BLUE}📈 Next Steps:${NC}"
    echo -e "   1. View results in MLflow:"
    echo -e "      ${PURPLE}mlflow ui --backend-store-uri file://mlruns${NC}"
    echo -e "      Then open: ${CYAN}http://localhost:5000${NC}"
    echo ""
    echo -e "   2. Check training plots:"
    echo -e "      ${PURPLE}ls -lh reports/plots/${NC}"
    echo ""
    echo -e "   3. Deploy model:"
    echo -e "      ${PURPLE}python src/watchtower/api/serve.py${NC}"
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

################################################################################
# PRE-FLIGHT CHECKS
################################################################################

print_banner "${PURPLE}" "WATCHTOWER - FULL TRAINING PIPELINE"

echo -e "${BLUE}Running pre-flight checks...${NC}\n"

# Check if we're in the right directory
if [ ! -d "$SCRIPT_DIR" ]; then
    print_error "Scripts directory not found: $SCRIPT_DIR"
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check if all scripts exist
print_info "Checking if all pipeline scripts exist..."
check_script_exists "$ETL_SCRIPT" "ETL Pipeline"
check_script_exists "$WINDOWING_SCRIPT" "Windowing Pipeline"
check_script_exists "$FEATURES_SCRIPT" "Feature Engineering"
check_script_exists "$PREPROCESSING_SCRIPT" "Preprocessing Pipeline"
check_script_exists "$TRAINING_SCRIPT" "XGBoost Training"
print_success "All pipeline scripts found"

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please ensure Python is installed and in PATH"
    exit 1
fi
print_success "Python found: $(python --version)"

# Check if virtual environment is activated (optional warning)
if [ -z "${VIRTUAL_ENV:-}" ]; then
    print_warning "No virtual environment detected"
    print_warning "Consider activating your virtual environment: source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Pipeline cancelled by user"
        exit 0
    fi
fi

print_success "Pre-flight checks passed"
echo ""

# Confirm before starting
print_info "This will run the complete training pipeline (5 steps)"
print_info "Estimated time: 5-15 minutes depending on data size"
echo ""
read -p "Start pipeline? (Y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Nn]$ ]]; then
    print_info "Pipeline cancelled by user"
    exit 0
fi

echo ""
print_info "Pipeline started at: $(date '+%Y-%m-%d %H:%M:%S')"
print_info "Logs will be saved to: $LOG_FILE"
echo ""

# Start timer
PIPELINE_START_TIME=$(date +%s)

################################################################################
# PIPELINE EXECUTION
################################################################################

# Step 1: ETL
print_step 1 "ETL (Extract, Transform, Load)"
if ! run_python_script "$ETL_SCRIPT" "ETL Pipeline"; then
    print_error "Pipeline failed at Step 1: ETL"
    exit 1
fi

# Step 2: Windowing
print_step 2 "WINDOWING (5-second windows at 2Hz)"
if ! run_python_script "$WINDOWING_SCRIPT" "Windowing Pipeline"; then
    print_error "Pipeline failed at Step 2: Windowing"
    exit 1
fi

# Step 3: Feature Engineering
print_step 3 "FEATURE ENGINEERING (27 numeric + 2 categorical)"
if ! run_python_script "$FEATURES_SCRIPT" "Feature Engineering"; then
    print_error "Pipeline failed at Step 3: Feature Engineering"
    exit 1
fi

# Step 4: Preprocessing
print_step 4 "PREPROCESSING (StandardScaler + OneHotEncoder)"
if ! run_python_script "$PREPROCESSING_SCRIPT" "Preprocessing Pipeline"; then
    print_error "Pipeline failed at Step 4: Preprocessing"
    exit 1
fi

# Step 5: XGBoost Training
print_step 5 "XGBOOST TRAINING (GroupKFold CV + Threshold Tuning)"
if ! run_python_script "$TRAINING_SCRIPT" "XGBoost Training"; then
    print_error "Pipeline failed at Step 5: XGBoost Training"
    exit 1
fi

################################################################################
# PIPELINE COMPLETION
################################################################################

# Calculate total duration
PIPELINE_END_TIME=$(date +%s)
TOTAL_DURATION=$((PIPELINE_END_TIME - PIPELINE_START_TIME))

# Display summary
display_summary "$TOTAL_DURATION"

# Save pipeline metadata
METADATA_FILE="${LOG_DIR}/pipeline_metadata_${TIMESTAMP}.json"
cat > "$METADATA_FILE" << EOF
{
  "pipeline_run": {
    "timestamp": "$TIMESTAMP",
    "start_time": "$(date -d @$PIPELINE_START_TIME '+%Y-%m-%d %H:%M:%S')",
    "end_time": "$(date -d @$PIPELINE_END_TIME '+%Y-%m-%d %H:%M:%S')",
    "duration_seconds": $TOTAL_DURATION,
    "status": "success"
  },
  "steps": {
    "1_etl": {
      "script": "$ETL_SCRIPT",
      "status": "success"
    },
    "2_windowing": {
      "script": "$WINDOWING_SCRIPT",
      "status": "success"
    },
    "3_features": {
      "script": "$FEATURES_SCRIPT",
      "status": "success"
    },
    "4_preprocessing": {
      "script": "$PREPROCESSING_SCRIPT",
      "status": "success"
    },
    "5_training": {
      "script": "$TRAINING_SCRIPT",
      "status": "success"
    }
  },
  "artifacts": {
    "data": [
      "data/clean_data.parquet",
      "data/windows.parquet",
      "data/features_table.parquet",
      "data/X.npy",
      "data/y.npy",
      "data/groups.npy"
    ],
    "models": [
      "models/xgboost_model_*.joblib"
    ],
    "preprocessing": [
      "artifacts/scaler_numeric.pkl",
      "artifacts/encoder_categorical.pkl"
    ]
  },
  "log_file": "$LOG_FILE"
}
EOF

print_success "Pipeline metadata saved: $METADATA_FILE"

exit 0