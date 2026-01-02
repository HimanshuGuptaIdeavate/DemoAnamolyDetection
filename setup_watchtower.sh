#!/bin/bash

###############################################################################
# WATCHTOWER Project Setup Script
# Purpose: One-time initialization for 5G Drone Anomaly Detection System
###############################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:-/home/himanshu/BlinklyRelated/WatchTowerRND/DemoAnamolyDetection}"
SUTD_REPO_URL="https://github.com/FCCLab/sutd_5g_dataset_2023.git"

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 not installed"
        exit 1
    fi
}

# Main script
echo ""
echo "WATCHTOWER Setup - 5G Drone Anomaly Detection"
echo "=============================================="
echo ""

log_info "Checking prerequisites..."
check_command python3
check_command git

mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

log_info "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

log_info "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet dvc mlflow pandas numpy scikit-learn xgboost torch shap optuna

log_info "Downloading SUTD dataset..."
TEMP_DIR="$PROJECT_ROOT/data/sutd_temp"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"
git init --quiet
git remote add origin "$SUTD_REPO_URL"
git fetch --depth=1 origin dataset --quiet
git checkout FETCH_HEAD --quiet
SUTD_COMMIT=$(git rev-parse HEAD)
echo "$SUTD_COMMIT" > "$PROJECT_ROOT/data/SUTD_VERSION.txt"

mkdir -p "$PROJECT_ROOT/data/raw/sutd"
find . -name "*.csv" -exec cp {} "$PROJECT_ROOT/data/raw/sutd/" \;
cd "$PROJECT_ROOT"
rm -rf "$TEMP_DIR"

log_success "Downloaded $(ls -1 data/raw/sutd/*.csv 2>/dev/null | wc -l) CSV files"

log_info "Initializing DVC..."
if [ ! -d ".dvc" ]; then dvc init --quiet; fi
dvc add data/raw/sutd 2>/dev/null || dvc add data/raw/sutd --force

log_info "Initializing Git..."
if [ ! -d ".git" ]; then 
    git init --quiet
    git config user.name "himanshuguptahemu"
    git config user.email "himanshu.gupta@ideavate.com"
fi

cat > .gitignore << 'EOF'
__pycache__/
.venv/
mlruns/
/data/raw/sutd
/data/interim/
/data/processed/
.ipynb_checkpoints/
EOF

git add .gitignore data/raw/sutd.dvc .dvc 2>/dev/null || true
git commit -m "Initial WATCHTOWER setup" --quiet 2>/dev/null || true

log_info "Creating project structure..."
mkdir -p src/watchtower/{data,features,models,training,evaluation,serving}
mkdir -p {configs,notebooks,scripts,tests,artifacts,mlruns,docs}
touch src/watchtower/__init__.py

log_success "Setup complete!"
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  ls -lh data/raw/sutd/"
echo ""
