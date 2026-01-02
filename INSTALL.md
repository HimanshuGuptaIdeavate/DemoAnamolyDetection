# 🚀 WATCHTOWER Installation Guide

Complete setup instructions for the 5G Drone Anomaly Detection System.

## Prerequisites

- **OS**: Ubuntu 24.04 LTS (recommended) or any Linux distribution
- **Python**: 3.10 or higher
- **Git**: For version control and dataset download
- **Disk Space**: ~2 GB for dataset and artifacts
- **RAM**: Minimum 8 GB (16 GB recommended)

## Quick Installation

### Method 1: Automated Setup (Recommended)

```bash
# 1. Navigate to your project directory
cd /home/himanshu/BlinklyRelated/WatchTowerRND/DemoAnamolyDetection

# 2. Download the setup script
# (Copy setup_watchtower.sh to your directory)

# 3. Make it executable
chmod +x setup_watchtower.sh

# 4. Run the setup
bash setup_watchtower.sh

# This will automatically:
# ✅ Create virtual environment
# ✅ Install all dependencies
# ✅ Download SUTD dataset
# ✅ Initialize DVC and Git
# ✅ Create project structure
# ✅ Validate dataset
```

**Expected Output:**
```
WATCHTOWER Setup - 5G Drone Anomaly Detection
==============================================

[INFO] Checking prerequisites...
[INFO] Creating virtual environment...
[SUCCESS] Virtual environment created
[INFO] Installing dependencies...
[SUCCESS] Environment setup complete

[INFO] Downloading SUTD dataset...
[SUCCESS] Dataset pinned to commit: abc1234
[SUCCESS] Extracted 4 CSV files

[INFO] Initializing DVC...
[SUCCESS] DVC initialized
[SUCCESS] Dataset added to DVC

[INFO] Validating dataset...
✅ Found 4 CSV files
  ✓ Lvl4_AllRRUOn_Anomaly_label.csv: 2,323 samples
  ✓ Lvl5_AllRRUOn_Anomaly_label.csv: 2,031 samples
  ✓ Lvl6_1RRUOn_Anomaly_label.csv: 2,156 samples
  ✓ Lvl6_AllRRUOn_Anomaly_label.csv: 2,222 samples

📊 Dataset Summary:
  Total samples: 8,732
  Anomaly samples: 3,741 (42.8%)

✅ Dataset validation PASSED
[SUCCESS] Setup complete!
```

### Method 2: Manual Setup

If the automated script fails, follow these manual steps:

#### Step 1: Create Virtual Environment

```bash
cd /home/himanshu/BlinklyRelated/WatchTowerRND/DemoAnamolyDetection
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 2: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel

# Install from requirements.txt
pip install -r requirements.txt

# OR install manually
pip install numpy pandas scikit-learn xgboost torch \
            mlflow dvc shap optuna fastapi uvicorn
```

#### Step 3: Download SUTD Dataset

```bash
mkdir -p data/sutd_temp
cd data/sutd_temp

# Clone repository
git init
git remote add origin https://github.com/FCCLab/sutd_5g_dataset_2023.git
git fetch --depth=1 origin main
git checkout FETCH_HEAD

# Record commit
git rev-parse HEAD > ../SUTD_VERSION.txt

# Extract CSV files
mkdir -p ../raw/sutd
find . -name "*.csv" -exec cp {} ../raw/sutd/ \;

cd ../..
rm -rf data/sutd_temp
```

#### Step 4: Initialize DVC

```bash
dvc init
dvc add data/raw/sutd

git add .dvc .gitignore data/.gitignore data/raw/sutd.dvc
git commit -m "Initialize WATCHTOWER with SUTD dataset"
```

#### Step 5: Create Project Structure

```bash
# Create directories
mkdir -p src/watchtower/{data,features,models,training,evaluation,serving,utils}
mkdir -p {configs,notebooks,scripts,tests,artifacts,mlruns,docs}

# Create __init__.py files
touch src/watchtower/__init__.py
touch src/watchtower/{data,features,models,training,evaluation,serving,utils}/__init__.py

# Make watchtower installable
pip install -e .
```

## Verification

### 1. Check Environment

```bash
# Activate if not already
source .venv/bin/activate

# Check Python
python --version
# Should output: Python 3.10+ 

# Check packages
python -c "import xgboost; print(f'XGBoost: {xgboost.__version__}')"
python -c "import mlflow; print(f'MLflow: {mlflow.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 2. Check Dataset

```bash
# List CSV files
ls -lh data/raw/sutd/*.csv

# Expected output:
# Lvl4_AllRRUOn_Anomaly_label.csv
# Lvl5_AllRRUOn_Anomaly_label.csv
# Lvl6_1RRUOn_Anomaly_label.csv
# Lvl6_AllRRUOn_Anomaly_label.csv

# Check DVC
dvc status
# Should output: Data and pipelines are up to date.
```

### 3. Validate Dataset

```bash
python << 'EOF'
import pandas as pd
from pathlib import Path

data_dir = Path("data/raw/sutd")
csvs = list(data_dir.glob("*.csv"))
print(f"Found {len(csvs)} CSV files")

for csv in csvs:
    df = pd.read_csv(csv, on_bad_lines='skip')
    print(f"{csv.name}: {len(df):,} samples")
EOF
```

### 4. Test Import

```bash
python -c "from watchtower import __version__; print(f'Watchtower: {__version__}')"
```

## Post-Installation

### View Setup Summary

```bash
cat SETUP_SUMMARY.md
```

### Start MLflow UI

```bash
mlflow ui
# Open http://localhost:5000
```

### Explore Data

```bash
# Start Jupyter
jupyter notebook notebooks/01_eda.ipynb

# OR view pre-generated analysis
open anomaly_analysis_report.html
```

## Troubleshooting

### Issue: "python3: command not found"

```bash
# Install Python 3.10+
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

### Issue: "git: command not found"

```bash
# Install Git
sudo apt update
sudo apt install git
```

### Issue: "DVC initialization failed"

```bash
# Remove and reinitialize
rm -rf .dvc
dvc init
dvc add data/raw/sutd
```

### Issue: "Dataset download failed"

```bash
# Manual download
cd data/raw
git clone https://github.com/FCCLab/sutd_5g_dataset_2023.git sutd_temp
cp sutd_temp/*.csv sutd/
rm -rf sutd_temp
```

### Issue: "Package conflicts"

```bash
# Create fresh environment
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Next Steps

After successful installation:

1. **Explore Data**: `jupyter notebook notebooks/01_eda.ipynb`
2. **Train Model**: `python src/watchtower/training/train_xgboost.py`
3. **View Experiments**: `mlflow ui`
4. **Read Documentation**: `cat docs/architecture.md`

## Directory Structure After Installation

```
DemoAnamolyDetection/
├── .venv/                  # Virtual environment
├── .dvc/                   # DVC cache
├── .git/                   # Git repository
├── data/
│   ├── raw/sutd/          # SUTD CSV files (DVC tracked)
│   ├── SUTD_VERSION.txt   # Dataset commit hash
│   └── sutd.dvc           # DVC pointer file
├── src/watchtower/        # Source code (empty, to be populated)
├── mlruns/                # MLflow tracking
├── setup_watchtower.sh    # Setup script
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── SETUP_SUMMARY.md       # Installation summary
```

## Support

For issues or questions:
1. Check `SETUP_SUMMARY.md` for configuration details
2. Review logs in the terminal output
3. Verify all prerequisites are installed
4. Try manual installation method if automated fails

---

**Installation Time**: ~5-10 minutes  
**Last Updated**: December 2024
