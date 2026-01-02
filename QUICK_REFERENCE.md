# ⚡ WATCHTOWER Quick Reference Card

## 🎯 One-Command Setup

```bash
cd /home/himanshu/BlinklyRelated/WatchTowerRND/DemoAnamolyDetection
bash setup_watchtower.sh
```

## 📦 What Gets Installed

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Runtime |
| XGBoost | 2.0+ | Primary model (M0) |
| PyTorch | 2.3+ | LSTM model (M1) |
| MLflow | 2.9+ | Experiment tracking |
| DVC | 3.0+ | Data versioning |
| SHAP | 0.44+ | Model explainability |
| FastAPI | 0.109+ | Production serving |

## 📊 Dataset

- **Name**: SUTD 5G Dataset 2023
- **Source**: [GitHub](https://github.com/FCCLab/sutd_5g_dataset_2023)
- **Size**: 8,732 samples (42.8% anomalies)
- **Files**: 4 CSV files (~50 MB total)
- **Location**: `data/raw/sutd/`

## 🚀 Common Commands

### Environment
```bash
# Activate
source .venv/bin/activate

# Deactivate
deactivate

# Check packages
pip list
```

### Dataset
```bash
# List files
ls -lh data/raw/sutd/

# Check DVC
dvc status

# View summary
cat SETUP_SUMMARY.md
```

### MLflow
```bash
# Start UI
mlflow ui

# Open: http://localhost:5000
```

### Jupyter
```bash
# Start notebook
jupyter notebook

# Or specific notebook
jupyter notebook notebooks/01_eda.ipynb
```

## 🔍 Verification

```bash
# All-in-one check
source .venv/bin/activate && \
python -c "import xgboost, torch, mlflow; print('✅ OK')" && \
ls data/raw/sutd/*.csv && \
dvc status && \
echo "✅ Setup verified!"
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `setup_watchtower.sh` | Automated setup script |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `INSTALL.md` | Installation guide |
| `EXECUTION_GUIDE.md` | Step-by-step instructions |
| `data/raw/sutd.dvc` | DVC tracking file |
| `SETUP_SUMMARY.md` | Setup report |

## 🐛 Quick Fixes

### Issue: Command not found
```bash
source .venv/bin/activate
```

### Issue: Package missing
```bash
pip install -r requirements.txt
```

### Issue: Dataset not found
```bash
ls data/raw/sutd/  # Should show 4 CSV files
```

### Issue: DVC error
```bash
dvc status
dvc pull  # If remote configured
```

## 📈 Next Steps

1. ✅ Run `setup_watchtower.sh`
2. ✅ Verify with `cat SETUP_SUMMARY.md`
3. ⏭️ **Next instruction**: Data ingestion pipeline
4. ⏭️ Feature engineering
5. ⏭️ XGBoost training
6. ⏭️ LSTM training (if needed)
7. ⏭️ Production deployment

## 💡 Pro Tips

- **Always activate** `.venv` before working
- **Check DVC status** regularly: `dvc status`
- **Use MLflow UI** to track experiments: `mlflow ui`
- **Read logs** if setup fails - they're detailed
- **Keep SUTD_VERSION.txt** - it tracks dataset version

## 🎯 Success Criteria

After setup, you should have:
- [x] Virtual environment (`.venv/`)
- [x] 4 CSV files in `data/raw/sutd/`
- [x] DVC initialized (`.dvc/` exists)
- [x] Git repository (`.git/` exists)
- [x] Project structure (`src/watchtower/` exists)
- [x] Setup summary (`SETUP_SUMMARY.md` generated)

## 📞 When Things Go Wrong

1. Read terminal output carefully
2. Check `INSTALL.md` for manual steps
3. Verify Python/Git installed
4. Try manual setup section in INSTALL.md
5. Check internet connection for downloads

---

**Setup Time**: ~5-10 minutes  
**Ready Status**: ✅ Ready to execute  
**Next**: Run the script and give me the next instruction!
