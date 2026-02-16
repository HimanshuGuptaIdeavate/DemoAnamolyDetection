#!/usr/bin/env python3
"""
WATCHTOWER - Run Ensemble Training Pipeline
Combines XGBoost (M0) + LSTM (M1) via Parallel Voting.

Prerequisites:
    1. python scripts/run_xgboost_training.py  (saves xgb_cv_y_proba.npy)
    2. python scripts/run_lstm_training.py     (saves lstm_cv_y_proba.npy)

Usage:
    python scripts/run_ensemble_training.py

Author: Himanshu's WATCHTOWER Project
Date: 2026-02-13
"""

import sys
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='mlflow')

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from watchtower.models.ensemble_voting import EnsembleVoting


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("WATCHTOWER - ENSEMBLE PARALLEL VOTING")
    print("XGBoost (M0) + LSTM (M1) Combination")
    print("="*80 + "\n")

    try:
        # Check required files
        required_files = [
            'reports/xgb_cv_y_true.npy',
            'reports/xgb_cv_y_proba.npy',
            'reports/lstm_cv_y_true.npy',
            'reports/lstm_cv_y_proba.npy',
        ]

        missing = [f for f in required_files if not Path(f).exists()]
        if missing:
            logger.error("Missing required files:")
            for f in missing:
                logger.error(f"  - {f}")
            logger.error("\nRun both training pipelines first:")
            logger.error("  1. python scripts/run_xgboost_training.py")
            logger.error("  2. python scripts/run_lstm_training.py")
            return 1

        # Run ensemble
        ensemble = EnsembleVoting()
        weight_result, comparison_df = ensemble.run()

        print("\n" + "="*80)
        print("ENSEMBLE PIPELINE COMPLETE!")
        print("="*80)
        print(f"\nOutputs:")
        print(f"  Config:   configs/ensemble_optimal_*.json")
        print(f"  Plots:    reports/plots/ensemble_*.png")
        print(f"  MLflow:   mlruns/")
        print("="*80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Ensemble pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
