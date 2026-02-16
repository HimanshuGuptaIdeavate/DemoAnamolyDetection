#!/usr/bin/env python3
"""
WATCHTOWER - Run LSTM Training Pipeline
Trains M1 LSTM model with GroupKFold cross-validation.

Usage:
    python scripts/run_lstm_training.py
    python scripts/run_lstm_training.py --config configs/lstm_config.yaml

Author: Himanshu's WATCHTOWER Project
Date: 2026-02-11
"""

import sys
import argparse
import logging
import warnings
from pathlib import Path

# Use non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='mlflow')
warnings.filterwarnings('ignore', message='.*artifact_path.*is deprecated.*')

# Suppress TensorFlow info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from watchtower.models.lstm_training import LSTMTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='WATCHTOWER LSTM Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_lstm_training.py
    python scripts/run_lstm_training.py --config configs/lstm_config.yaml
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/lstm_config.yaml',
        help='Path to LSTM configuration file'
    )

    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_args()

    print("\n" + "="*80)
    print("WATCHTOWER - LSTM TRAINING PIPELINE (M1)")
    print("Parallel Voting Ensemble - LSTM Component")
    print("="*80 + "\n")

    try:
        # Check required files
        required_files = [
            'data/parquet/X_lstm.npy',
            'data/parquet/y.npy',
            'data/parquet/groups_lstm.npy',
        ]

        missing = [f for f in required_files if not Path(f).exists()]
        if missing:
            logger.error("Missing required files:")
            for f in missing:
                logger.error(f"  - {f}")
            logger.error("\nRun LSTM data prep first:")
            logger.error("  python scripts/run_lstm_data_prep.py")
            return 1

        # Initialize trainer
        trainer = LSTMTrainer(config_path=args.config)

        # Run training pipeline
        model, cv_results = trainer.run()

        # Success
        print("\n" + "="*80)
        print("LSTM TRAINING PIPELINE COMPLETE!")
        print("="*80)
        print("\nOutputs:")
        print(f"  Model:        models/lstm_model_*.keras")
        print(f"  Scaler:       models/lstm_scaler_*.joblib")
        print(f"  Plots:        reports/plots/lstm_*.png")
        print(f"  CV Preds:     reports/lstm_cv_y_proba.npy (for ensemble)")
        print(f"  MLflow:       mlruns/")
        print("\nNext step:")
        print("  python scripts/run_ensemble_training.py")
        print("="*80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"LSTM training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
