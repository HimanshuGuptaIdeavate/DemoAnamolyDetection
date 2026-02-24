#!/usr/bin/env python3
"""
WATCHTOWER - Run LSTM Data Preparation
Extracts raw 2Hz sequences from clean_data.parquet for LSTM input.

Usage:
    python scripts/run_lstm_data_prep.py

Output:
    data/parquet/X_lstm.npy  - shape (871, 10, 16)
        8 raw features + 8 intra-window delta features
    data/parquet/groups_lstm.npy - scenario IDs

Author: Himanshu's WATCHTOWER Project
Date: 2026-02-11
"""

import sys
import logging
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from watchtower.models.lstm_data_prep import LSTMDataPrep


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("WATCHTOWER - LSTM DATA PREPARATION")
    print("Extract raw 2Hz sequences for LSTM model")
    print("="*80 + "\n")

    # Check required files
    required_files = [
        'data/parquet/clean_data.parquet',
        'data/parquet/y.npy',
    ]

    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        logger.error("Missing required files:")
        for f in missing:
            logger.error(f"  - {f}")
        return 1

    try:
        prep = LSTMDataPrep()
        X_lstm, y, groups = prep.run()

        print("\n" + "="*80)
        print("LSTM DATA PREPARATION COMPLETE")
        print("="*80)
        print(f"\nOutputs:")
        print(f"  X_lstm:  data/parquet/X_lstm.npy  -> {X_lstm.shape}")
        print(f"  y:       data/parquet/y.npy       -> {y.shape} (shared with XGBoost)")
        print(f"  groups:  data/parquet/groups_lstm.npy -> {groups.shape}")
        print(f"\nNext step:")
        print(f"  python scripts/run_lstm_training.py")
        print("="*80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"LSTM data preparation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
