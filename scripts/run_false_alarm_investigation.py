#!/usr/bin/env python3
"""
WATCHTOWER - Run False Alarm Investigation
Investigates high-probability normal samples causing high FPR.

Usage:
    python scripts/run_false_alarm_investigation.py

Author: Himanshu's WATCHTOWER Project
Date: 2026-02-06
"""

import sys
import logging
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from watchtower.analysis.investigate_false_alarms import FalseAlarmInvestigator


def main():
    """Main execution function."""
    print("\n" + "🔍 "*40)
    print("WATCHTOWER - FALSE ALARM INVESTIGATION")
    print("Analyzing high-probability normal samples")
    print("🔍 "*40 + "\n")

    try:
        # Check if required files exist
        required_files = [
            'data/parquet/X.npy',
            'data/parquet/y.npy',
            'data/parquet/features_table.parquet'
        ]

        missing_files = [f for f in required_files if not Path(f).exists()]

        if missing_files:
            logger.error("Missing required files:")
            for f in missing_files:
                logger.error(f"  - {f}")
            return 1

        # Run investigation
        investigator = FalseAlarmInvestigator()
        df = investigator.generate_report()

        # Save predictions
        output_path = 'reports/cv_predictions_with_features.parquet'
        Path('reports').mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

        print("\n" + "="*80)
        print("✅ INVESTIGATION COMPLETE!")
        print("="*80)
        print(f"\nOutputs:")
        print(f"  📊 Predictions: {output_path}")
        print(f"  📈 Feature plot: reports/plots/false_alarm_features.png")
        print("="*80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Investigation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
