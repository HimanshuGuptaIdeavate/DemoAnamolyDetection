#!/usr/bin/env python3
"""
WATCHTOWER - Run Preprocessing Pipeline
Fits StandardScaler and OneHotEncoder, exports X, y matrices.

Usage:
    python run_preprocessing_pipeline.py
    python run_preprocessing_pipeline.py --config configs/preprocessing_config.yaml

Author: Himanshu's WATCHTOWER Project
Date: 2026-01-08
"""

import sys
import argparse
import logging
from pathlib import Path

# Configure logging BEFORE imports to affect all loggers
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path if running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from watchtower.features.preprocessing_pipeline import PreprocessingPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='WATCHTOWER Preprocessing Pipeline - Fit transformers and export data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config
    python run_preprocessing_pipeline.py
    
    # Run with custom config
    python run_preprocessing_pipeline.py --config custom_config.yaml
    
    # Run with verbose logging
    python run_preprocessing_pipeline.py --verbose
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/preprocessing_config.yaml',
        help='Path to preprocessing configuration file (default: configs/preprocessing_config.yaml)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print header
    print("\n" + "🎯 "*40)
    print("WATCHTOWER PREPROCESSING PIPELINE")
    print("Fit Transformers and Export Preprocessed Data")
    print("🎯 "*40 + "\n")
    
    try:
        # Initialize pipeline
        logger.info(f"Loading configuration from: {args.config}")
        pipeline = PreprocessingPipeline(config_path=args.config)
        
        # Run pipeline
        logger.info("Starting preprocessing pipeline...")
        X, y, meta_ts = pipeline.run()
        
        # Success message
        print("\n" + "="*80)
        print("✅ PREPROCESSING PIPELINE COMPLETE!")
        print("="*80)
        print("\nOutputs:")
        print(f"  📊 X.npy: {X.shape[0]:,} samples × {X.shape[1]} features")
        print(f"  🎯 y.npy: {y.shape[0]:,} labels")
        print(f"  🔧 scaler.joblib: StandardScaler")
        print(f"  🔧 onehot.joblib: OneHotEncoder")
        print(f"  📋 feature_order.json: {X.shape[1]} features")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Load X.npy and y.npy")
        print("2. Perform train-test split (GroupKFold)")
        print("3. Train XGBoost model")
        print("4. Evaluate with SHAP values")
        print("="*80 + "\n")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Make sure features_table.parquet exists in data/parquet/")
        return 1
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        logger.error("Check that all required columns exist in features_table.parquet")
        return 1
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
