#!/usr/bin/env python3
"""
WATCHTOWER - Run XGBoost Training Pipeline
Trains M0 XGBoost model with GroupKFold cross-validation

Usage:
    python run_xgboost_training.py
    python run_xgboost_training.py --config configs/xgboost_config.yaml
    python run_xgboost_training.py --verbose

Author: Himanshu's WATCHTOWER Project
Date: 2026-01-12
"""

import sys
import argparse
import logging
import warnings
from pathlib import Path

# Suppress MLflow deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='mlflow')
warnings.filterwarnings('ignore', message='.*artifact_path.*is deprecated.*')

# Configure logging BEFORE imports to affect all loggers
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path if running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from watchtower.models.xgboost_training import XGBoostTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='WATCHTOWER XGBoost Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config
    python run_xgboost_training.py
    
    # Run with custom config
    python run_xgboost_training.py --config custom_xgboost.yaml
    
    # Run with verbose logging
    python run_xgboost_training.py --verbose
    
    # View MLflow UI after training
    mlflow ui --backend-store-uri file://mlruns
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/xgboost_config.yaml',
        help='Path to XGBoost configuration file (default: configs/xgboost_config.yaml)'
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
    print("WATCHTOWER - XGBOOST TRAINING PIPELINE")
    print("M0: XGBoost Anomaly Detection Model")
    print("🎯 "*40 + "\n")
    
    try:
        # Check if preprocessed data exists
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
            logger.error("\nPlease run preprocessing pipeline first:")
            logger.error("  python scripts/run_preprocessing_pipeline.py")
            return 1
        
        # Initialize trainer
        logger.info(f"Loading configuration from: {args.config}")
        trainer = XGBoostTrainer(config_path=args.config)
        
        # Run training pipeline
        logger.info("Starting XGBoost training pipeline...")
        model, cv_results = trainer.run()
        
        # Success message
        print("\n" + "="*80)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("="*80)
        
        print("\nOutputs:")
        print(f"  🎯 Model: models/xgboost_model_*.joblib")
        print(f"  📊 Plots: reports/plots/")
        print(f"  📈 MLflow: mlruns/")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. View training results:")
        print("   mlflow ui --backend-store-uri file://mlruns")
        print("   Then open: http://localhost:5000")
        print()
        print("2. Inspect model performance:")
        print("   - Check reports/plots/ for visualizations")
        print("   - Review confusion matrix and ROC curve")
        print()
        print("3. Deploy model:")
        print("   - Use saved model in models/ directory")
        print("   - Implement FastAPI inference endpoint")
        print("="*80 + "\n")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Make sure preprocessed data exists in data/parquet/")
        return 1
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
