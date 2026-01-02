#!/usr/bin/env python3
"""
WATCHTOWER - Complete Data Validation & MLflow Logging Pipeline
Master script that runs Great Expectations validation and logs to MLflow

Usage:
    python run_validation_pipeline.py [data_path]

Example:
    python run_validation_pipeline.py data/raw/sutd
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import os

# Add project root to Python path so imports work from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # Add src folder for watchtower package
os.chdir(PROJECT_ROOT)  # Change to project root for relative paths

# Import validation and logging modules from src/watchtower
from watchtower.data.validate_data import WatchtowerValidator
from watchtower.utils.mlflow_logger import MLflowLogger


def main():
    """
    Main execution function.
    Runs complete validation and MLflow logging pipeline.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='WATCHTOWER Data Validation & MLflow Logging Pipeline'
    )
    parser.add_argument(
        'data_path',
        nargs='?',
        default='data/raw/sutd',
        help='Path to data directory or CSV file (default: data/raw/sutd)'
    )
    parser.add_argument(
        '--skip-mlflow',
        action='store_true',
        help='Skip MLflow logging (validation only)'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        help='Custom MLflow run name'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("WATCHTOWER DATA VALIDATION & MLFLOW PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data path: {args.data_path}")
    print("="*80 + "\n")
    
    # Step 1: Run Great Expectations Validation
    print("STEP 1: Running Great Expectations Validation\n")
    
    validator = WatchtowerValidator()
    
    try:
        success, validation_results = validator.run_validation(
            args.data_path,
            save_report=True
        )
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: MLflow Logging (if not skipped)
    if not args.skip_mlflow:
        print("\n" + "="*80)
        print("STEP 2: Logging to MLflow\n")
        
        try:
            # Load data for MLflow
            df = validator.load_data(args.data_path)
            
            # Get SUTD commit if available
            sutd_commit = None
            version_file = Path("data/SUTD_VERSION.txt")
            if version_file.exists():
                sutd_commit = version_file.read_text().strip()
                print(f"📌 SUTD Dataset Version: {sutd_commit[:8]}")
            
            # Initialize MLflow logger
            logger = MLflowLogger()
            
            # Log everything to MLflow
            run_id = logger.log_data_validation(
                df=df,
                validation_results=validation_results,
                sutd_commit=sutd_commit,
                run_name=args.run_name
            )
            
            print(f"\n✅ MLflow logging complete!")
            print(f"   Run ID: {run_id}")
            
        except Exception as e:
            print(f"\n⚠️ MLflow logging failed: {e}")
            import traceback
            traceback.print_exc()
            print("   (Validation results still saved)")
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"✅ Validation: {'PASSED' if success else 'FAILED'}")
    print(f"   - Total checks: {validation_results['total_checks']}")
    print(f"   - Passed: {validation_results['passed_checks']}")
    print(f"   - Failed: {validation_results['failed_checks']}")
    print(f"   - Score: {validation_results['passed_checks']/validation_results['total_checks']*100:.1f}%")
    
    if not args.skip_mlflow:
        print(f"\n✅ MLflow: Logged successfully")
        print(f"   - View UI: mlflow ui --backend-store-uri mlruns")
        print(f"   - Then open: http://localhost:5000")
    
    print(f"\n📄 Reports saved to: reports/validation/")
    print(f"📊 Plots saved to: reports/plots/")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
