#!/usr/bin/env python3
"""
WATCHTOWER - Data Curation and Cleaning (ETL) Master Script
Run complete ETL pipeline: Extract → Transform → Validate → Load

Usage:
    python run_etl_pipeline.py [data_path]

Examples:
    python run_etl_pipeline.py data/raw/sutd
    python run_etl_pipeline.py data/raw/sutd --skip-validation
    python run_etl_pipeline.py data/raw/sutd --skip-mlflow
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import os

# Add project root to Python path so imports work from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # Add src folder for watchtower package
os.chdir(PROJECT_ROOT)  # Change to project root for relative paths

# Import ETL modules from src/watchtower
from watchtower.data.etl_pipeline import ETLPipeline
from watchtower.data.etl_config import print_config_summary


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='WATCHTOWER ETL Pipeline - Data Curation and Cleaning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/raw/sutd
  %(prog)s data/raw/sutd --skip-validation
  %(prog)s data/raw/sutd --skip-mlflow --output custom.parquet
        """
    )
    
    parser.add_argument(
        'data_path',
        nargs='?',
        default='data/raw/sutd',
        help='Path to raw CSV files (default: data/raw/sutd)'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip Great Expectations validation'
    )
    
    parser.add_argument(
        '--skip-mlflow',
        action='store_true',
        help='Skip MLflow logging'
    )
    
    parser.add_argument(
        '--skip-export',
        action='store_true',
        help='Skip Parquet export (for testing)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Custom output filename (default: clean_data.parquet)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Print header
    print("\n" + "="*80)
    print("WATCHTOWER ETL PIPELINE")
    print("Data Curation and Cleaning")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data path: {args.data_path}")
    print("="*80)
    
    # Print configuration summary
    if not args.quiet:
        print_config_summary()
    
    # Check if data path exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"\n❌ Error: Data path does not exist: {args.data_path}")
        return 1
    
    # Initialize ETL pipeline
    etl = ETLPipeline(verbose=not args.quiet)
    
    try:
        # Run full pipeline
        clean_df, stats = etl.run_full_pipeline(
            data_path=str(data_path),
            validate=not args.skip_validation,
            export_parquet=not args.skip_export,
            log_mlflow=not args.skip_mlflow
        )
        
        # Print final summary
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        print(f"✅ Status: SUCCESS")
        print(f"✅ Input rows: {stats['input_rows']:,}")
        print(f"✅ Output rows: {stats['output_rows']:,}")
        print(f"✅ Rows dropped: {stats['input_rows'] - stats['output_rows']:,}")
        print(f"✅ Values clipped: {stats.get('values_clipped', 0):,}")
        print(f"✅ Duration: {stats['duration_seconds']:.2f} seconds")
        
        if not args.skip_export:
            print(f"\n📄 Output file: {stats['output_file']}")
            print(f"📊 File size: {stats['output_size_mb']:.2f} MB")
        
        if not args.skip_validation:
            validation = stats.get('validation', {})
            if validation.get('success'):
                print(f"\n✅ Validation: PASSED ({validation.get('passed_checks', 0)}/{validation.get('total_checks', 0)} checks)")
            elif validation.get('success') is False:
                print(f"\n⚠️  Validation: {validation.get('failed_checks', 0)} checks failed")
        
        if not args.skip_mlflow:
            print(f"\n📊 MLflow: Logged to experiment 'watchtower_etl'")
            print(f"   View UI: mlflow ui")
        
        print("\n" + "="*80)
        print("Next steps:")
        print("  1. Review cleaned data: data/parquet/clean_data.parquet")
        print("  2. Check ETL report: reports/etl_report.json")
        print("  3. Proceed to windowing: python run_windowing.py")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: ETL pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
