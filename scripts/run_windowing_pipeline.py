#!/usr/bin/env python3
"""
WATCHTOWER - Windowing and Feature Engineering Master Script
Create 5-second time windows with derived features from clean telemetry data

Usage:
    python run_windowing_pipeline.py [input_path]

Examples:
    python run_windowing_pipeline.py
    python run_windowing_pipeline.py data/parquet/clean_data.parquet
    python run_windowing_pipeline.py --skip-mlflow
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Setup project root and Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

# Import windowing modules
from watchtower.data.windowing_pipeline import WindowingPipeline
from watchtower.data.windowing_config import print_config_summary


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='WATCHTOWER Windowing Pipeline - 5-Second Window Aggregation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s data/parquet/clean_data.parquet
  %(prog)s --skip-visualizations
  %(prog)s --skip-mlflow
        """
    )
    
    parser.add_argument(
        'input_path',
        nargs='?',
        default='data/parquet/clean_data.parquet',
        help='Path to clean data parquet file (default: data/parquet/clean_data.parquet)'
    )
    
    parser.add_argument(
        '--skip-export',
        action='store_true',
        help='Skip parquet export (for testing)'
    )
    
    parser.add_argument(
        '--skip-visualizations',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--skip-mlflow',
        action='store_true',
        help='Skip MLflow logging'
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
    print("WATCHTOWER WINDOWING PIPELINE")
    print("5-Second Window Aggregation + Derived Features")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input path: {args.input_path}")
    print("="*80)
    
    # Print configuration
    if not args.quiet:
        print_config_summary()
    
    # Check if input exists
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"\n❌ Error: Input file does not exist: {args.input_path}")
        print(f"\nPlease run ETL pipeline first:")
        print(f"  python scripts/run_etl_pipeline.py data/raw/sutd")
        return 1
    
    # Initialize pipeline
    pipeline = WindowingPipeline(verbose=not args.quiet)
    
    try:
        # Run full pipeline
        windows_df, stats = pipeline.run_full_pipeline(
            input_path=str(input_path),
            export=not args.skip_export,
            visualize=not args.skip_visualizations,
            log_mlflow=not args.skip_mlflow
        )
        
        # Print final summary
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        print(f"✅ Status: SUCCESS")
        print(f"✅ Input samples: {stats['input_rows']:,}")
        print(f"✅ Output windows: {stats['output_windows']:,}")
        print(f"✅ Window size: {stats.get('window_sec', 5)} seconds")
        print(f"✅ Samples per window: {stats.get('samples_per_window', 10)}")
        print(f"✅ Gap ratio (avg): {stats.get('gap_ratio_mean', 0):.2%}")
        print(f"✅ Anomaly rate: {stats.get('weak_label_rate', 0):.1%}")
        print(f"✅ Duration: {stats['duration_seconds']:.2f} seconds")
        
        if not args.skip_export:
            print(f"\n📄 Output file: {stats['output_file']}")
            print(f"📊 File size: {stats['output_size_mb']:.2f} MB")
            print(f"📊 Columns: {stats['output_columns']}")
        
        if not args.skip_visualizations:
            print(f"\n📈 Visualizations:")
            print(f"   - SINR time series: reports/plots/windowing_preview.png")
            print(f"   - Feature distributions: reports/plots/windowing_distribution.png")
        
        if not args.skip_mlflow:
            print(f"\n📊 MLflow: Logged to experiment 'watchtower_windowing'")
            print(f"   View UI: mlflow ui")
        
        print("\n" + "="*80)
        print("Next steps:")
        print("  1. Review windowed data: data/parquet/windows.parquet")
        print("  2. Check metrics: reports/windowing_metrics.json")
        print("  3. View plots: reports/plots/windowing_preview.png")
        print("  4. Proceed to model training!")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: Windowing pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
