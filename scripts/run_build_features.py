#!/usr/bin/env python3
"""
WATCHTOWER - Build Feature Table Master Script
Build features_table.parquet from windows.parquet

This script ONLY builds the feature table - NO splitting, scaling, or encoding!

Usage:
    python run_build_features.py [input_path]

Examples:
    python run_build_features.py
    python run_build_features.py data/parquet/windows.parquet
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

# Import feature builder
from watchtower.features.build_features import FeatureTableBuilder
from watchtower.features.build_features_config import print_config_summary


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='WATCHTOWER Feature Table Builder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s data/parquet/windows.parquet
  %(prog)s --quiet
        """
    )
    
    parser.add_argument(
        'input_path',
        nargs='?',
        default='data/parquet/windows.parquet',
        help='Path to windows parquet file (default: data/parquet/windows.parquet)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Custom output path (default: data/parquet/features_table.parquet)'
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
    print("WATCHTOWER FEATURE TABLE BUILDER")
    print("Build features_table.parquet from windows.parquet")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input path: {args.input_path}")
    if args.output:
        print(f"Output path: {args.output}")
    print("="*80)
    
    # Print configuration
    if not args.quiet:
        print_config_summary()
    
    # Check if input exists
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"\n❌ Error: Input file does not exist: {args.input_path}")
        print(f"\nPlease run windowing pipeline first:")
        print(f"  python scripts/run_windowing_pipeline.py")
        return 1
    
    # Initialize builder
    builder = FeatureTableBuilder(verbose=not args.quiet)
    
    try:
        # Build feature table
        df_feat = builder.build_feature_table(
            input_path=str(input_path),
            output_path=args.output
        )
        
        # Print final summary
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        print(f"✅ Status: SUCCESS")
        print(f"✅ Input windows: {builder.stats['input_windows']:,}")
        print(f"✅ Output rows: {len(df_feat):,}")
        print(f"✅ Output columns: {len(df_feat.columns)}")
        print(f"✅ Features created: {builder.stats['output_features']}")
        print(f"✅ New derived features: {builder.stats['derived_features_created']}")
        print(f"✅ Duration: {builder.stats['duration_seconds']:.2f} seconds")
        
        print(f"\n📄 Output file:")
        print(f"   {builder.stats['output_file']}")
        print(f"   Size: {builder.stats['output_size_mb']:.2f} MB")
        print(f"   Shape: {df_feat.shape}")
        
        print(f"\n📊 Statistics saved to:")
        print(f"   reports/feature_table_stats.json")
        
        print("\n" + "="*80)
        print("Next steps:")
        print("  1. Review feature table: data/parquet/features_table.parquet")
        print("  2. Check statistics: reports/feature_table_stats.json")
        print("  3. Proceed to scaling & encoding (next instruction)")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: Feature table building failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
