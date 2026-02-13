"""
WATCHTOWER - LSTM Data Preparation
Extracts raw 2Hz sequences from clean_data.parquet for LSTM input.

Each 5-second window has 10 raw samples at 2Hz.
XGBoost sees aggregated stats (mean, std, etc.) - LSTM sees the raw time-series.

Input:  clean_data.parquet (8,726 raw 2Hz samples)
Output: X_lstm.npy shape (871, 10, 4) - one sequence per window
        Reuses existing y.npy (871,) labels

Author: Himanshu's WATCHTOWER Project
Date: 2026-02-11
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

# Window parameters (same as windowing_config.py)
WINDOW_SEC = 5
SAMPLE_RATE_HZ = 2
SAMPLES_PER_WINDOW = WINDOW_SEC * SAMPLE_RATE_HZ  # 10

# Raw features for LSTM input (4 key telemetry signals)
LSTM_RAW_FEATURES = [
    'sinr_db',       # Signal-to-Interference-Noise Ratio (PRIMARY anomaly indicator)
    'rsrp_dbm',      # Reference Signal Received Power
    'rsrq_db',       # Reference Signal Received Quality
    'app_dl_mbps',   # Downlink Throughput
]


class LSTMDataPrep:
    """
    Prepare raw 2Hz sequences for LSTM from clean_data.parquet.

    Creates 1:1 mapping with XGBoost windows:
    - Window #1: XGBoost sees 35 engineered features, LSTM sees (10, 4) raw sequence
    - Window #2: same pattern
    - ...
    - Window #871: same pattern

    This ensures parallel voting ensemble can combine predictions directly.
    """

    def __init__(
        self,
        clean_data_path: str = 'data/parquet/clean_data.parquet',
        output_dir: str = 'data/parquet',
        features: List[str] = None
    ):
        self.clean_data_path = clean_data_path
        self.output_dir = output_dir
        self.features = features or LSTM_RAW_FEATURES
        self.stats = {}

    def load_clean_data(self) -> pd.DataFrame:
        """Load raw 2Hz telemetry data."""
        logger.info("="*80)
        logger.info("LSTM DATA PREPARATION")
        logger.info("="*80)

        df = pd.read_parquet(self.clean_data_path)

        # Sort by scenario and timestamp (same order as windowing pipeline)
        df = df.sort_values(['scenario_id', 'ts_ns']).reset_index(drop=True)

        logger.info(f"Loaded clean_data: {len(df):,} rows x {len(df.columns)} columns")
        logger.info(f"Scenarios: {df['scenario_id'].nunique()}")
        logger.info(f"LSTM features: {self.features}")

        # Verify features exist
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features in clean_data: {missing}")

        self.stats['input_rows'] = len(df)
        self.stats['input_scenarios'] = df['scenario_id'].nunique()

        return df

    def extract_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract raw 2Hz sequences aligned with XGBoost windows.

        Uses the SAME windowing logic as windowing_pipeline.py:
        - Group by scenario_id
        - Take every 10 consecutive samples
        - No overlap

        Returns:
            X_lstm: shape (n_windows, 10, n_features) - raw sequences
            y: shape (n_windows,) - weak labels
            groups: shape (n_windows,) - scenario IDs for GroupKFold
        """
        logger.info("\nExtracting raw sequences...")

        all_sequences = []
        all_labels = []
        all_groups = []

        for scenario_id, scenario_df in df.groupby('scenario_id', sort=False):
            n_samples = len(scenario_df)
            n_windows = n_samples // SAMPLES_PER_WINDOW

            logger.info(f"  {scenario_id}: {n_samples} samples -> {n_windows} windows")

            for i in range(n_windows):
                start_idx = i * SAMPLES_PER_WINDOW
                end_idx = start_idx + SAMPLES_PER_WINDOW
                segment = scenario_df.iloc[start_idx:end_idx]

                # Extract raw feature values: shape (10, 4)
                sequence = segment[self.features].values.astype(np.float32)
                all_sequences.append(sequence)

                # Weak label: 1 if ANY sample has anomaly
                label = int(segment[['lab_anom', 'lab_inf']].any(axis=1).any())
                all_labels.append(label)

                # Scenario ID for GroupKFold
                all_groups.append(scenario_id)

        X_lstm = np.array(all_sequences)  # (n_windows, 10, 4)
        y = np.array(all_labels)           # (n_windows,)
        groups = np.array(all_groups)      # (n_windows,)

        logger.info(f"\nExtracted sequences:")
        logger.info(f"  X_lstm shape: {X_lstm.shape} (windows, timesteps, features)")
        logger.info(f"  y shape:      {y.shape}")
        logger.info(f"  groups shape: {groups.shape}")
        logger.info(f"  Features:     {self.features}")

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            logger.info(f"  Class {cls}: {count:,} ({count/len(y)*100:.1f}%)")

        self.stats['n_windows'] = len(y)
        self.stats['n_features'] = len(self.features)
        self.stats['n_timesteps'] = SAMPLES_PER_WINDOW

        return X_lstm, y, groups

    def validate_alignment(self, y_lstm: np.ndarray, y_xgboost_path: str = 'data/parquet/y.npy') -> bool:
        """
        Validate that LSTM labels match XGBoost labels (1:1 alignment).

        This is CRITICAL for parallel voting ensemble - both models must
        predict on the same windows in the same order.
        """
        logger.info("\nValidating alignment with XGBoost...")

        y_xgb = np.load(y_xgboost_path)

        if len(y_lstm) != len(y_xgb):
            logger.error(f"LENGTH MISMATCH: LSTM={len(y_lstm)}, XGBoost={len(y_xgb)}")
            return False

        match = np.array_equal(y_lstm, y_xgb)

        if match:
            logger.info(f"  ALIGNMENT VERIFIED: {len(y_lstm)} labels match perfectly")
        else:
            n_diff = np.sum(y_lstm != y_xgb)
            logger.warning(f"  MISMATCH: {n_diff}/{len(y_lstm)} labels differ!")
            logger.warning("  This may indicate different windowing order.")

        self.stats['alignment_verified'] = match
        return match

    def log_sequence_stats(self, X_lstm: np.ndarray):
        """Log statistics about the raw sequences."""
        logger.info("\nSequence Statistics (per feature):")
        logger.info("-"*60)

        for i, fname in enumerate(self.features):
            feature_data = X_lstm[:, :, i]
            logger.info(f"  {fname}:")
            logger.info(f"    Mean:  {feature_data.mean():.3f}")
            logger.info(f"    Std:   {feature_data.std():.3f}")
            logger.info(f"    Range: [{feature_data.min():.3f}, {feature_data.max():.3f}]")
            logger.info(f"    NaN:   {np.isnan(feature_data).sum()}")

    def save(self, X_lstm: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """Save LSTM data to numpy files."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        x_path = output_dir / 'X_lstm.npy'
        groups_path = output_dir / 'groups_lstm.npy'

        np.save(x_path, X_lstm)
        np.save(groups_path, groups)

        # y.npy already exists and is shared with XGBoost
        logger.info(f"\nSaved LSTM data:")
        logger.info(f"  X_lstm:  {x_path} -> {X_lstm.shape}")
        logger.info(f"  groups:  {groups_path} -> {groups.shape}")
        logger.info(f"  y:       data/parquet/y.npy (shared with XGBoost) -> {y.shape}")

        self.stats['x_path'] = str(x_path)
        self.stats['groups_path'] = str(groups_path)

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run complete LSTM data preparation pipeline.

        Returns:
            X_lstm: (n_windows, 10, 4) raw sequences
            y: (n_windows,) labels
            groups: (n_windows,) scenario IDs
        """
        # Load raw data
        df = self.load_clean_data()

        # Extract sequences
        X_lstm, y, groups = self.extract_sequences(df)

        # Validate alignment with XGBoost
        self.validate_alignment(y)

        # Log statistics
        self.log_sequence_stats(X_lstm)

        # Save
        self.save(X_lstm, y, groups)

        logger.info("\n" + "="*80)
        logger.info("LSTM DATA PREPARATION COMPLETE")
        logger.info("="*80)
        logger.info(f"  X_lstm: {X_lstm.shape} (windows={X_lstm.shape[0]}, timesteps={X_lstm.shape[1]}, features={X_lstm.shape[2]})")
        logger.info(f"  Features: {self.features}")
        logger.info(f"  Alignment with XGBoost: {'VERIFIED' if self.stats.get('alignment_verified') else 'FAILED'}")
        logger.info("="*80)

        return X_lstm, y, groups


def main():
    """Main execution."""
    prep = LSTMDataPrep()
    X_lstm, y, groups = prep.run()
    return X_lstm, y, groups


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
