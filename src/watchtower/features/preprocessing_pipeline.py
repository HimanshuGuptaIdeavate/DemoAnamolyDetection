"""
WATCHTOWER - Preprocessing Pipeline
Fits StandardScaler and OneHotEncoder, exports X, y matrices for XGBoost.

Author: Himanshu's WATCHTOWER Project
Date: 2026-01-08
"""

import pandas as pd
import numpy as np
import joblib
import json
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Preprocessing pipeline for WATCHTOWER.
    Fits scalers and encoders on complete dataset for XGBoost training.
    """
    
    def __init__(self, config_path: str = 'configs/preprocessing_config.yaml'):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config_path: Path to preprocessing configuration
        """
        self.config = self._load_config(config_path)
        self.df = None
        self.scaler = None
        self.encoder = None
        self.feature_order = None
        
        logger.info("Preprocessing pipeline initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        if Path(config_path).exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
        else:
            # Default configuration
            config = self._get_default_config()
            logger.warning(f"Config not found at {config_path}, using defaults")
        
        return config
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'input_path': 'data/parquet/features_table.parquet',
            'output_dir': 'data/parquet',
            'artifacts_dir': 'artifacts',
            'configs_dir': 'configs',
            'numeric_features': [
                # Signal quality (8)
                'rsrp_mean', 'rsrp_std', 'rsrq_mean', 'rsrq_std',
                'sinr_mean', 'sinr_std', 'sinr_min', 'sinr_max',
                # Resource allocation (4)
                'prb_dl_mean', 'prb_ul_mean', 'mcs_dl_mode', 'mcs_ul_mode',
                # Throughput (2)
                'app_dl_mean', 'app_dl_std',
                # Existing derived (8)
                'd_rsrp_mean', 'd_rsrq_mean', 'd_sinr_mean', 'd_sinr_std',
                'd_app_dl_mean', 'rsrp_rsrq_ratio', 'sinr_range', 'throughput_per_prb',
                # New derived (5)
                'prb_util_ratio', 'sinr_cv', 'throughput_efficiency',
                'hour_sin', 'hour_cos',
            ],
            'categorical_features': ['pci', 'scenario_id'],
            'target': 'weak_label',
            'metadata': 'ts_start_ns',
        }
    
    def load_data(self) -> pd.DataFrame:
        """Load features table."""
        input_path = self.config['input_path']
        logger.info(f"Loading features from: {input_path}")
        
        self.df = pd.read_parquet(input_path)
        
        logger.info(f"✅ Loaded {len(self.df):,} samples × {self.df.shape[1]} columns")
        
        # Validate required columns
        numeric_features = self.config['numeric_features']
        categorical_features = self.config['categorical_features']
        target = self.config['target']
        
        missing_numeric = [col for col in numeric_features if col not in self.df.columns]
        missing_categorical = [col for col in categorical_features if col not in self.df.columns]
        
        if missing_numeric:
            raise ValueError(f"Missing numeric columns: {missing_numeric}")
        if missing_categorical:
            raise ValueError(f"Missing categorical columns: {missing_categorical}")
        if target not in self.df.columns:
            raise ValueError(f"Missing target column: {target}")
        
        logger.info("✅ All required columns present")
        return self.df
    
    def fit_scaler(self) -> StandardScaler:
        """Fit StandardScaler on numeric features."""
        logger.info("="*80)
        logger.info("FITTING STANDARD SCALER")
        logger.info("="*80)
        
        numeric_features = self.config['numeric_features']
        X_num = self.df[numeric_features].values.astype('float32')
        
        logger.info(f"Shape: {X_num.shape} ({len(numeric_features)} features)")
        
        # Handle missing values
        n_missing = np.isnan(X_num).sum()
        if n_missing > 0:
            logger.warning(f"⚠️  {n_missing} missing values → filling with column means")
            for i in range(X_num.shape[1]):
                col_mean = np.nanmean(X_num[:, i])
                X_num[np.isnan(X_num[:, i]), i] = col_mean
        
        # Fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(X_num)
        
        logger.info("✅ Fitted StandardScaler")
        logger.info("\nSample scaling (first 5 features):")
        for i in range(min(5, len(numeric_features))):
            logger.info(f"  {numeric_features[i]:20s} → mean={self.scaler.mean_[i]:8.3f}, std={self.scaler.scale_[i]:8.3f}")
        
        return self.scaler
    
    def fit_encoder(self) -> OneHotEncoder:
        """Fit OneHotEncoder on categorical features."""
        logger.info("="*80)
        logger.info("FITTING ONE-HOT ENCODER")
        logger.info("="*80)
        
        categorical_features = self.config['categorical_features']
        X_cat = self.df[categorical_features].astype('string').values
        
        logger.info(f"Shape: {X_cat.shape} ({len(categorical_features)} features)")
        
        # Fit encoder
        self.encoder = OneHotEncoder(
            handle_unknown='ignore',  # Critical for production!
            sparse_output=False,
            drop=None,
            dtype='float32'
        )
        self.encoder.fit(X_cat)
        
        logger.info("✅ Fitted OneHotEncoder (handle_unknown='ignore')")
        logger.info("\nCategories:")
        for col, cats in zip(categorical_features, self.encoder.categories_):
            logger.info(f"  {col:15s} → {len(cats):2d} unique values")
        
        total_ohe = sum(len(cats) for cats in self.encoder.categories_)
        logger.info(f"\nTotal one-hot features: {total_ohe}")
        
        return self.encoder
    
    def transform_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform data using fitted scaler and encoder."""
        logger.info("="*80)
        logger.info("TRANSFORMING DATA")
        logger.info("="*80)
        
        numeric_features = self.config['numeric_features']
        categorical_features = self.config['categorical_features']
        target = self.config['target']
        metadata = self.config['metadata']
        
        # Scale numeric
        X_num = self.df[numeric_features].values.astype('float32')
        X_num_scaled = self.scaler.transform(X_num)
        logger.info(f"✅ Scaled numeric: {X_num_scaled.shape}")
        
        # Encode categorical
        X_cat = self.df[categorical_features].astype('string').values
        X_cat_enc = self.encoder.transform(X_cat)
        logger.info(f"✅ Encoded categorical: {X_cat_enc.shape}")
        
        # Concatenate
        X = np.hstack([X_num_scaled, X_cat_enc]).astype('float32')
        logger.info(f"✅ Combined X: {X.shape}")
        
        # Extract target
        y = self.df[target].astype('int8').values
        logger.info(f"✅ Target y: {y.shape}")
        logger.info(f"   Class 0/1: {np.bincount(y)}, Anomaly rate: {y.mean()*100:.1f}%")
        
        # Extract metadata
        meta_ts = self.df[metadata].values
        logger.info(f"✅ Metadata: {meta_ts.shape}")
        
        return X, y, meta_ts
    
    def generate_feature_order(self) -> List[str]:
        """Generate feature order for train-serve contract."""
        numeric_features = self.config['numeric_features']
        categorical_features = self.config['categorical_features']
        
        feature_order = numeric_features.copy()
        
        for col, cats in zip(categorical_features, self.encoder.categories_):
            for i in range(len(cats)):
                feature_order.append(f'ohe_{col}_{i}')
        
        self.feature_order = feature_order
        return feature_order
    
    def save_artifacts(self):
        """Save preprocessing artifacts."""
        logger.info("="*80)
        logger.info("SAVING ARTIFACTS")
        logger.info("="*80)
        
        artifacts_dir = self.config['artifacts_dir']
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
        
        # Scaler
        scaler_path = Path(artifacts_dir) / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"✅ {scaler_path}")
        
        # Encoder
        encoder_path = Path(artifacts_dir) / 'onehot.joblib'
        joblib.dump(self.encoder, encoder_path)
        logger.info(f"✅ {encoder_path}")
        
        # Feature order
        feature_order_path = Path(artifacts_dir) / 'feature_order.json'
        with open(feature_order_path, 'w') as f:
            json.dump(self.feature_order, f, indent=2)
        logger.info(f"✅ {feature_order_path} ({len(self.feature_order)} features)")
    
    def save_config(self):
        """Save feature configuration."""
        logger.info("="*80)
        logger.info("SAVING CONFIGURATION")
        logger.info("="*80)
        
        configs_dir = self.config['configs_dir']
        Path(configs_dir).mkdir(parents=True, exist_ok=True)
        
        cfg = {
            'numeric_cols': self.config['numeric_features'],
            'categorical_cols': self.config['categorical_features'],
            'target': self.config['target'],
            'window_seconds': 5,
            'n_numeric_features': len(self.config['numeric_features']),
            'n_categorical_features': len(self.config['categorical_features']),
            'total_features_after_encoding': len(self.feature_order),
        }
        
        config_path = Path(configs_dir) / 'feature_config.yaml'
        with open(config_path, 'w') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)
        logger.info(f"✅ {config_path}")
    
    def export_matrices(self, X: np.ndarray, y: np.ndarray, meta_ts: np.ndarray):
        """Export design matrices."""
        logger.info("="*80)
        logger.info("EXPORTING DESIGN MATRICES")
        logger.info("="*80)

        output_dir = self.config['output_dir']
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # X
        X_path = Path(output_dir) / 'X.npy'
        np.save(X_path, X)
        logger.info(f"✅ {X_path}")
        logger.info(f"   Shape: {X.shape}, dtype: {X.dtype}, size: {X.nbytes / 1e6:.2f} MB")

        # y
        y_path = Path(output_dir) / 'y.npy'
        np.save(y_path, y)
        logger.info(f"✅ {y_path}")
        logger.info(f"   Shape: {y.shape}, dtype: {y.dtype}, size: {y.nbytes / 1e3:.2f} KB")

        # metadata
        meta_path = Path(output_dir) / 'meta_ts.npy'
        np.save(meta_path, meta_ts)
        logger.info(f"✅ {meta_path}")
        logger.info(f"   Shape: {meta_ts.shape}, dtype: {meta_ts.dtype}, size: {meta_ts.nbytes / 1e3:.2f} KB")

        # Export CSV for inspection
        csv_path = Path(output_dir) / 'preprocessing_features.csv'
        df_export = pd.DataFrame(X, columns=self.feature_order)
        df_export['weak_label'] = y
        df_export['ts_start_ns'] = meta_ts
        df_export.to_csv(csv_path, index=False)
        logger.info(f"✅ {csv_path}")
        logger.info(f"   Shape: {df_export.shape}, size: {csv_path.stat().st_size / 1e6:.2f} MB")
    
    def run(self):
        """Run complete preprocessing pipeline."""
        logger.info("\n" + "="*80)
        logger.info("WATCHTOWER PREPROCESSING PIPELINE")
        logger.info("="*80)
        
        # Execute pipeline steps
        self.load_data()
        self.fit_scaler()
        self.fit_encoder()
        X, y, meta_ts = self.transform_data()
        self.generate_feature_order()
        self.save_artifacts()
        self.save_config()
        self.export_matrices(X, y, meta_ts)
        
        # Summary
        logger.info("="*80)
        logger.info("PREPROCESSING COMPLETE ✅")
        logger.info("="*80)
        logger.info(f"\nX: {X.shape[0]:,} samples × {X.shape[1]} features")
        logger.info(f"y: {y.shape[0]:,} samples")
        logger.info(f"\n✅ Ready for XGBoost training!")
        logger.info("="*80)
        
        return X, y, meta_ts


def main():
    """Main execution function."""
    pipeline = PreprocessingPipeline(config_path='configs/preprocessing_config.yaml')
    pipeline.run()


if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
