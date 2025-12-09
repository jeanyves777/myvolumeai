"""
Data pipeline for ML training.

Handles data loading, preprocessing, and feature extraction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import logging
import time
from pathlib import Path

from ..features import FeatureEngineering

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Data pipeline for preparing training data.
    
    Handles:
    - Loading historical OHLCV data
    - Feature extraction
    - Label creation
    - Train/val/test splitting
    - Class imbalance handling
    """
    
    def __init__(self, lookahead: int = 5, profit_threshold: float = 0.5):
        """
        Initialize data pipeline.
        
        Args:
            lookahead: Bars to look ahead for labeling
            profit_threshold: Minimum profit % for positive label
        """
        self.lookahead = lookahead
        self.profit_threshold = profit_threshold
        self.feature_engineer = FeatureEngineering()
        
    def load_historical_data(self, file_path: str) -> pd.DataFrame:
        """
        Load historical OHLCV data.
        
        Args:
            file_path: Path to CSV file with OHLCV data
            
        Returns:
            DataFrame with historical data
        """
        logger.info(f"Loading data from {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def prepare_training_data(self, bars: pd.DataFrame,
                             symbols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with features and labels.
        
        Args:
            bars: Historical OHLCV data
            symbols: List of symbols (optional, for multi-symbol training)
            
        Returns:
            Tuple of (features, labels)
        """
        logger.info("Extracting features...")
        
        # Calculate technical indicators (simplified version)
        # In real implementation, use your actual indicator calculations
        indicators = self._calculate_indicators(bars)
        
        # Extract features
        features = self.feature_engineer.extract_features(bars, indicators)
        
        # Create labels
        logger.info("Creating labels...")
        labels = self.feature_engineer.create_labels(
            bars,
            lookahead=self.lookahead,
            profit_threshold=self.profit_threshold
        )
        
        # Remove rows with NaN labels
        valid_idx = ~labels.isna()
        features = features[valid_idx]
        labels = labels[valid_idx]
        
        logger.info(f"Prepared {len(features)} samples")
        logger.info(f"Positive labels: {labels.sum()} ({labels.mean()*100:.1f}%)")
        logger.info(f"Negative labels: {(~labels.astype(bool)).sum()} ({(1-labels.mean())*100:.1f}%)")
        
        return features, labels
    
    def _calculate_indicators(self, bars: pd.DataFrame) -> dict:
        """
        Calculate technical indicators for feature extraction.
        
        Args:
            bars: OHLCV data
            
        Returns:
            Dictionary of indicator values
        """
        indicators = {}
        
        # RSI
        delta = bars['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        indicators['rsi'] = {'value': 100 - (100 / (1 + rs))}
        
        # MACD
        exp1 = bars['close'].ewm(span=12, adjust=False).mean()
        exp2 = bars['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        indicators['macd'] = {
            'macd': macd,
            'signal': signal,
            'histogram': macd - signal
        }
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma = bars['close'].rolling(window=bb_period).mean()
        std = bars['close'].rolling(window=bb_period).std()
        indicators['bollinger'] = {
            'upper': sma + (std * bb_std),
            'middle': sma,
            'lower': sma - (std * bb_std)
        }
        
        # Stochastic
        low_14 = bars['low'].rolling(window=14).min()
        high_14 = bars['high'].rolling(window=14).max()
        k = 100 * ((bars['close'] - low_14) / (high_14 - low_14 + 1e-10))
        d = k.rolling(window=3).mean()
        indicators['stochastic'] = {'k': k, 'd': d}
        
        # ADX (simplified)
        indicators['adx'] = {'value': pd.Series(25, index=bars.index)}  # Placeholder
        
        # ATR
        tr1 = bars['high'] - bars['low']
        tr2 = abs(bars['high'] - bars['close'].shift())
        tr3 = abs(bars['low'] - bars['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        indicators['atr'] = {'value': atr}
        
        # Current close price
        indicators['close'] = bars['close']
        
        return indicators
    
    def split_data(self, features: pd.DataFrame, labels: pd.Series,
                   train_size: float = 0.7,
                   val_size: float = 0.15) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            train_size: Proportion for training (default 0.7)
            val_size: Proportion for validation (default 0.15)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n = len(features)
        
        # Calculate split indices (chronological order preserved)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        
        # Split data
        X_train = features.iloc[:train_end]
        X_val = features.iloc[train_end:val_end]
        X_test = features.iloc[val_end:]
        
        y_train = labels.iloc[:train_end]
        y_val = labels.iloc[train_end:val_end]
        y_test = labels.iloc[val_end:]
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def handle_class_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series,
                               method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using simple oversampling (no external deps).

        Args:
            X_train: Training features
            y_train: Training labels
            method: Method to use ('smote', 'random_oversample', 'none', 'simple')

        Returns:
            Tuple of (resampled features, resampled labels)
        """
        if method == 'none':
            return X_train, y_train

        # Use simple numpy-based oversampling to avoid slow imblearn import
        start_time = time.time()
        n_samples = len(X_train)

        print(f">>> [data_pipeline] +=========================================================+", flush=True)
        print(f">>> [data_pipeline] |  SIMPLE RANDOM OVERSAMPLING (no imblearn needed)        |", flush=True)
        print(f">>> [data_pipeline] +---------------------------------------------------------+", flush=True)
        print(f">>> [data_pipeline] |  Samples:    {n_samples:>10,}                              |", flush=True)
        print(f">>> [data_pipeline] |  Est. time:  <1 second                                    |", flush=True)
        print(f">>> [data_pipeline] +=========================================================+", flush=True)

        logger.info("Applying simple random oversampling for class balancing...")

        # Count classes
        y_array = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        unique, counts = np.unique(y_array, return_counts=True)

        if len(unique) < 2:
            logger.warning("Only one class found, skipping resampling")
            return X_train, y_train

        max_count = counts.max()

        # Find indices for each class
        X_resampled_list = []
        y_resampled_list = []

        for cls, count in zip(unique, counts):
            cls_mask = y_array == cls
            X_cls = X_train[cls_mask]
            y_cls = y_train[cls_mask]

            if count < max_count:
                # Oversample minority class
                n_oversample = max_count - count
                indices = np.random.choice(len(X_cls), size=n_oversample, replace=True)
                X_oversampled = X_cls.iloc[indices]
                y_oversampled = y_cls.iloc[indices]

                X_resampled_list.append(X_cls)
                X_resampled_list.append(X_oversampled)
                y_resampled_list.append(y_cls)
                y_resampled_list.append(y_oversampled)
            else:
                X_resampled_list.append(X_cls)
                y_resampled_list.append(y_cls)

        # Combine all
        X_resampled = pd.concat(X_resampled_list, ignore_index=True)
        y_resampled = pd.concat(y_resampled_list, ignore_index=True)

        # Shuffle
        shuffle_idx = np.random.permutation(len(X_resampled))
        X_resampled = X_resampled.iloc[shuffle_idx].reset_index(drop=True)
        y_resampled = y_resampled.iloc[shuffle_idx].reset_index(drop=True)

        elapsed = time.time() - start_time
        print(f">>> [data_pipeline] [OK] Resampling complete in {elapsed:.1f}s", flush=True)

        logger.info(f"Resampled from {len(X_train)} to {len(X_resampled)} samples")
        logger.info(f"New class distribution: {y_resampled.value_counts().to_dict()}")

        return X_resampled, y_resampled
    
    def save_processed_data(self, features: pd.DataFrame, labels: pd.Series,
                           output_path: str):
        """
        Save processed features and labels.
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            output_path: Path to save data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine features and labels
        data = features.copy()
        data['label'] = labels
        
        # Save to CSV
        data.to_csv(output_path)
        
        logger.info(f"✅ Saved processed data to {output_path}")
