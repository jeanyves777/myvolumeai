"""
Data pipeline for ML training.

Handles data loading, preprocessing, and feature extraction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import logging
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
        Handle class imbalance using SMOTE or other methods.
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: Method to use ('smote', 'random_oversample', 'none')
            
        Returns:
            Tuple of (resampled features, resampled labels)
        """
        if method == 'none':
            return X_train, y_train
        
        try:
            from imblearn.over_sampling import SMOTE, RandomOverSampler
            
            if method == 'smote':
                resampler = SMOTE(random_state=42)
                logger.info("Applying SMOTE for class balancing...")
            elif method == 'random_oversample':
                resampler = RandomOverSampler(random_state=42)
                logger.info("Applying random oversampling for class balancing...")
            else:
                logger.warning(f"Unknown method {method}, skipping resampling")
                return X_train, y_train
            
            X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
            
            logger.info(f"Resampled from {len(X_train)} to {len(X_resampled)} samples")
            logger.info(f"New class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
            
            # Convert back to DataFrame/Series
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
            y_resampled = pd.Series(y_resampled, name=y_train.name)
            
            return X_resampled, y_resampled
            
        except ImportError:
            logger.warning("imbalanced-learn not installed. Skipping class balancing.")
            logger.warning("Install with: pip install imbalanced-learn")
            return X_train, y_train
    
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
