"""
Feature engineering for ML models.

Extracts 60+ features from price data and technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class FeatureEngineering:
    """
    Feature engineering for ML ensemble models.
    
    Extracts features from:
    - Price action (returns, momentum, acceleration)
    - Technical indicators (RSI, MACD, BB, Stochastic, etc.)
    - Candlestick patterns
    - Volume analysis
    - Time-based features
    - Volatility metrics
    """
    
    def __init__(self):
        """Initialize feature engineering."""
        self.feature_names = []
        
    def extract_features(self, bars: pd.DataFrame, 
                        indicators: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract all features from bars and indicators.
        
        Args:
            bars: OHLCV data
            indicators: Dictionary of indicator values
            
        Returns:
            DataFrame with all features
        """
        features = pd.DataFrame(index=bars.index)
        
        # 1. Price-based features
        features = self._add_price_features(features, bars)
        
        # 2. Technical indicator features
        features = self._add_indicator_features(features, indicators)
        
        # 3. Candlestick pattern features
        features = self._add_pattern_features(features, bars)
        
        # 4. Volume features
        features = self._add_volume_features(features, bars)
        
        # 5. Time-based features
        features = self._add_time_features(features, bars)
        
        # 6. Volatility features
        features = self._add_volatility_features(features, bars)
        
        # Store feature names
        self.feature_names = features.columns.tolist()

        # Fill NaN values and replace infinity
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().fillna(0)

        # Clip extreme values to prevent overflow
        features = features.clip(lower=-1e10, upper=1e10)

        return features
    
    def _add_price_features(self, features: pd.DataFrame, 
                           bars: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features (returns, momentum, etc.)."""
        
        # Returns for different periods
        for period in [1, 2, 3, 5, 10]:
            features[f'returns_{period}'] = bars['close'].pct_change(period)
        
        # Log returns
        for period in [1, 2, 3]:
            features[f'log_returns_{period}'] = np.log(bars['close'] / bars['close'].shift(period))
        
        # Momentum (rate of change)
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = (
                (bars['close'] - bars['close'].shift(period)) / bars['close'].shift(period) * 100
            )
        
        # Acceleration (change in momentum)
        features['acceleration_5'] = features['momentum_5'].diff()
        
        return features
    
    def _add_indicator_features(self, features: pd.DataFrame,
                                indicators: Dict[str, Any]) -> pd.DataFrame:
        """Add technical indicator features."""
        
        # RSI
        if 'rsi' in indicators:
            features['rsi_value'] = indicators['rsi'].get('value', 50)
            features['rsi_slope'] = pd.Series(features['rsi_value']).diff()
            features['rsi_oversold'] = (features['rsi_value'] < 30).astype(int)
            features['rsi_overbought'] = (features['rsi_value'] > 70).astype(int)
        
        # MACD
        if 'macd' in indicators:
            features['macd_value'] = indicators['macd'].get('macd', 0)
            features['macd_signal'] = indicators['macd'].get('signal', 0)
            features['macd_hist'] = indicators['macd'].get('histogram', 0)
            features['macd_bullish'] = (features['macd_hist'] > 0).astype(int)
        
        # Bollinger Bands
        if 'bollinger' in indicators:
            bb = indicators['bollinger']
            if 'upper' in bb and 'lower' in bb and 'middle' in bb:
                bb_upper = bb['upper']
                bb_lower = bb['lower']
                bb_middle = bb['middle']
                
                # BB position (0 = lower band, 0.5 = middle, 1 = upper band)
                features['bb_position'] = (
                    (indicators.get('close', bb_middle) - bb_lower) / 
                    (bb_upper - bb_lower + 1e-10)
                )
                
                # BB width (normalized volatility)
                features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Stochastic
        if 'stochastic' in indicators:
            features['stoch_k'] = indicators['stochastic'].get('k', 50)
            features['stoch_d'] = indicators['stochastic'].get('d', 50)
            features['stoch_oversold'] = (features['stoch_k'] < 20).astype(int)
            features['stoch_overbought'] = (features['stoch_k'] > 80).astype(int)
        
        # ADX (trend strength)
        if 'adx' in indicators:
            features['adx_value'] = indicators['adx'].get('value', 25)
            features['adx_slope'] = pd.Series(features['adx_value']).diff()
            features['adx_strong_trend'] = (features['adx_value'] > 25).astype(int)
        
        # ATR (volatility)
        if 'atr' in indicators:
            atr = indicators['atr'].get('value', 0)
            close = indicators.get('close', 1)
            # Handle both Series and scalar values
            if isinstance(close, pd.Series):
                features['atr_normalized'] = atr / (close + 1e-10)
            else:
                features['atr_normalized'] = atr / close if close > 0 else 0
        
        return features
    
    def _add_pattern_features(self, features: pd.DataFrame,
                             bars: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features."""
        
        # Candle body size
        features['body_size'] = abs(bars['close'] - bars['open']) / (bars['high'] - bars['low'] + 1e-10)
        
        # Wicks
        features['upper_wick'] = (
            (bars['high'] - bars[['open', 'close']].max(axis=1)) / 
            (bars['high'] - bars['low'] + 1e-10)
        )
        features['lower_wick'] = (
            (bars[['open', 'close']].min(axis=1) - bars['low']) / 
            (bars['high'] - bars['low'] + 1e-10)
        )
        
        # Candle direction
        features['bullish_candle'] = (bars['close'] > bars['open']).astype(int)
        
        # Pattern detection (simplified)
        features['is_hammer'] = (
            (features['lower_wick'] > 0.6) & 
            (features['upper_wick'] < 0.2) &
            (features['body_size'] < 0.3)
        ).astype(int)
        
        features['is_doji'] = (features['body_size'] < 0.1).astype(int)
        
        # Count recent bullish/bearish candles
        features['bullish_count_3'] = features['bullish_candle'].rolling(3).sum()
        features['bearish_count_3'] = (1 - features['bullish_candle']).rolling(3).sum()
        
        return features
    
    def _add_volume_features(self, features: pd.DataFrame,
                            bars: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        
        # Volume ratio (current vs average)
        volume_ma_20 = bars['volume'].rolling(20).mean()
        features['volume_ratio'] = bars['volume'] / (volume_ma_20 + 1e-10)
        
        # Volume spike
        features['volume_spike'] = (features['volume_ratio'] > 1.5).astype(int)
        
        # Volume-price correlation
        features['volume_price_corr_5'] = (
            bars['volume'].rolling(5).corr(bars['close'])
        )
        
        # On-Balance Volume slope
        obv = (np.sign(bars['close'].diff()) * bars['volume']).cumsum()
        features['obv_slope'] = obv.diff()
        
        return features
    
    def _add_time_features(self, features: pd.DataFrame,
                          bars: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        
        if isinstance(bars.index, pd.DatetimeIndex):
            features['hour'] = bars.index.hour
            features['day_of_week'] = bars.index.dayofweek
            
            # Trading sessions (UTC times)
            features['asian_session'] = (
                (features['hour'] >= 0) & (features['hour'] < 8)
            ).astype(int)
            features['european_session'] = (
                (features['hour'] >= 8) & (features['hour'] < 16)
            ).astype(int)
            features['us_session'] = (
                (features['hour'] >= 14) & (features['hour'] < 22)
            ).astype(int)
        
        return features
    
    def _add_volatility_features(self, features: pd.DataFrame,
                                 bars: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        
        # Realized volatility
        features['realized_vol_10'] = bars['close'].pct_change().rolling(10).std()
        
        # Volatility ratio
        vol_ma_20 = features['realized_vol_10'].rolling(20).mean()
        features['volatility_ratio'] = features['realized_vol_10'] / (vol_ma_20 + 1e-10)
        
        # High-Low range
        features['hl_range'] = (bars['high'] - bars['low']) / bars['close']
        
        # Close position in range (0 = at low, 1 = at high)
        features['close_position'] = (
            (bars['close'] - bars['low']) / (bars['high'] - bars['low'] + 1e-10)
        )
        
        return features
    
    def create_labels(self, bars: pd.DataFrame, 
                     lookahead: int = 5,
                     profit_threshold: float = 0.5) -> pd.Series:
        """
        Create labels for supervised learning.
        
        Args:
            bars: OHLCV data
            lookahead: Bars to look ahead
            profit_threshold: Minimum profit % to label as BUY (1)
            
        Returns:
            Series of labels (1 = BUY, 0 = NO_BUY)
        """
        # Look ahead at maximum high
        future_max = bars['high'].rolling(window=lookahead).max().shift(-lookahead)
        
        # Calculate future return
        future_return = ((future_max - bars['close']) / bars['close']) * 100
        
        # Label: 1 if profit >= threshold, 0 otherwise
        labels = (future_return >= profit_threshold).astype(int)
        
        # Remove last lookahead bars (no future data)
        labels.iloc[-lookahead:] = np.nan
        
        return labels
