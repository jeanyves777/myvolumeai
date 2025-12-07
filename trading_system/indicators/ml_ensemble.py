"""
ML Ensemble Indicator - Wrapper for using ensemble as a trading indicator.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path

try:
    from ..base import Indicator
except ImportError:
    # Fallback if base not found
    class Indicator:
        def __init__(self, name, outputs):
            self.name = name
            self.outputs = outputs

logger = logging.getLogger(__name__)


class MLEnsembleIndicator(Indicator):
    """
    ML Ensemble Indicator for trading strategies.
    
    Uses trained ML ensemble to generate trading signals based on
    pattern recognition across 60+ features.
    
    Output:
    - value: Probability (0-1) of positive price movement
    - signal: Text signal (STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL)
    - confidence: Confidence level (VERY_HIGH, HIGH, MODERATE, LOW, VERY_LOW)
    """
    
    def __init__(self, model_path: str, window: int = 50):
        """
        Initialize ML Ensemble Indicator.
        
        Args:
            model_path: Path to trained ensemble model
            window: Number of bars to keep for feature calculation
        """
        super().__init__("ml_ensemble", ["value", "signal", "confidence"])
        
        self.model_path = model_path
        self.window = window
        self.ensemble = None
        self.feature_engineer = None
        
        # Historical data for feature calculation
        self.bars_history = []
        self.indicators_history = []
        
        # Current values
        self.current_value = 0.5  # Neutral
        self.current_signal = "NEUTRAL"
        self.current_confidence = "LOW"
        
        # Load model if path exists
        self._load_model()
        
    def _load_model(self):
        """Load trained ensemble model."""
        import os

        # Try multiple paths to find the model
        possible_paths = [
            self.model_path,
            os.path.join(os.path.dirname(__file__), '..', '..', self.model_path),
            os.path.join(os.getcwd(), self.model_path),
            os.path.abspath(self.model_path),
        ]

        model_found = None
        for path in possible_paths:
            if Path(path).exists():
                model_found = path
                break

        if model_found is None:
            logger.warning(f"❌ Model not found. Tried paths: {possible_paths}")
            logger.warning("ML Ensemble Indicator will return neutral signals until model is trained")
            return

        try:
            from ..ml.ensemble import EnsemblePredictor
            from ..ml.features import FeatureEngineering

            self.ensemble = EnsemblePredictor()
            self.ensemble.load(model_found)

            self.feature_engineer = FeatureEngineering()

            logger.info(f"✅ ML Ensemble loaded from {model_found}")
            logger.info(f"   Models: {len(self.ensemble.models)} trained, is_trained={self.ensemble.is_trained}")

        except Exception as e:
            logger.error(f"❌ Failed to load ML ensemble: {str(e)}")
            import traceback
            traceback.print_exc()
            self.ensemble = None
    
    def update(self, bar: Dict[str, Any], indicators: Dict[str, Any]):
        """
        Update indicator with new bar and indicators.

        Args:
            bar: Current OHLCV bar
            indicators: Dictionary of other indicator values
        """
        # Store bar and indicators in history
        self.bars_history.append(bar)
        self.indicators_history.append(indicators.copy())

        # Keep only last 'window' bars
        if len(self.bars_history) > self.window:
            self.bars_history.pop(0)
            self.indicators_history.pop(0)

        # Need minimum data to calculate features
        if len(self.bars_history) < 30:
            self.current_value = 0.5
            self.current_signal = "NEUTRAL"
            self.current_confidence = "LOW"
            return

        if self.ensemble is None or not self.ensemble.is_trained:
            self.current_value = 0.5
            self.current_signal = "NEUTRAL"
            self.current_confidence = "LOW"
            return

        try:
            # Convert history to DataFrame
            bars_df = pd.DataFrame(self.bars_history)

            # Ensure timestamp column exists and set as index
            if 'timestamp' in bars_df.columns:
                bars_df['timestamp'] = pd.to_datetime(bars_df['timestamp'])
                bars_df = bars_df.set_index('timestamp')

            # Extract features
            features = self._extract_features_from_history(bars_df)

            if features is not None and not features.empty:
                # Get prediction for last bar only
                last_features = features.tail(1)

                # Handle any NaN/infinity values
                last_features = last_features.replace([np.inf, -np.inf], np.nan)
                last_features = last_features.fillna(0)
                last_features = last_features.clip(lower=-1e10, upper=1e10)

                prediction = self.ensemble.predict_proba(last_features)

                if len(prediction) > 0:
                    prob = prediction[0]

                    # Update current values
                    self.current_value = float(prob)
                    self.current_signal = self._get_signal(prob)
                    self.current_confidence = self._get_confidence(prob)

        except Exception as e:
            logger.warning(f"ML Ensemble prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            self.current_value = 0.5
            self.current_signal = "NEUTRAL"
            self.current_confidence = "LOW"
    
    def _extract_features_from_history(self, bars_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract features from historical bars and indicators."""
        try:
            # Get latest indicators
            latest_indicators = self.indicators_history[-1] if self.indicators_history else {}
            
            # Extract features
            features = self.feature_engineer.extract_features(bars_df, latest_indicators)
            
            return features
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {str(e)}")
            return None
    
    def _get_signal(self, probability: float) -> str:
        """Convert probability to trading signal."""
        if probability >= 0.70:
            return "STRONG_BUY"
        elif probability >= 0.60:
            return "BUY"
        elif probability >= 0.50:
            return "WEAK_BUY"
        elif probability >= 0.40:
            return "NEUTRAL"
        elif probability >= 0.30:
            return "WEAK_SELL"
        else:
            return "SELL"
    
    def _get_confidence(self, probability: float) -> str:
        """Convert probability to confidence level."""
        distance_from_neutral = abs(probability - 0.5)
        
        if distance_from_neutral >= 0.30:
            return "VERY_HIGH"
        elif distance_from_neutral >= 0.20:
            return "HIGH"
        elif distance_from_neutral >= 0.10:
            return "MODERATE"
        elif distance_from_neutral >= 0.05:
            return "LOW"
        else:
            return "VERY_LOW"
    
    @property
    def value(self) -> float:
        """Get current probability value."""
        return self.current_value
    
    @property
    def signal(self) -> str:
        """Get current trading signal."""
        return self.current_signal
    
    @property
    def confidence(self) -> str:
        """Get current confidence level."""
        return self.current_confidence
    
    @property
    def is_available(self) -> bool:
        """Check if ML ensemble is available and working."""
        return self.ensemble is not None and self.ensemble.is_trained
    
    def get_ml_score(self) -> float:
        """
        Get ML score for entry/exit decisions.
        
        Returns:
            Score from -2 to +2 based on signal strength
        """
        if not self.is_available:
            return 0.0
        
        prob = self.current_value
        
        if prob >= 0.70:
            return 2.0  # STRONG_BUY
        elif prob >= 0.65:
            return 1.5  # Strong BUY
        elif prob >= 0.60:
            return 1.0  # BUY
        elif prob >= 0.55:
            return 0.5  # Weak BUY
        elif prob >= 0.45:
            return 0.0  # NEUTRAL
        elif prob >= 0.40:
            return -0.5  # Weak SELL
        else:
            return -1.0  # SELL
    
    def should_enter(self, threshold: float = 0.60) -> bool:
        """
        Check if ML signal suggests entry.
        
        Args:
            threshold: Minimum probability for entry
            
        Returns:
            True if should enter
        """
        return self.is_available and self.current_value >= threshold
    
    def should_exit(self, threshold: float = 0.40) -> bool:
        """
        Check if ML signal suggests exit.
        
        Args:
            threshold: Maximum probability before exit
            
        Returns:
            True if should exit
        """
        return self.is_available and self.current_value < threshold
