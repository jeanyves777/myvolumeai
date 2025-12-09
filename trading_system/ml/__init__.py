"""
ML Ensemble Trading System

Provides machine learning models and ensemble prediction
for pattern recognition and trade signal enhancement.

NOTE: Imports are lazy to avoid slow sklearn/xgboost loading at import time.
"""

# Lazy imports to avoid slow startup
def get_base_classes():
    """Get BaseMLModel and MLModelConfig (lazy load)."""
    from .base import BaseMLModel, MLModelConfig
    return BaseMLModel, MLModelConfig

def get_ensemble_predictor():
    """Get EnsemblePredictor class (lazy load to avoid sklearn/xgboost import)."""
    from .ensemble import EnsemblePredictor
    return EnsemblePredictor

def get_feature_engineering():
    """Get FeatureEngineering class (lazy load)."""
    from .features import FeatureEngineering
    return FeatureEngineering

__all__ = [
    'get_base_classes',
    'get_ensemble_predictor',
    'get_feature_engineering',
]
