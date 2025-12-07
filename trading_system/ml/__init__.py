"""
ML Ensemble Trading System

Provides machine learning models and ensemble prediction
for pattern recognition and trade signal enhancement.
"""

from .base import BaseMLModel, MLModelConfig
from .ensemble import EnsemblePredictor
from .features import FeatureEngineering

__all__ = [
    'BaseMLModel',
    'MLModelConfig',
    'EnsemblePredictor',
    'FeatureEngineering',
]
