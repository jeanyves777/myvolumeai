"""
Technical indicators library.

All indicators follow a consistent interface:
- update(value) or update(bar) to add new data
- value property to get current indicator value
- initialized property to check if enough data exists
- reset() to clear all data
"""

from .base import Indicator
from .moving_averages import ExponentialMovingAverage, SimpleMovingAverage
from .oscillators import RelativeStrengthIndex, MACD
from .volatility import BollingerBands, AverageTrueRange
from .volume import VWAP, VolumeMA
from .trend import ADX, Stochastic

__all__ = [
    'Indicator',
    'ExponentialMovingAverage',
    'SimpleMovingAverage',
    'RelativeStrengthIndex',
    'MACD',
    'BollingerBands',
    'AverageTrueRange',
    'VWAP',
    'VolumeMA',
    'ADX',
    'Stochastic',
]
