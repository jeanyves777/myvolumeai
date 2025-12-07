"""
Base indicator class that all indicators inherit from.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from ..core.models import Bar


class Indicator(ABC):
    """
    Abstract base class for all technical indicators.

    All indicators must implement:
    - update(): Add new data point
    - value: Current indicator value
    - initialized: Whether indicator has enough data
    - reset(): Clear all data
    """

    def __init__(self, period: int = 14):
        """
        Initialize indicator.

        Parameters
        ----------
        period : int
            Lookback period for the indicator
        """
        self.period = period
        self._values: List[float] = []
        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Check if indicator has enough data to produce valid values"""
        return self._initialized

    @property
    @abstractmethod
    def value(self) -> float:
        """Get current indicator value"""
        pass

    @abstractmethod
    def update(self, value: float) -> None:
        """
        Update indicator with new value.

        Parameters
        ----------
        value : float
            New data point (typically close price)
        """
        pass

    def update_from_bar(self, bar: Bar) -> None:
        """
        Update indicator from a Bar object.
        Uses close price by default.

        Parameters
        ----------
        bar : Bar
            Bar object with OHLCV data
        """
        self.update(bar.close)

    def reset(self) -> None:
        """Reset indicator to initial state"""
        self._values.clear()
        self._initialized = False

    @property
    def count(self) -> int:
        """Number of values received"""
        return len(self._values)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(period={self.period}, value={self.value:.4f})"
