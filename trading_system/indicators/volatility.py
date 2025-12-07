"""
Volatility indicators (Bollinger Bands, ATR, etc.)
"""

from typing import Optional, List
from .base import Indicator
from .moving_averages import SimpleMovingAverage, ExponentialMovingAverage
from ..core.models import Bar
import math


class BollingerBands(Indicator):
    """
    Bollinger Bands.

    Volatility bands placed above and below a moving average.
    Bands widen when volatility increases and narrow when it decreases.

    Components:
    - Middle Band: SMA of close prices
    - Upper Band: Middle Band + (k * Standard Deviation)
    - Lower Band: Middle Band - (k * Standard Deviation)

    Default: 20-period SMA with 2 standard deviations
    """

    def __init__(self, period: int = 20, k: float = 2.0):
        super().__init__(period)
        self.k = k  # Number of standard deviations
        self._prices: List[float] = []
        self._middle: float = 0.0
        self._upper: float = 0.0
        self._lower: float = 0.0
        self._std: float = 0.0

    @property
    def value(self) -> float:
        """Middle band value"""
        return self._middle

    @property
    def middle(self) -> float:
        """Middle band (SMA)"""
        return self._middle

    @property
    def upper(self) -> float:
        """Upper band"""
        return self._upper

    @property
    def lower(self) -> float:
        """Lower band"""
        return self._lower

    @property
    def bandwidth(self) -> float:
        """Bandwidth = (Upper - Lower) / Middle"""
        if self._middle == 0:
            return 0.0
        return (self._upper - self._lower) / self._middle

    @property
    def percent_b(self) -> float:
        """
        %B = (Price - Lower) / (Upper - Lower)
        Shows where price is relative to the bands.
        """
        band_width = self._upper - self._lower
        if band_width == 0:
            return 0.5
        return (self._prices[-1] - self._lower) / band_width if self._prices else 0.5

    def update(self, value: float) -> None:
        self._prices.append(value)

        # Keep only last 'period' prices
        if len(self._prices) > self.period:
            self._prices.pop(0)

        # Need at least period data points
        if len(self._prices) < self.period:
            return

        # Calculate middle band (SMA)
        self._middle = sum(self._prices) / len(self._prices)

        # Calculate standard deviation
        variance = sum((p - self._middle) ** 2 for p in self._prices) / len(self._prices)
        self._std = math.sqrt(variance)

        # Calculate bands
        self._upper = self._middle + (self.k * self._std)
        self._lower = self._middle - (self.k * self._std)

        self._initialized = True
        self._values.append(self._middle)

    def reset(self) -> None:
        super().reset()
        self._prices = []
        self._middle = 0.0
        self._upper = 0.0
        self._lower = 0.0
        self._std = 0.0


class AverageTrueRange(Indicator):
    """
    Average True Range (ATR).

    Measures market volatility by analyzing the range of price movement.
    Higher ATR = Higher volatility.

    True Range = max of:
    - Current High - Current Low
    - abs(Current High - Previous Close)
    - abs(Current Low - Previous Close)

    ATR = Smoothed average of True Range (typically 14 periods)
    """

    def __init__(self, period: int = 14):
        super().__init__(period)
        self._prev_close: Optional[float] = None
        self._tr_values: List[float] = []
        self._atr: float = 0.0

    @property
    def value(self) -> float:
        return self._atr

    def update(self, value: float) -> None:
        """
        For ATR, we need high, low, and close.
        This method accepts close price for compatibility.
        Use update_from_bar() for proper ATR calculation.
        """
        # If only close is provided, approximate TR as a percentage of close
        if self._prev_close is not None:
            tr = abs(value - self._prev_close)
            self._update_atr(tr)
        self._prev_close = value

    def update_from_bar(self, bar: Bar) -> None:
        """
        Update ATR from a Bar object (recommended method).

        Parameters
        ----------
        bar : Bar
            Bar object with OHLCV data
        """
        if self._prev_close is None:
            self._prev_close = bar.close
            return

        # Calculate True Range
        tr1 = bar.high - bar.low
        tr2 = abs(bar.high - self._prev_close)
        tr3 = abs(bar.low - self._prev_close)
        true_range = max(tr1, tr2, tr3)

        self._update_atr(true_range)
        self._prev_close = bar.close

    def _update_atr(self, true_range: float) -> None:
        """Update ATR with new True Range value"""
        self._tr_values.append(true_range)

        if len(self._tr_values) < self.period:
            return

        # First calculation - simple average
        if len(self._tr_values) == self.period:
            self._atr = sum(self._tr_values) / self.period
        else:
            # Smoothed average (Wilder's smoothing)
            self._atr = (self._atr * (self.period - 1) + true_range) / self.period

        self._initialized = True
        self._values.append(self._atr)

        # Keep only last 2*period values
        if len(self._tr_values) > self.period * 2:
            self._tr_values = self._tr_values[-self.period:]

    def reset(self) -> None:
        super().reset()
        self._prev_close = None
        self._tr_values = []
        self._atr = 0.0
