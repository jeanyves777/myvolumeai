"""
Volume indicators (VWAP, VolumeMA, etc.)

Indicators for volume analysis used in scalping strategies.
"""

from typing import Optional, List
from .base import Indicator
from ..core.models import Bar


class VWAP(Indicator):
    """
    Volume Weighted Average Price (VWAP).

    Calculates the average price weighted by volume.
    Used to identify trend direction and potential support/resistance.

    Formula: VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    where Typical Price = (High + Low + Close) / 3

    For intraday trading, VWAP resets daily.
    For crypto (24/7), use rolling VWAP with a period.
    """

    def __init__(self, period: int = 0, reset_daily: bool = False):
        """
        Initialize VWAP.

        Parameters
        ----------
        period : int
            Rolling period for VWAP calculation.
            0 = cumulative (standard VWAP), >0 = rolling window
        reset_daily : bool
            If True, reset VWAP at the start of each day (for stocks).
            For crypto 24/7 trading, set to False and use rolling period.
        """
        super().__init__(period if period > 0 else 1)
        self._period = period
        self._reset_daily = reset_daily
        self._cumulative_tp_vol: float = 0.0
        self._cumulative_vol: float = 0.0
        self._vwap: float = 0.0
        self._tp_vol_values: List[float] = []  # For rolling calculation
        self._vol_values: List[float] = []     # For rolling calculation
        self._last_date: Optional[str] = None

    @property
    def value(self) -> float:
        """Current VWAP value"""
        return self._vwap

    def update(self, value: float) -> None:
        """
        Update VWAP with close price only.
        Note: For proper VWAP, use update_from_bar() with OHLCV data.

        When only close is available, typical price = close.
        """
        self._update_vwap(value, value, value, 1.0)

    def update_from_bar(self, bar: Bar) -> None:
        """
        Update VWAP from a Bar object (recommended method).

        Parameters
        ----------
        bar : Bar
            Bar object with OHLCV data
        """
        # Check for daily reset (for stock trading)
        if self._reset_daily and self._last_date is not None:
            bar_date = bar.timestamp.strftime("%Y-%m-%d") if hasattr(bar.timestamp, 'strftime') else str(bar.timestamp)[:10]
            if bar_date != self._last_date:
                self._reset_cumulative()
            self._last_date = bar_date

        self._update_vwap(bar.high, bar.low, bar.close, bar.volume)

    def _update_vwap(self, high: float, low: float, close: float, volume: float) -> None:
        """Update VWAP calculation with OHLCV data."""
        # Calculate typical price
        typical_price = (high + low + close) / 3.0
        tp_vol = typical_price * volume

        if self._period > 0:
            # Rolling VWAP
            self._tp_vol_values.append(tp_vol)
            self._vol_values.append(volume)

            # Keep only last 'period' values
            if len(self._tp_vol_values) > self._period:
                self._tp_vol_values.pop(0)
                self._vol_values.pop(0)

            # Calculate rolling VWAP
            total_tp_vol = sum(self._tp_vol_values)
            total_vol = sum(self._vol_values)

            if total_vol > 0:
                self._vwap = total_tp_vol / total_vol
                if len(self._tp_vol_values) >= min(self._period, 5):
                    self._initialized = True
        else:
            # Cumulative VWAP
            self._cumulative_tp_vol += tp_vol
            self._cumulative_vol += volume

            if self._cumulative_vol > 0:
                self._vwap = self._cumulative_tp_vol / self._cumulative_vol
                self._initialized = True

        self._values.append(self._vwap)

    def _reset_cumulative(self) -> None:
        """Reset cumulative values for daily VWAP."""
        self._cumulative_tp_vol = 0.0
        self._cumulative_vol = 0.0
        self._tp_vol_values.clear()
        self._vol_values.clear()

    def is_price_below(self, price: float) -> bool:
        """Check if price is below VWAP (potential buy signal)."""
        return self._initialized and price < self._vwap

    def is_price_above(self, price: float) -> bool:
        """Check if price is above VWAP (potential sell signal)."""
        return self._initialized and price > self._vwap

    def get_distance_pct(self, price: float) -> float:
        """Get percentage distance from VWAP."""
        if not self._initialized or self._vwap == 0:
            return 0.0
        return ((price - self._vwap) / self._vwap) * 100.0

    def reset(self) -> None:
        super().reset()
        self._cumulative_tp_vol = 0.0
        self._cumulative_vol = 0.0
        self._vwap = 0.0
        self._tp_vol_values.clear()
        self._vol_values.clear()
        self._last_date = None


class VolumeMA(Indicator):
    """
    Volume Moving Average.

    Calculates a moving average of volume to detect volume spikes.
    A spike is detected when current volume exceeds the average by a multiplier.

    Used for:
    - Confirming breakouts
    - Detecting institutional activity
    - Entry/exit confirmation in scalping
    """

    def __init__(self, period: int = 20):
        """
        Initialize Volume MA.

        Parameters
        ----------
        period : int
            Lookback period for volume average (default 20)
        """
        super().__init__(period)
        self._volumes: List[float] = []
        self._volume_ma: float = 0.0
        self._current_volume: float = 0.0

    @property
    def value(self) -> float:
        """Current volume moving average"""
        return self._volume_ma

    @property
    def current_volume(self) -> float:
        """Most recent volume value"""
        return self._current_volume

    @property
    def volume_ratio(self) -> float:
        """Ratio of current volume to average (useful for spike detection)"""
        if self._volume_ma == 0:
            return 1.0
        return self._current_volume / self._volume_ma

    def update(self, value: float) -> None:
        """
        Update with new volume value.

        Parameters
        ----------
        value : float
            Current volume
        """
        self._current_volume = value
        self._volumes.append(value)

        # Keep only last 'period' values
        if len(self._volumes) > self.period:
            self._volumes.pop(0)

        # Calculate MA once we have enough data
        if len(self._volumes) >= self.period:
            self._volume_ma = sum(self._volumes) / len(self._volumes)
            self._initialized = True
        elif len(self._volumes) > 0:
            # Use available data before full initialization
            self._volume_ma = sum(self._volumes) / len(self._volumes)

        self._values.append(self._volume_ma)

    def update_from_bar(self, bar: Bar) -> None:
        """
        Update from a Bar object.

        Parameters
        ----------
        bar : Bar
            Bar object with OHLCV data
        """
        self.update(bar.volume)

    def is_spike(self, multiplier: float = 1.5) -> bool:
        """
        Check if current volume is a spike above average.

        Parameters
        ----------
        multiplier : float
            Volume spike threshold (default 1.5 = 50% above average)

        Returns
        -------
        bool
            True if current volume > (average * multiplier)
        """
        if not self._initialized:
            return False
        return self._current_volume > (self._volume_ma * multiplier)

    def is_low_volume(self, multiplier: float = 0.5) -> bool:
        """
        Check if current volume is below average.

        Parameters
        ----------
        multiplier : float
            Low volume threshold (default 0.5 = 50% below average)

        Returns
        -------
        bool
            True if current volume < (average * multiplier)
        """
        if not self._initialized:
            return False
        return self._current_volume < (self._volume_ma * multiplier)

    def get_spike_strength(self) -> float:
        """
        Get how many standard deviations current volume is from average.

        Returns
        -------
        float
            Z-score of current volume (positive = above average)
        """
        if not self._initialized or len(self._volumes) < 2:
            return 0.0

        import math
        mean = self._volume_ma
        variance = sum((v - mean) ** 2 for v in self._volumes) / len(self._volumes)
        std = math.sqrt(variance) if variance > 0 else 1.0

        return (self._current_volume - mean) / std if std > 0 else 0.0

    def reset(self) -> None:
        super().reset()
        self._volumes.clear()
        self._volume_ma = 0.0
        self._current_volume = 0.0
