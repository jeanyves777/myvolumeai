"""
Trend indicators (ADX, Stochastic, etc.)

Indicators for trend strength and momentum used in scalping strategies.
"""

from typing import Optional, List
from .base import Indicator
from ..core.models import Bar


class ADX(Indicator):
    """
    Average Directional Index (ADX).

    Measures trend strength regardless of direction.
    Used to determine if a market is trending or ranging.

    Components:
    - +DI (Positive Directional Indicator): Measures upward trend strength
    - -DI (Negative Directional Indicator): Measures downward trend strength
    - ADX: Smoothed average of the absolute difference between +DI and -DI

    Interpretation:
    - ADX > 25: Strong trend (good for trend-following)
    - ADX > 20: Trending market
    - ADX < 20: Weak/no trend (ranging market, good for scalping)
    - ADX < 15: Very weak trend

    Default period: 14
    """

    def __init__(self, period: int = 14):
        """
        Initialize ADX.

        Parameters
        ----------
        period : int
            Lookback period for ADX calculation (default 14)
        """
        super().__init__(period)
        self._prev_high: Optional[float] = None
        self._prev_low: Optional[float] = None
        self._prev_close: Optional[float] = None

        self._tr_values: List[float] = []
        self._plus_dm_values: List[float] = []
        self._minus_dm_values: List[float] = []

        self._smoothed_tr: float = 0.0
        self._smoothed_plus_dm: float = 0.0
        self._smoothed_minus_dm: float = 0.0

        self._plus_di: float = 0.0
        self._minus_di: float = 0.0
        self._dx_values: List[float] = []
        self._adx: float = 0.0

    @property
    def value(self) -> float:
        """Current ADX value"""
        return self._adx

    @property
    def plus_di(self) -> float:
        """Positive Directional Indicator (+DI)"""
        return self._plus_di

    @property
    def minus_di(self) -> float:
        """Negative Directional Indicator (-DI)"""
        return self._minus_di

    @property
    def trend_strength(self) -> str:
        """Get descriptive trend strength"""
        if self._adx >= 25:
            return "strong"
        elif self._adx >= 20:
            return "moderate"
        elif self._adx >= 15:
            return "weak"
        else:
            return "absent"

    def update(self, value: float) -> None:
        """
        Update ADX with close price only.
        Note: For proper ADX, use update_from_bar() with OHLCV data.

        This approximation uses close-to-close changes.
        """
        if self._prev_close is None:
            self._prev_close = value
            self._prev_high = value
            self._prev_low = value
            return

        # Approximate high/low from close
        change = abs(value - self._prev_close)
        approx_high = max(value, self._prev_close) + change * 0.1
        approx_low = min(value, self._prev_close) - change * 0.1

        self._update_adx(approx_high, approx_low, value)
        self._prev_close = value
        self._prev_high = approx_high
        self._prev_low = approx_low

    def update_from_bar(self, bar: Bar) -> None:
        """
        Update ADX from a Bar object (recommended method).

        Parameters
        ----------
        bar : Bar
            Bar object with OHLCV data
        """
        if self._prev_high is None:
            self._prev_high = bar.high
            self._prev_low = bar.low
            self._prev_close = bar.close
            return

        self._update_adx(bar.high, bar.low, bar.close)
        self._prev_high = bar.high
        self._prev_low = bar.low
        self._prev_close = bar.close

    def _update_adx(self, high: float, low: float, close: float) -> None:
        """Calculate ADX from OHLC data."""
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - self._prev_close)
        tr3 = abs(low - self._prev_close)
        true_range = max(tr1, tr2, tr3)

        # Calculate Directional Movement
        up_move = high - self._prev_high
        down_move = self._prev_low - low

        plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0

        self._tr_values.append(true_range)
        self._plus_dm_values.append(plus_dm)
        self._minus_dm_values.append(minus_dm)

        # Need at least period data points
        if len(self._tr_values) < self.period:
            return

        # First smoothing - simple sum
        if len(self._tr_values) == self.period:
            self._smoothed_tr = sum(self._tr_values)
            self._smoothed_plus_dm = sum(self._plus_dm_values)
            self._smoothed_minus_dm = sum(self._minus_dm_values)
        else:
            # Wilder's smoothing
            self._smoothed_tr = self._smoothed_tr - (self._smoothed_tr / self.period) + true_range
            self._smoothed_plus_dm = self._smoothed_plus_dm - (self._smoothed_plus_dm / self.period) + plus_dm
            self._smoothed_minus_dm = self._smoothed_minus_dm - (self._smoothed_minus_dm / self.period) + minus_dm

        # Calculate +DI and -DI
        if self._smoothed_tr > 0:
            self._plus_di = 100.0 * self._smoothed_plus_dm / self._smoothed_tr
            self._minus_di = 100.0 * self._smoothed_minus_dm / self._smoothed_tr
        else:
            self._plus_di = 0.0
            self._minus_di = 0.0

        # Calculate DX
        di_sum = self._plus_di + self._minus_di
        if di_sum > 0:
            dx = 100.0 * abs(self._plus_di - self._minus_di) / di_sum
        else:
            dx = 0.0

        self._dx_values.append(dx)

        # Calculate ADX (smoothed DX)
        if len(self._dx_values) >= self.period:
            if len(self._dx_values) == self.period:
                self._adx = sum(self._dx_values) / self.period
            else:
                # Wilder's smoothing for ADX
                self._adx = (self._adx * (self.period - 1) + dx) / self.period

            self._initialized = True

        self._values.append(self._adx)

        # Keep buffers manageable
        if len(self._tr_values) > self.period * 2:
            self._tr_values = self._tr_values[-self.period:]
            self._plus_dm_values = self._plus_dm_values[-self.period:]
            self._minus_dm_values = self._minus_dm_values[-self.period:]
            self._dx_values = self._dx_values[-self.period:]

    def is_trending(self, threshold: float = 20.0) -> bool:
        """
        Check if market is trending.

        Parameters
        ----------
        threshold : float
            ADX threshold for trending market (default 20)

        Returns
        -------
        bool
            True if ADX > threshold
        """
        return self._initialized and self._adx > threshold

    def is_ranging(self, threshold: float = 20.0) -> bool:
        """
        Check if market is ranging (not trending).

        Parameters
        ----------
        threshold : float
            ADX threshold for ranging market (default 20)

        Returns
        -------
        bool
            True if ADX < threshold
        """
        return self._initialized and self._adx < threshold

    def is_bullish(self) -> bool:
        """Check if trend is bullish (+DI > -DI)"""
        return self._initialized and self._plus_di > self._minus_di

    def is_bearish(self) -> bool:
        """Check if trend is bearish (-DI > +DI)"""
        return self._initialized and self._minus_di > self._plus_di

    def reset(self) -> None:
        super().reset()
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None
        self._tr_values.clear()
        self._plus_dm_values.clear()
        self._minus_dm_values.clear()
        self._smoothed_tr = 0.0
        self._smoothed_plus_dm = 0.0
        self._smoothed_minus_dm = 0.0
        self._plus_di = 0.0
        self._minus_di = 0.0
        self._dx_values.clear()
        self._adx = 0.0


class Stochastic(Indicator):
    """
    Stochastic Oscillator.

    Momentum indicator comparing closing price to the range over a period.
    Used to identify overbought/oversold conditions.

    Components:
    - %K (Fast Stochastic): (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    - %D (Slow Stochastic): SMA of %K

    Interpretation:
    - Above 80: Overbought
    - Below 20: Oversold
    - %K crosses above %D: Bullish signal
    - %K crosses below %D: Bearish signal

    Default: %K period = 14, %D period = 3
    """

    def __init__(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3):
        """
        Initialize Stochastic Oscillator.

        Parameters
        ----------
        k_period : int
            Lookback period for %K (default 14)
        d_period : int
            Smoothing period for %D (default 3)
        smooth_k : int
            Smoothing period for slow %K (default 3, use 1 for fast stochastic)
        """
        super().__init__(k_period)
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k

        self._highs: List[float] = []
        self._lows: List[float] = []
        self._closes: List[float] = []

        self._fast_k_values: List[float] = []
        self._k_values: List[float] = []
        self._d_values: List[float] = []

        self._k: float = 50.0  # %K
        self._d: float = 50.0  # %D
        self._prev_k: float = 50.0
        self._prev_d: float = 50.0

    @property
    def value(self) -> float:
        """Current %K value"""
        return self._k

    @property
    def k(self) -> float:
        """Slow %K value (smoothed)"""
        return self._k

    @property
    def d(self) -> float:
        """Slow %D value"""
        return self._d

    @property
    def fast_k(self) -> float:
        """Fast %K value (unsmoothed)"""
        return self._fast_k_values[-1] if self._fast_k_values else 50.0

    def update(self, value: float) -> None:
        """
        Update stochastic with close price only.
        Note: For proper calculation, use update_from_bar() with OHLCV data.

        This approximation uses close as high/low.
        """
        self._update_stochastic(value, value, value)

    def update_from_bar(self, bar: Bar) -> None:
        """
        Update Stochastic from a Bar object (recommended method).

        Parameters
        ----------
        bar : Bar
            Bar object with OHLCV data
        """
        self._update_stochastic(bar.high, bar.low, bar.close)

    def _update_stochastic(self, high: float, low: float, close: float) -> None:
        """Calculate Stochastic from HLC data."""
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

        # Keep only last k_period values
        if len(self._highs) > self.k_period:
            self._highs.pop(0)
            self._lows.pop(0)
            self._closes.pop(0)

        # Need at least k_period data points
        if len(self._closes) < self.k_period:
            return

        # Calculate Fast %K
        highest_high = max(self._highs)
        lowest_low = min(self._lows)
        hl_range = highest_high - lowest_low

        if hl_range > 0:
            fast_k = 100.0 * (close - lowest_low) / hl_range
        else:
            fast_k = 50.0

        self._fast_k_values.append(fast_k)

        # Calculate Slow %K (smoothed Fast %K)
        if len(self._fast_k_values) >= self.smooth_k:
            self._prev_k = self._k
            self._k = sum(self._fast_k_values[-self.smooth_k:]) / self.smooth_k
            self._k_values.append(self._k)

            # Calculate %D (smoothed %K)
            if len(self._k_values) >= self.d_period:
                self._prev_d = self._d
                self._d = sum(self._k_values[-self.d_period:]) / self.d_period
                self._d_values.append(self._d)
                self._initialized = True

        self._values.append(self._k)

        # Keep buffers manageable
        if len(self._fast_k_values) > self.k_period * 2:
            self._fast_k_values = self._fast_k_values[-self.k_period:]
        if len(self._k_values) > self.d_period * 2:
            self._k_values = self._k_values[-self.d_period:]
        if len(self._d_values) > self.d_period * 2:
            self._d_values = self._d_values[-self.d_period:]

    def is_overbought(self, threshold: float = 80.0) -> bool:
        """
        Check if stochastic indicates overbought.

        Parameters
        ----------
        threshold : float
            Overbought threshold (default 80)

        Returns
        -------
        bool
            True if %K > threshold
        """
        return self._initialized and self._k > threshold

    def is_oversold(self, threshold: float = 20.0) -> bool:
        """
        Check if stochastic indicates oversold.

        Parameters
        ----------
        threshold : float
            Oversold threshold (default 20)

        Returns
        -------
        bool
            True if %K < threshold
        """
        return self._initialized and self._k < threshold

    def is_bullish_cross(self) -> bool:
        """
        Check for bullish crossover (%K crosses above %D).

        Returns
        -------
        bool
            True if %K just crossed above %D
        """
        return (self._initialized and
                self._prev_k <= self._prev_d and
                self._k > self._d)

    def is_bearish_cross(self) -> bool:
        """
        Check for bearish crossover (%K crosses below %D).

        Returns
        -------
        bool
            True if %K just crossed below %D
        """
        return (self._initialized and
                self._prev_k >= self._prev_d and
                self._k < self._d)

    def reset(self) -> None:
        super().reset()
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._fast_k_values.clear()
        self._k_values.clear()
        self._d_values.clear()
        self._k = 50.0
        self._d = 50.0
        self._prev_k = 50.0
        self._prev_d = 50.0
