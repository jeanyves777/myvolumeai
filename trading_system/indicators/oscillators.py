"""
Oscillator indicators (RSI, MACD, etc.)
"""

from typing import Optional, List
from .base import Indicator
from .moving_averages import ExponentialMovingAverage


class RelativeStrengthIndex(Indicator):
    """
    Relative Strength Index (RSI).

    Measures the speed and magnitude of recent price changes.
    Values range from 0 to 100.
    - Above 70: Overbought
    - Below 30: Oversold

    Formula: RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    """

    def __init__(self, period: int = 14):
        super().__init__(period)
        self._prev_value: Optional[float] = None
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._gains: List[float] = []
        self._losses: List[float] = []
        self._rsi: float = 50.0

    @property
    def value(self) -> float:
        return self._rsi

    def update(self, value: float) -> None:
        if self._prev_value is None:
            self._prev_value = value
            return

        # Calculate change
        change = value - self._prev_value
        gain = max(0, change)
        loss = max(0, -change)

        self._gains.append(gain)
        self._losses.append(loss)
        self._prev_value = value

        # Need at least 'period' data points
        if len(self._gains) < self.period:
            return

        # First calculation - simple average
        if len(self._gains) == self.period:
            self._avg_gain = sum(self._gains) / self.period
            self._avg_loss = sum(self._losses) / self.period
        else:
            # Smoothed average (Wilder's smoothing)
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        # Calculate RSI
        if self._avg_loss == 0:
            self._rsi = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            self._rsi = 100.0 - (100.0 / (1.0 + rs))

        self._initialized = True
        self._values.append(self._rsi)

        # Keep only last period values
        if len(self._gains) > self.period * 2:
            self._gains = self._gains[-self.period:]
            self._losses = self._losses[-self.period:]

    def reset(self) -> None:
        super().reset()
        self._prev_value = None
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._gains = []
        self._losses = []
        self._rsi = 50.0


class MACD(Indicator):
    """
    Moving Average Convergence Divergence (MACD).

    Trend-following momentum indicator showing relationship between
    two exponential moving averages.

    Components:
    - MACD Line: Fast EMA - Slow EMA
    - Signal Line: EMA of MACD Line
    - Histogram: MACD Line - Signal Line

    Default periods: 12, 26, 9
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ):
        super().__init__(slow_period)  # Use slow period as main period
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        self._fast_ema = ExponentialMovingAverage(fast_period)
        self._slow_ema = ExponentialMovingAverage(slow_period)
        self._signal_ema = ExponentialMovingAverage(signal_period)

        self._macd_line: float = 0.0
        self._signal_line: float = 0.0
        self._histogram: float = 0.0

    @property
    def value(self) -> float:
        """MACD line value"""
        return self._macd_line

    @property
    def signal(self) -> float:
        """Signal line value"""
        return self._signal_line

    @property
    def histogram(self) -> float:
        """Histogram value (MACD - Signal)"""
        return self._histogram

    def update(self, value: float) -> None:
        # Update component EMAs
        self._fast_ema.update(value)
        self._slow_ema.update(value)

        # Calculate MACD line once both EMAs are initialized
        if self._fast_ema.initialized and self._slow_ema.initialized:
            self._macd_line = self._fast_ema.value - self._slow_ema.value

            # Update signal line
            self._signal_ema.update(self._macd_line)

            if self._signal_ema.initialized:
                self._signal_line = self._signal_ema.value
                self._histogram = self._macd_line - self._signal_line
                self._initialized = True

        self._values.append(self._macd_line)

    def reset(self) -> None:
        super().reset()
        self._fast_ema.reset()
        self._slow_ema.reset()
        self._signal_ema.reset()
        self._macd_line = 0.0
        self._signal_line = 0.0
        self._histogram = 0.0
