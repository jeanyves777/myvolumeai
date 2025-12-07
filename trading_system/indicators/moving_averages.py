"""
Moving average indicators.
"""

from typing import Optional, List
from .base import Indicator


class SimpleMovingAverage(Indicator):
    """
    Simple Moving Average (SMA).

    Calculates the arithmetic mean of the last N values.

    Formula: SMA = (P1 + P2 + ... + Pn) / n
    """

    def __init__(self, period: int = 20):
        super().__init__(period)
        self._sum = 0.0

    @property
    def value(self) -> float:
        if not self._initialized or len(self._values) == 0:
            return 0.0
        return self._sum / len(self._values)

    def update(self, value: float) -> None:
        self._values.append(value)
        self._sum += value

        # Remove oldest value if we have more than period
        if len(self._values) > self.period:
            oldest = self._values.pop(0)
            self._sum -= oldest

        # Mark as initialized once we have enough data
        if len(self._values) >= self.period:
            self._initialized = True

    def reset(self) -> None:
        super().reset()
        self._sum = 0.0


class ExponentialMovingAverage(Indicator):
    """
    Exponential Moving Average (EMA).

    Gives more weight to recent prices using exponential smoothing.

    Formula: EMA = Price * k + EMA(prev) * (1 - k)
    where k = 2 / (period + 1)
    """

    def __init__(self, period: int = 20):
        super().__init__(period)
        self._ema: float = 0.0
        self._multiplier: float = 2.0 / (period + 1)
        self._count: int = 0

    @property
    def value(self) -> float:
        return self._ema

    def update(self, value: float) -> None:
        self._count += 1

        if self._count == 1:
            # First value - use as initial EMA
            self._ema = value
        elif self._count <= self.period:
            # Build up initial SMA
            self._ema = (self._ema * (self._count - 1) + value) / self._count
        else:
            # Standard EMA calculation
            self._ema = (value - self._ema) * self._multiplier + self._ema

        # Mark as initialized once we have enough data
        if self._count >= self.period:
            self._initialized = True

        self._values.append(self._ema)

    def reset(self) -> None:
        super().reset()
        self._ema = 0.0
        self._count = 0
