"""
Crypto Scalping Strategy V10 - 24/7 Spot Trading on Alpaca

VERSION 10 ENHANCEMENTS:
========================
- NEW: 1H MACRO CONTEXT (Bias Layer, not hard filter)
  - 50/200 EMA trend direction
  - Key S/R level detection
  - ADX regime detection (trending vs ranging)
  - Provides score bonus/penalty, NOT blocking trades

V9 INDICATORS:
- M0 (15-min): RSI-14 and ADX-14 for trend strength
- M1 (1-min): Williams %R and CCI for oversold confirmation
- M2 (5-min): ATR volatility filter and VWAP distance check

SUPPORTED ASSETS (Alpaca Spot):
- BTC/USD, ETH/USD, SOL/USD, DOGE/USD, AVAX/USD

SIGNAL HIERARCHY (V10):
=======================
MACRO (1H Context Layer) - Provides bias/boost, NOT a blocker:
  - 50/200 EMA: Trend direction
  - ADX: Trending vs Ranging regime
  - S/R: Key levels for context

M0 (15-min Master Trend) - Main trend filter:
  - EMA20 slope + Price vs EMA
  - Candle patterns + HH/HL analysis
  - RSI-14 trend zone + ADX-14 strength

M1 (1-min Technical Entry) - Entry timing:
  - RSI oversold + MACD + Trend EMA (mandatory)
  - Bollinger Bands + Volume + Candlesticks
  - Williams %R + CCI confirmation

M2 (5-min Price Action) - Confirmation:
  - Candle colors + HH/HL patterns
  - Momentum + Bar strength
  - ATR volatility filter + VWAP distance

Exit Conditions (SELL):
1. Take Profit: +1.5%
2. Stop Loss: -1.0%
3. Trailing Stop after +0.8% profit (locks in gains)

Time Filter: Only trade 00:00-08:00 UTC and 13:00-21:00 UTC
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
import pytz
import uuid

from ..strategy.base import Strategy, StrategyConfig
from ..strategy.logger import LogColor
from ..core.models import (
    Bar, Order, OrderSide, OrderType, OrderStatus, TimeInForce,
    Position, Instrument, InstrumentType
)
from ..core.events import FillEvent
from ..indicators import (
    ExponentialMovingAverage,
    SimpleMovingAverage,
    RelativeStrengthIndex,
    MACD,
    BollingerBands,
    AverageTrueRange,
    VWAP,
    VolumeMA,
    ADX,
    Stochastic,
)
# ML Ensemble disabled for now due to slow sklearn imports on Windows
# from ..indicators.ml_ensemble import MLEnsembleIndicator
MLEnsembleIndicator = None  # Disabled


# Supported crypto symbols on Alpaca (V6: reduced to 5 best performers)
ALPACA_CRYPTO_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"
]

# Per-symbol risk parameters based on volatility characteristics
# V10.2: Lowered min_entry_score by 1-2 points since we have multi-layer confirmation:
#   - MACRO(1H) provides bias/boost
#   - M0(15m) master trend must be UP
#   - M2(5m) price action must be BULLISH
# With these extra filters, we can be more lenient on M1 technical score
SYMBOL_RISK_PARAMS = {
    "BTC/USD": {
        "target_profit_pct": 1.2,    # Lower TP - BTC moves slower
        "stop_loss_pct": 0.8,        # Tighter SL - less noise
        "trailing_stop_pct": 0.4,
        "trailing_activation": 0.6,
        "min_entry_score": 4,        # V10.2: Lowered from 5 - M0+M2 already filtering
    },
    "ETH/USD": {
        "target_profit_pct": 1.3,    # Slightly higher than BTC
        "stop_loss_pct": 0.9,
        "trailing_stop_pct": 0.45,
        "trailing_activation": 0.7,
        "min_entry_score": 4,        # V10.2: Lowered from 5 - M0+M2 already filtering
    },
    "SOL/USD": {
        "target_profit_pct": 1.5,    # Standard V6 levels
        "stop_loss_pct": 1.0,
        "trailing_stop_pct": 0.5,
        "trailing_activation": 0.8,
        "min_entry_score": 5,        # V10.2: Lowered from 6 - M0+M2 already filtering
    },
    "DOGE/USD": {
        "target_profit_pct": 2.0,    # Wider TP - DOGE is volatile
        "stop_loss_pct": 1.2,        # Wider SL - more noise
        "trailing_stop_pct": 0.6,
        "trailing_activation": 1.0,
        "min_entry_score": 5,        # V10.2: Lowered from 7 - MACRO boost helps
    },
    "AVAX/USD": {
        "target_profit_pct": 1.5,    # Same as SOL
        "stop_loss_pct": 1.0,
        "trailing_stop_pct": 0.5,
        "trailing_activation": 0.8,
        "min_entry_score": 5,        # V10.2: Lowered from 6 - M0+M2 already filtering
    },
}


# Candlestick Pattern Types
class CandlePattern:
    NONE = "none"
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    BULLISH_ENGULFING = "bullish_engulfing"
    MORNING_STAR = "morning_star"
    DOJI = "doji"
    DRAGONFLY_DOJI = "dragonfly_doji"
    BULLISH_HARAMI = "bullish_harami"


@dataclass
class CryptoScalpingConfig(StrategyConfig):
    """
    Configuration for Crypto Scalping Strategy V2.

    Improved with pattern recognition and momentum filters.
    """
    # Trading symbols
    symbols: List[str] = field(default_factory=lambda: ALPACA_CRYPTO_SYMBOLS.copy())

    # Position sizing
    fixed_position_value: float = 500.0
    max_position_value: float = 2000.0

    # Profit/Loss targets - V6: More realistic for crypto volatility
    target_profit_pct: float = 1.5      # 1.5% take profit
    stop_loss_pct: float = 1.0          # 1.0% stop loss (1.5:1 ratio)
    trailing_stop_pct: float = 0.5      # 0.5% trailing stop
    trailing_stop_activation: float = 0.8  # Activate trailing stop after +0.8% profit
    use_trailing_stop: bool = True

    # Indicator parameters
    rsi_period: int = 14
    rsi_oversold: float = 35.0          # V6: Widened from 30 to 35 (less strict)
    rsi_overbought: float = 70.0        # Lower - exit earlier
    bb_period: int = 20
    bb_std_dev: float = 2.0
    vwap_period: int = 50
    volume_ma_period: int = 20
    volume_spike_multiplier: float = 1.3  # Volume confirmation
    adx_period: int = 14
    adx_trend_threshold: float = 20.0   # Need clear trend
    fast_ema_period: int = 9
    slow_ema_period: int = 21
    trend_ema_period: int = 50          # Longer EMA for trend direction
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_oversold: float = 20.0        # Stochastic oversold level
    stoch_overbought: float = 80.0      # Stochastic overbought level

    # Pattern recognition
    min_confirmations: int = 4          # Need 4+ confirmations to enter
    support_lookback: int = 50          # Bars to look back for support
    support_tolerance_pct: float = 0.5  # % tolerance for support level

    # Risk controls
    max_trades_per_day: int = 200       # Reduced - focus on quality
    max_concurrent_positions: int = 9   # One per symbol max
    min_time_between_trades: int = 60   # 1 minute cooldown
    max_daily_loss: float = 1500.0      # Stop if losing too much

    # Entry quality score
    min_entry_score: int = 7            # V6: Increased from 6 to 7 (RSI + TREND + MACD + 4 more)

    # Minimum holding time (minutes) before technical exits
    min_hold_minutes: int = 15          # Must hold at least 15 minutes before RSI/MACD exits
    min_profit_for_technical_exit: float = 0.5  # Need at least 0.5% profit for technical exits

    # Time-of-day filter (UTC hours) - avoid low-volume periods
    # Peak crypto volume: 13:00-21:00 UTC (US trading hours)
    # Secondary peak: 00:00-08:00 UTC (Asian trading hours)
    use_time_filter: bool = True
    allowed_trading_hours: List[int] = field(default_factory=lambda: list(range(0, 9)) + list(range(13, 22)))  # 00-08 UTC and 13-21 UTC

    # ML Ensemble Configuration (disabled for now - sklearn imports too slow on Windows)
    use_ml_ensemble: bool = False       # Disabled - sklearn imports hang
    ml_model_path: str = "models/crypto_scalping_ensemble.pkl"  # Path to trained model
    ml_entry_threshold: float = 0.60    # Min ML probability for entry boost
    ml_strong_threshold: float = 0.70   # Min ML probability for strong boost


class SymbolState:
    """Tracks state for a single symbol with pattern recognition."""

    def __init__(self, symbol: str, config: CryptoScalpingConfig):
        self.symbol = symbol
        self.config = config

        # Per-symbol risk parameters (use symbol-specific or fall back to config defaults)
        symbol_params = SYMBOL_RISK_PARAMS.get(symbol, {})
        self.target_profit_pct = symbol_params.get('target_profit_pct', config.target_profit_pct)
        self.stop_loss_pct = symbol_params.get('stop_loss_pct', config.stop_loss_pct)
        self.trailing_stop_pct = symbol_params.get('trailing_stop_pct', config.trailing_stop_pct)
        self.trailing_stop_activation = symbol_params.get('trailing_activation', config.trailing_stop_activation)
        self.min_entry_score = symbol_params.get('min_entry_score', config.min_entry_score)

        # Core indicators
        self.rsi = RelativeStrengthIndex(config.rsi_period)
        self.bb = BollingerBands(config.bb_period, config.bb_std_dev)
        self.vwap = VWAP(period=config.vwap_period, reset_daily=False)
        self.volume_ma = VolumeMA(config.volume_ma_period)
        self.adx = ADX(config.adx_period)
        self.stoch = Stochastic(config.stoch_k_period, config.stoch_d_period)
        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)
        self.trend_ema = ExponentialMovingAverage(config.trend_ema_period)
        self.macd = MACD(12, 26, 9)
        self.atr = AverageTrueRange(14)

        # ML Ensemble Indicator (disabled for now)
        self.ml_ensemble = None
        # ML loading disabled - sklearn imports too slow on Windows
        # if config.use_ml_ensemble and MLEnsembleIndicator is not None:
        #     try:
        #         self.ml_ensemble = MLEnsembleIndicator(...)
        #     except Exception as e:
        #         pass

        # Bar history for pattern recognition
        self.bar_history: deque = deque(maxlen=100)
        self.price_history: deque = deque(maxlen=config.support_lookback)

        # MACD and Stochastic history for crossover detection
        self.macd_hist_prev: Optional[float] = None
        self.stoch_k_prev: Optional[float] = None
        self.stoch_d_prev: Optional[float] = None

        # Position tracking
        self.position: Optional[Position] = None
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[datetime] = None
        self.highest_price_since_entry: Optional[float] = None
        self.entry_score: int = 0
        self.entry_pattern: str = CandlePattern.NONE

        # Order tracking
        self.pending_entry_order_id: Optional[str] = None
        self.active_sl_order_id: Optional[str] = None
        self.active_tp_order_id: Optional[str] = None

        # Cooldown
        self.last_trade_time: Optional[datetime] = None
        self.consecutive_losses: int = 0

        # V10.5: Multi-timeframe bar storage (populated by engine)
        # These are set by the engine before calling on_bar
        self.bars_5min: Optional[list] = None   # For M2 (Price Action)
        self.bars_15min: Optional[list] = None  # For M0 (Master Trend)
        self.bars_1h: Optional[list] = None     # For MACRO context

        # V10.5: Cached layer results (to avoid recalculating)
        self.last_m0_result: Optional[dict] = None
        self.last_m2_result: Optional[dict] = None
        self.last_macro_result: Optional[dict] = None

    def update_indicators(self, bar: Bar) -> None:
        """Update all indicators with new bar data."""
        # Store previous MACD/Stoch values for crossover detection
        if self.macd.initialized:
            self.macd_hist_prev = self.macd.histogram
        if self.stoch.initialized:
            self.stoch_k_prev = self.stoch.k
            self.stoch_d_prev = self.stoch.d

        # Update core indicators
        self.rsi.update(bar.close)
        self.bb.update(bar.close)
        self.vwap.update_from_bar(bar)
        self.volume_ma.update_from_bar(bar)
        self.adx.update_from_bar(bar)
        self.stoch.update_from_bar(bar)
        self.fast_ema.update(bar.close)
        self.slow_ema.update(bar.close)
        self.trend_ema.update(bar.close)
        self.macd.update(bar.close)
        self.atr.update_from_bar(bar)

        # Store bar and price history
        self.bar_history.append(bar)
        self.price_history.append(bar.low)  # Track lows for support

        # Update ML ensemble if available
        if self.ml_ensemble is not None:
            try:
                # Prepare indicator dictionary for ML
                indicators_dict = {
                    'rsi': {'value': self.rsi.value if self.rsi.initialized else 50},
                    'macd': {
                        'macd': self.macd.value if self.macd.initialized else 0,
                        'signal': self.macd.signal if self.macd.initialized else 0,
                        'histogram': self.macd.histogram if self.macd.initialized else 0
                    },
                    'bollinger': {
                        'upper': self.bb.upper if self.bb.initialized else bar.close,
                        'middle': self.bb.middle if self.bb.initialized else bar.close,
                        'lower': self.bb.lower if self.bb.initialized else bar.close
                    },
                    'stochastic': {
                        'k': self.stoch.k if self.stoch.initialized else 50,
                        'd': self.stoch.d if self.stoch.initialized else 50
                    },
                    'adx': {'value': self.adx.value if self.adx.initialized else 25},
                    'atr': {'value': self.atr.value if self.atr.initialized else 0},
                    'close': bar.close
                }
                
                # Convert bar to dict
                bar_dict = {
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }
                
                self.ml_ensemble.update(bar_dict, indicators_dict)
            except Exception as e:
                # ML update failed - not critical
                pass

        # Update trailing stop tracking
        if self.position is not None and self.entry_price is not None:
            if self.highest_price_since_entry is None:
                self.highest_price_since_entry = bar.close
            else:
                self.highest_price_since_entry = max(self.highest_price_since_entry, bar.close)

    def indicators_ready(self) -> bool:
        """Check if all indicators are initialized."""
        return all([
            self.rsi.initialized,
            self.bb.initialized,
            self.vwap.initialized,
            self.volume_ma.initialized,
            self.adx.initialized,
            self.fast_ema.initialized,
            self.slow_ema.initialized,
            self.trend_ema.initialized,
            self.macd.initialized,
            self.stoch.initialized,
            len(self.bar_history) >= 3,  # Need at least 3 bars for patterns
        ])

    def reset_position_state(self) -> None:
        """Reset position-related state."""
        self.position = None
        self.entry_price = None
        self.entry_time = None
        self.highest_price_since_entry = None
        self.entry_score = 0
        self.entry_pattern = CandlePattern.NONE
        self.pending_entry_order_id = None
        self.active_sl_order_id = None
        self.active_tp_order_id = None

    def detect_candlestick_pattern(self) -> str:
        """
        Detect bullish candlestick reversal patterns.

        Returns pattern type or NONE if no pattern detected.
        """
        if len(self.bar_history) < 3:
            return CandlePattern.NONE

        current = self.bar_history[-1]
        prev = self.bar_history[-2]
        prev2 = self.bar_history[-3] if len(self.bar_history) >= 3 else None

        # Calculate body and wick sizes
        body = abs(current.close - current.open)
        upper_wick = current.high - max(current.open, current.close)
        lower_wick = min(current.open, current.close) - current.low
        total_range = current.high - current.low

        if total_range == 0:
            return CandlePattern.NONE

        body_pct = body / total_range
        upper_wick_pct = upper_wick / total_range
        lower_wick_pct = lower_wick / total_range

        is_bullish = current.close > current.open
        prev_is_bearish = prev.close < prev.open

        # 1. HAMMER: Small body at top, long lower wick (2x body), small upper wick
        if (body_pct < 0.35 and
            lower_wick_pct > 0.5 and
            upper_wick_pct < 0.15 and
            is_bullish):
            return CandlePattern.HAMMER

        # 2. DRAGONFLY DOJI: Almost no body, long lower wick, no upper wick
        if (body_pct < 0.1 and
            lower_wick_pct > 0.7 and
            upper_wick_pct < 0.1):
            return CandlePattern.DRAGONFLY_DOJI

        # 3. BULLISH ENGULFING: Current bullish candle completely engulfs previous bearish
        if (is_bullish and
            prev_is_bearish and
            current.open < prev.close and
            current.close > prev.open and
            body > abs(prev.close - prev.open) * 1.1):  # At least 10% bigger
            return CandlePattern.BULLISH_ENGULFING

        # 4. DOJI: Very small body (< 10% of range) - indecision, potential reversal
        if body_pct < 0.1:
            return CandlePattern.DOJI

        # 5. BULLISH HARAMI: Small bullish inside previous large bearish
        if (is_bullish and
            prev_is_bearish and
            current.high < prev.open and
            current.low > prev.close and
            body < abs(prev.close - prev.open) * 0.5):
            return CandlePattern.BULLISH_HARAMI

        # 6. MORNING STAR: 3-bar pattern - bearish, small body, bullish
        if prev2 is not None:
            prev2_is_bearish = prev2.close < prev2.open
            prev_body = abs(prev.close - prev.open)
            prev2_body = abs(prev2.close - prev2.open)

            if (prev2_is_bearish and
                prev_body < prev2_body * 0.3 and  # Small middle candle
                is_bullish and
                current.close > (prev2.open + prev2.close) / 2):  # Close above midpoint
                return CandlePattern.MORNING_STAR

        return CandlePattern.NONE

    def find_support_level(self) -> Optional[float]:
        """Find recent support level from price history."""
        if len(self.price_history) < 10:
            return None

        prices = list(self.price_history)

        # Find local minima as potential support levels
        supports = []
        for i in range(2, len(prices) - 2):
            if (prices[i] < prices[i-1] and
                prices[i] < prices[i-2] and
                prices[i] < prices[i+1] and
                prices[i] < prices[i+2]):
                supports.append(prices[i])

        if not supports:
            return min(prices[-20:]) if len(prices) >= 20 else min(prices)

        # Return average of recent support levels
        return sum(supports[-3:]) / len(supports[-3:]) if supports else None

    def is_near_support(self, price: float) -> bool:
        """Check if price is near a support level."""
        support = self.find_support_level()
        if support is None:
            return False

        tolerance = support * (self.config.support_tolerance_pct / 100)
        return abs(price - support) <= tolerance

    def is_macd_bullish_crossover(self) -> bool:
        """Check if MACD histogram is turning positive (momentum shift)."""
        if self.macd_hist_prev is None or not self.macd.initialized:
            return False
        return self.macd_hist_prev < 0 and self.macd.histogram >= 0

    def is_macd_momentum_positive(self) -> bool:
        """Check if MACD histogram is positive or improving."""
        if self.macd_hist_prev is None or not self.macd.initialized:
            return False
        # Either positive or improving (less negative)
        return self.macd.histogram > 0 or self.macd.histogram > self.macd_hist_prev

    def is_stoch_bullish_crossover(self) -> bool:
        """Check if Stochastic %K crossed above %D from oversold."""
        if (self.stoch_k_prev is None or
            self.stoch_d_prev is None or
            not self.stoch.initialized):
            return False

        # Previous: K below D, Current: K above D, in oversold territory
        was_below = self.stoch_k_prev < self.stoch_d_prev
        now_above = self.stoch.k > self.stoch.d
        is_oversold = self.stoch.k < self.config.stoch_overbought

        return was_below and now_above and is_oversold

    def is_stoch_oversold(self) -> bool:
        """Check if Stochastic is in oversold territory."""
        if not self.stoch.initialized:
            return False
        return self.stoch.k < self.config.stoch_oversold

    def is_trend_up(self, price: float) -> Tuple[bool, str]:
        """
        V6: Check if overall trend is UP - CRITICAL to avoid catching falling knives.

        Returns:
            (is_up, reason): True if trend is favorable for long entry
        """
        if not self.trend_ema.initialized:
            return False, "EMA not initialized"

        # Check 1: Price above 50 EMA (uptrend)
        price_above_ema = price > self.trend_ema.value

        # Check 2: EMA is rising (compare to 3 bars ago)
        ema_rising = False
        if len(self.bar_history) >= 4:
            # Calculate EMA slope using recent values
            current_ema = self.trend_ema.value
            # We need to estimate where EMA was 3 bars ago
            # Use fast EMA vs slow EMA as proxy for momentum direction
            if self.fast_ema.initialized and self.slow_ema.initialized:
                ema_rising = self.fast_ema.value > self.slow_ema.value

        # Check 3: MACD line above signal (momentum turning up)
        macd_bullish = False
        if self.macd.initialized:
            macd_bullish = self.macd.value > self.macd.signal

        # V6: Need at least ONE trend confirmation
        if price_above_ema:
            return True, "price > EMA50"
        if ema_rising:
            return True, "fast EMA > slow EMA"
        if macd_bullish:
            return True, "MACD > signal"

        return False, "downtrend"

    def is_macd_positive(self) -> bool:
        """V6: Check if MACD histogram is strictly positive (not just improving)."""
        if not self.macd.initialized:
            return False
        return self.macd.histogram > 0

    def check_v10_entry_conditions(self, m1_score: int) -> Tuple[bool, str, dict]:
        """
        V10.5: Check ALL V10 signal hierarchy layers for entry.

        This method validates the complete V10 signal stack:
          - MACRO (1H): Market bias/context - provides score adjustment
          - M0: Master Trend (15-min) = must be UP
          - M1: Technical Score (1-min) = adjusted by MACRO, must meet threshold
          - M2: Price Action (5-min) = must be BULLISH

        Returns:
            (can_enter, reason, context_dict)
        """
        from trading_system.strategies.crypto_scalping import CryptoScalping

        context = {
            'm0_ready': False,
            'm1_ready': False,
            'm2_ready': False,
            'm0_result': None,
            'm2_result': None,
            'macro_result': None,
            'adjusted_score': m1_score,
        }

        # ===== CHECK MACRO (1H) - Provides score adjustment =====
        if self.bars_1h and len(self.bars_1h) >= 20:
            macro_result = CryptoScalping.calculate_macro_context(self.bars_1h)
            context['macro_result'] = macro_result
            self.last_macro_result = macro_result

            score_adjustment = macro_result.get('score_adjustment', 0)
            adjusted_score = m1_score + score_adjustment
            context['adjusted_score'] = adjusted_score
        else:
            # No macro data - use base score
            adjusted_score = m1_score
            context['adjusted_score'] = adjusted_score

        # ===== CHECK M0: Master Trend (15-min) - MUST BE UP =====
        if self.bars_15min and len(self.bars_15min) >= 25:
            m0_result = CryptoScalping.calculate_master_trend_signal(self.bars_15min)
            context['m0_result'] = m0_result
            self.last_m0_result = m0_result

            if m0_result['trend'] == 'UP':
                context['m0_ready'] = True
            else:
                return False, f"M0 not UP ({m0_result['trend']})", context
        else:
            return False, "No 15-min bars for M0", context

        # ===== CHECK M1: Technical Score (adjusted) - MUST MEET THRESHOLD =====
        if adjusted_score >= self.min_entry_score:
            context['m1_ready'] = True
        else:
            return False, f"M1 score {adjusted_score} < {self.min_entry_score}", context

        # ===== CHECK M2: Price Action (5-min) - MUST BE BULLISH =====
        if self.bars_5min and len(self.bars_5min) >= 10:
            m2_result = CryptoScalping.calculate_price_action_signal(self.bars_5min)
            context['m2_result'] = m2_result
            self.last_m2_result = m2_result

            if m2_result['signal'] == 'BULLISH':
                context['m2_ready'] = True
            else:
                return False, f"M2 not BULLISH ({m2_result['signal']})", context
        else:
            return False, "No 5-min bars for M2", context

        # ALL LAYERS ALIGNED!
        return True, "All V10 layers aligned", context


class CryptoScalping(Strategy):
    """
    Crypto Scalping Strategy V2 Implementation.

    Improved with pattern recognition, momentum confirmation,
    and optimized risk/reward ratio.
    """

    def __init__(self, config: CryptoScalpingConfig):
        super().__init__(config)
        self.config: CryptoScalpingConfig = config

        self.log.info("=" * 80, LogColor.BLUE)
        self.log.info("INITIALIZING CRYPTO SCALPING STRATEGY V6", LogColor.BLUE)
        self.log.info("   V6 FIX: Mandatory TREND FILTER to avoid catching falling knives", LogColor.BLUE)
        self.log.info("=" * 80, LogColor.BLUE)

        # Initialize state for each symbol with per-symbol risk params
        self.symbol_states: Dict[str, SymbolState] = {}
        for symbol in config.symbols:
            state = SymbolState(symbol, config)
            self.symbol_states[symbol] = state
            self.log.info(f"   {symbol}: TP={state.target_profit_pct}% SL={state.stop_loss_pct}% MinScore={state.min_entry_score}", LogColor.CYAN)

        # Daily tracking
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.daily_pnl = 0.0
        self.last_trade_date: Optional[datetime] = None
        self.daily_trading_stopped = False

        # UTC timezone for crypto (24/7)
        self.utc_tz = pytz.UTC

        self.log.info(f"   Symbols: {len(config.symbols)}", LogColor.CYAN)
        self.log.info(f"   Position Size: ${config.fixed_position_value}", LogColor.CYAN)
        self.log.info(f"   TP/SL: +{config.target_profit_pct}% / -{config.stop_loss_pct}% (Ratio: {config.target_profit_pct/config.stop_loss_pct:.1f}:1)", LogColor.CYAN)
        self.log.info(f"   Min Entry Score: {config.min_entry_score} confirmations", LogColor.CYAN)
        self.log.info(f"   Time Filter: {config.use_time_filter} (Hours: 00-08 UTC, 13-21 UTC)", LogColor.CYAN)
        self.log.info("Strategy V10.5 initialization complete - V10 layers in STRATEGY", LogColor.GREEN)

    def on_start(self) -> None:
        """Strategy startup."""
        self.log.info("=" * 80, LogColor.GREEN)
        self.log.info("STARTING CRYPTO SCALPING STRATEGY V10", LogColor.GREEN)
        self.log.info("=" * 80, LogColor.GREEN)
        self.log.info("Strategy Parameters:", LogColor.CYAN)
        self.log.info(f"   RSI: {self.config.rsi_period} (oversold={self.config.rsi_oversold})", LogColor.CYAN)
        self.log.info(f"   BB: {self.config.bb_period} period, {self.config.bb_std_dev} std", LogColor.CYAN)
        self.log.info(f"   Stochastic: K={self.config.stoch_k_period}, D={self.config.stoch_d_period}", LogColor.CYAN)
        self.log.info(f"   Pattern Recognition: ENABLED", LogColor.CYAN)
        self.log.info(f"   Support/Resistance: {self.config.support_lookback} bar lookback", LogColor.CYAN)

    def on_bar(self, bar: Bar) -> None:
        """Process each bar."""
        try:
            symbol = bar.symbol

            # Check if this is a symbol we're tracking
            if symbol not in self.symbol_states:
                return

            state = self.symbol_states[symbol]

            # Get bar time in UTC
            bar_time_utc = bar.timestamp.astimezone(self.utc_tz) if bar.timestamp.tzinfo else \
                          self.utc_tz.localize(bar.timestamp)
            current_date = bar_time_utc.date()

            # Check for new trading day (UTC)
            if self.last_trade_date is None:
                self.last_trade_date = current_date
            elif current_date > self.last_trade_date:
                self._reset_daily_stats(current_date)

            # Update indicators
            state.update_indicators(bar)

            # Check if daily trading is stopped
            if self.daily_trading_stopped:
                return

            # Manage existing position
            if state.position is not None:
                self._manage_position(state, bar, bar_time_utc)
                return

            # Check if indicators are ready
            if not state.indicators_ready():
                return

            # Check risk limits
            if not self._can_trade(state, bar_time_utc):
                return

            # Calculate M1 entry score (1-min technical indicators)
            entry_score, pattern, signals = self._calculate_entry_score(state, bar)

            # V10.5: Check ALL V10 layers before entry
            # The strategy validates M0, M1 (adjusted by MACRO), and M2
            can_enter, reason, v10_context = state.check_v10_entry_conditions(entry_score)

            if can_enter:
                # Use adjusted score (includes MACRO boost)
                adjusted_score = v10_context.get('adjusted_score', entry_score)
                state.entry_score = adjusted_score
                state.entry_pattern = pattern

                # Log V10 layer alignment
                self.log.info("=" * 60, LogColor.GREEN)
                self.log.info(f"V10.5 ALL LAYERS ALIGNED: {symbol}", LogColor.GREEN)
                m0 = v10_context.get('m0_result', {})
                m2 = v10_context.get('m2_result', {})
                macro = v10_context.get('macro_result', {})
                if macro:
                    self.log.info(f"   MACRO (1H): {macro.get('bias', 'N/A')} | Adj: {macro.get('score_adjustment', 0):+d}", LogColor.CYAN)
                if m0:
                    self.log.info(f"   M0 (15m): {m0.get('trend', 'N/A')} ({m0.get('strength', 'N/A')})", LogColor.CYAN)
                self.log.info(f"   M1 (1m): {adjusted_score}/{state.min_entry_score} (base: {entry_score})", LogColor.CYAN)
                if m2:
                    self.log.info(f"   M2 (5m): {m2.get('signal', 'N/A')} ({m2.get('strength', 'N/A')})", LogColor.CYAN)

                self._enter_position(state, bar, bar_time_utc, signals)

        except Exception as e:
            self.log.error(f"Error in on_bar() for {bar.symbol}: {e}", LogColor.RED)
            import traceback
            traceback.print_exc()

    def _reset_daily_stats(self, current_date) -> None:
        """Reset daily statistics."""
        win_rate = (self.wins_today / self.trades_today * 100) if self.trades_today > 0 else 0

        self.log.info("=" * 80, LogColor.CYAN)
        self.log.info(f"NEW TRADING DAY: {current_date}", LogColor.CYAN)
        self.log.info(f"   Previous P&L: ${self.daily_pnl:.2f}", LogColor.CYAN)
        self.log.info(f"   Trades: {self.trades_today} (W:{self.wins_today} L:{self.losses_today})", LogColor.CYAN)
        self.log.info(f"   Win Rate: {win_rate:.1f}%", LogColor.CYAN)

        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.daily_pnl = 0.0
        self.last_trade_date = current_date
        self.daily_trading_stopped = False

        # Reset consecutive losses for all symbols
        for state in self.symbol_states.values():
            state.consecutive_losses = 0

    def _can_trade(self, state: SymbolState, current_time: datetime) -> bool:
        """Check if we can enter a new trade."""
        # Check time-of-day filter (avoid low-volume periods)
        if self.config.use_time_filter:
            current_hour = current_time.hour
            if current_hour not in self.config.allowed_trading_hours:
                return False

        # Check max trades per day
        if self.trades_today >= self.config.max_trades_per_day:
            return False

        # Check max concurrent positions
        active_positions = sum(1 for s in self.symbol_states.values() if s.position is not None)
        if active_positions >= self.config.max_concurrent_positions:
            return False

        # Check daily loss limit
        if self.daily_pnl <= -self.config.max_daily_loss:
            if not self.daily_trading_stopped:
                self.log.warning(f"DAILY LOSS LIMIT HIT: ${self.daily_pnl:.2f}", LogColor.RED)
                self.daily_trading_stopped = True
            return False

        # Check cooldown for this symbol
        if state.last_trade_time is not None:
            elapsed = (current_time - state.last_trade_time).total_seconds()
            if elapsed < self.config.min_time_between_trades:
                return False

        # Reduce trading after 3 consecutive losses on this symbol
        if state.consecutive_losses >= 3:
            # Require longer cooldown after losing streak
            if state.last_trade_time is not None:
                elapsed = (current_time - state.last_trade_time).total_seconds()
                if elapsed < self.config.min_time_between_trades * 3:  # 3x cooldown
                    return False

        return True

    def _calculate_williams_r(self, state: SymbolState) -> float:
        """
        V9: Calculate Williams %R indicator.

        Williams %R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100

        Range: 0 to -100
        - Above -20: Overbought
        - Below -80: Oversold

        Uses the price history from the state's bar history.
        """
        if len(state.bar_history) < 14:
            return -50.0  # Neutral if not enough data

        # Get last 14 bars for calculation
        recent_bars = list(state.bar_history)[-14:]
        highs = [b.high for b in recent_bars]
        lows = [b.low for b in recent_bars]
        current_close = recent_bars[-1].close

        highest_high = max(highs)
        lowest_low = min(lows)

        if highest_high == lowest_low:
            return -50.0  # Avoid division by zero

        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        return williams_r

    def _calculate_cci(self, state: SymbolState) -> float:
        """
        V9: Calculate Commodity Channel Index (CCI).

        CCI = (Typical Price - SMA of TP) / (0.015 * Mean Deviation)
        where Typical Price = (High + Low + Close) / 3

        Interpretation:
        - CCI > 100: Overbought
        - CCI < -100: Oversold

        Uses 20-period calculation.
        """
        period = 20
        if len(state.bar_history) < period:
            return 0.0  # Neutral if not enough data

        recent_bars = list(state.bar_history)[-period:]

        # Calculate Typical Prices
        typical_prices = [(b.high + b.low + b.close) / 3 for b in recent_bars]

        # SMA of Typical Prices
        sma_tp = sum(typical_prices) / period

        # Mean Deviation
        mean_deviation = sum(abs(tp - sma_tp) for tp in typical_prices) / period

        if mean_deviation == 0:
            return 0.0  # Avoid division by zero

        current_tp = typical_prices[-1]
        cci = (current_tp - sma_tp) / (0.015 * mean_deviation)
        return cci

    def _calculate_entry_score(self, state: SymbolState, bar: Bar) -> Tuple[int, str, Dict[str, bool]]:
        """
        V10.3: Calculate entry quality score - NO mandatory blockers.

        V10 already provides multi-layer filtering:
        - M0 (15-min): Master trend must be UP
        - M2 (5-min): Price action must be BULLISH
        - MACRO (1H): Provides score adjustment

        So the M1 (1-min) layer just needs to calculate a SCORE, not block.
        The engine will check M0+M2 before entering.

        Returns:
            score: Number of confirmations (no blockers, just points)
            pattern: Detected candlestick pattern
            signals: Dict of which signals are active
        """
        price = bar.close
        score = 0
        signals = {}

        # Initialize all signals to False
        signals['rsi_oversold'] = False
        signals['trend_up'] = False
        signals['trend_reason'] = "not checked"
        signals['macd_histogram_positive'] = False
        signals['below_bb'] = False
        signals['volume_spike'] = False
        signals['pattern'] = CandlePattern.NONE
        signals['stoch_oversold'] = False
        signals['near_support'] = False
        signals['below_vwap'] = False
        signals['adx_trending'] = False

        # ===== V10.3: RSI Score (not mandatory, just points) =====
        rsi_oversold = state.rsi.value < self.config.rsi_oversold
        signals['rsi_oversold'] = rsi_oversold

        if rsi_oversold:
            score += 2  # 2 points for RSI oversold
            if state.rsi.value < 25:  # Extra point for deeply oversold
                score += 1

        # ===== V10.3: TREND Score (not mandatory - M0 checks this) =====
        trend_up, trend_reason = state.is_trend_up(price)
        signals['trend_up'] = trend_up
        signals['trend_reason'] = trend_reason

        if trend_up:
            score += 2  # 2 points for trend up (bonus, M0 already checks)

        # ===== V10.3: MACD Score (not mandatory, just points) =====
        macd_positive = state.is_macd_positive()
        signals['macd_histogram_positive'] = macd_positive

        if macd_positive:
            score += 2  # 2 points for MACD positive

        # ===== OPTIONAL CONFIRMATIONS (add more points) =====

        # 4. Price at/below Lower BB - 1 point (oversold bounce setup)
        bb_signal = price <= state.bb.lower
        signals['below_bb'] = bb_signal
        if bb_signal:
            score += 1

        # 5. Volume spike - 1 point (capitulation/interest)
        volume_spike = state.volume_ma.is_spike(self.config.volume_spike_multiplier)
        signals['volume_spike'] = volume_spike
        if volume_spike:
            score += 1

        # 6. Bullish candlestick pattern - 1-2 points (reversal confirmation)
        pattern = state.detect_candlestick_pattern()
        signals['pattern'] = pattern
        if pattern != CandlePattern.NONE:
            score += 1
            if pattern in [CandlePattern.BULLISH_ENGULFING, CandlePattern.MORNING_STAR]:
                score += 1  # Extra point for strong patterns

        # 7. Stochastic oversold or bullish crossover - 1 point
        stoch_signal = state.is_stoch_oversold() or state.is_stoch_bullish_crossover()
        signals['stoch_oversold'] = stoch_signal
        if stoch_signal:
            score += 1

        # 8. Near support level - 1 point
        near_support = state.is_near_support(price)
        signals['near_support'] = near_support
        if near_support:
            score += 1

        # 9. Price below VWAP - 1 point (discount to average)
        below_vwap = state.vwap.is_price_below(price)
        signals['below_vwap'] = below_vwap
        if below_vwap:
            score += 1

        # 10. ADX trending - 1 point (only if trend is up!)
        # V6: Only add ADX point if we're in an uptrend (already confirmed)
        adx_trending = state.adx.is_trending(self.config.adx_trend_threshold)
        signals['adx_trending'] = adx_trending
        if adx_trending:
            score += 1

        # ===== V9: ADDITIONAL OVERSOLD CONFIRMATIONS =====

        # 11. V9: Williams %R - 1 point (oversold confirmation)
        # Williams %R < -80 = oversold (0 to -100 scale)
        williams_r = self._calculate_williams_r(state)
        signals['williams_r'] = williams_r
        signals['williams_oversold'] = williams_r < -80

        if williams_r < -80:
            score += 1
            if williams_r < -90:  # Extra deeply oversold
                score += 0.5

        # 12. V9: CCI (Commodity Channel Index) - 1 point
        # CCI < -100 = oversold (momentum divergence)
        cci = self._calculate_cci(state)
        signals['cci'] = cci
        signals['cci_oversold'] = cci < -100

        if cci < -100:
            score += 1
            if cci < -150:  # Extra point for extreme oversold
                score += 0.5

        # 13. ML Ensemble Prediction - 0 to 2 bonus points
        ml_score = 0
        ml_probability = 0.5
        ml_signal = "UNAVAILABLE"
        ml_confidence = "N/A"

        if state.ml_ensemble is not None and state.ml_ensemble.is_available:
            try:
                ml_probability = state.ml_ensemble.value
                ml_signal = state.ml_ensemble.signal
                ml_confidence = state.ml_ensemble.confidence

                if ml_probability >= self.config.ml_strong_threshold:
                    ml_score = 2
                elif ml_probability >= self.config.ml_entry_threshold:
                    ml_score = 1
                elif ml_probability >= 0.50:
                    ml_score = 0.5
                elif ml_probability < 0.45:
                    ml_score = -1  # ML says NO

                score += ml_score
            except:
                pass

        signals['ml_probability'] = ml_probability
        signals['ml_signal'] = ml_signal
        signals['ml_confidence'] = ml_confidence
        signals['ml_score'] = ml_score

        return score, pattern, signals

    def _enter_position(self, state: SymbolState, bar: Bar, bar_time: datetime, signals: Dict[str, bool]) -> None:
        """
        Signal an entry for a long position.

        V7: When paper_trading_mode is True, this method only sets the
        pending_entry_order_id flag. The CryptoPaperTradingEngine will then
        check dual signal confirmation before actually placing the order.
        """
        symbol = state.symbol
        price = bar.close

        self.log.info("=" * 60, LogColor.GREEN)
        self.log.info(f"ENTRY SIGNAL (METHOD 1 - Technical): {symbol} (Score: {state.entry_score})", LogColor.GREEN)
        self.log.info("=" * 60, LogColor.GREEN)

        # Log all signal details - V6 with mandatory trend filter
        self.log.info(f"   Price: ${price:.4f}", LogColor.CYAN)
        self.log.info(f"   Pattern: {state.entry_pattern}", LogColor.CYAN)
        self.log.info(f"   RSI: {state.rsi.value:.1f} {'[X]' if signals.get('rsi_oversold') else '[ ]'} (MANDATORY)", LogColor.CYAN)
        self.log.info(f"   TREND: {signals.get('trend_reason', 'N/A')} {'[X]' if signals.get('trend_up') else '[ ]'} (MANDATORY V6)", LogColor.GREEN)
        self.log.info(f"   MACD Hist: {state.macd.histogram:.4f} {'[X]' if signals.get('macd_histogram_positive') else '[ ]'} (MANDATORY V6)", LogColor.GREEN)
        self.log.info(f"   BB Lower: ${state.bb.lower:.4f} {'[X]' if signals.get('below_bb') else '[ ]'}", LogColor.CYAN)
        self.log.info(f"   Volume: {state.volume_ma.volume_ratio:.1f}x {'[X]' if signals.get('volume_spike') else '[ ]'}", LogColor.CYAN)
        self.log.info(f"   Stoch K/D: {state.stoch.k:.1f}/{state.stoch.d:.1f} {'[X]' if signals.get('stoch_oversold') else '[ ]'}", LogColor.CYAN)
        self.log.info(f"   Support: {'[X]' if signals.get('near_support') else '[ ]'}", LogColor.CYAN)

        # Log ML ensemble signal if available
        if signals.get('ml_score', 0) != 0:
            ml_prob = signals.get('ml_probability', 0.5)
            ml_sig = signals.get('ml_signal', 'N/A')
            ml_conf = signals.get('ml_confidence', 'N/A')
            ml_points = signals.get('ml_score', 0)
            self.log.info(f"   🤖 ML Ensemble: {ml_prob:.3f} ({ml_sig}) [{ml_conf}] +{ml_points:.1f} pts", LogColor.MAGENTA)

        # Calculate position size
        quantity = self.config.fixed_position_value / price

        self.log.info(f"   Quantity: {quantity:.6f}", LogColor.CYAN)
        self.log.info(f"   Value: ${self.config.fixed_position_value:.2f}", LogColor.CYAN)

        # V7: Set flag for paper trading engine to pick up
        # The engine will check METHOD 2 (Price Action 5-min) before placing order
        # Use a unique ID based on timestamp
        state.pending_entry_order_id = f"SIGNAL_{symbol}_{uuid.uuid4().hex[:8]}"
        state.last_trade_time = bar_time
        self.trades_today += 1

        self.log.info(f"   Signal ID: {state.pending_entry_order_id}", LogColor.GREEN)
        self.log.info(">>> METHOD 1 PASSED - Waiting for METHOD 2 (Price Action) confirmation...", LogColor.YELLOW)

    def _manage_position(self, state: SymbolState, bar: Bar, bar_time: datetime) -> None:
        """Manage an open position - check exit conditions using per-symbol risk params."""
        if state.position is None or state.entry_price is None:
            return

        price = bar.close
        entry_price = state.entry_price

        # Calculate current P&L percentage
        pnl_pct = ((price - entry_price) / entry_price) * 100

        # Calculate how long we've held the position
        hold_minutes = 0
        if state.entry_time is not None:
            hold_minutes = (bar_time - state.entry_time).total_seconds() / 60

        # Use per-symbol risk parameters
        tp_pct = state.target_profit_pct
        sl_pct = state.stop_loss_pct
        ts_pct = state.trailing_stop_pct
        ts_activation = state.trailing_stop_activation

        # Check exit conditions in priority order

        # 1. STOP LOSS - Always check first (no hold time requirement)
        if pnl_pct <= -sl_pct:
            self.log.info(f"SL HIT: {state.symbol} {pnl_pct:.2f}% (SL={sl_pct}%)", LogColor.RED)
            self._close_position(state, bar, "Stop Loss")
            state.consecutive_losses += 1
            self.losses_today += 1
            return

        # 2. TAKE PROFIT (no hold time requirement)
        if pnl_pct >= tp_pct:
            self.log.info(f"TP HIT: {state.symbol} +{pnl_pct:.2f}% (TP={tp_pct}%)", LogColor.GREEN)
            self._close_position(state, bar, "Take Profit")
            state.consecutive_losses = 0
            self.wins_today += 1
            return

        # 3. TRAILING STOP (only if in sufficient profit)
        if self.config.use_trailing_stop and state.highest_price_since_entry is not None:
            current_profit = ((state.highest_price_since_entry - entry_price) / entry_price) * 100

            # Only activate trailing stop after reaching activation threshold
            if current_profit >= ts_activation:
                trailing_stop_price = state.highest_price_since_entry * (1 - ts_pct / 100)
                if price <= trailing_stop_price:
                    self.log.info(f"TRAILING STOP: {state.symbol} +{pnl_pct:.2f}% (TS={ts_pct}%)", LogColor.YELLOW)
                    self._close_position(state, bar, "Trailing Stop")
                    if pnl_pct > 0:
                        state.consecutive_losses = 0
                        self.wins_today += 1
                    else:
                        state.consecutive_losses += 1
                        self.losses_today += 1
                    return

        # TECHNICAL EXITS DISABLED - Let trades develop to SL/TP
        # Only SL, TP, and Trailing Stop exits are active
        # RSI Overbought, MACD Reversal, and Upper BB exits removed

    def _close_position(self, state: SymbolState, bar: Bar, reason: str) -> None:
        """Close a position."""
        if state.position is None:
            return

        symbol = state.symbol
        price = bar.close

        self.log.info(f"CLOSING POSITION: {symbol} - {reason}", LogColor.YELLOW)

        # Calculate P&L
        if state.entry_price:
            pnl = (price - state.entry_price) * abs(state.position.quantity)
            pnl_pct = ((price - state.entry_price) / state.entry_price) * 100
            self.daily_pnl += pnl
            self.log.info(f"   Entry: ${state.entry_price:.4f}", LogColor.CYAN)
            self.log.info(f"   Exit: ${price:.4f}", LogColor.CYAN)
            self.log.info(f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)", LogColor.GREEN if pnl > 0 else LogColor.RED)

        # Create closing order
        close_order = self.order_factory.market(
            instrument_id=symbol,
            order_side=OrderSide.SELL,
            quantity=abs(state.position.quantity),
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
        )

        self.submit_order(close_order)

        # Reset state
        state.reset_position_state()

    def on_order_filled(self, event: FillEvent) -> None:
        """Handle order fills."""
        try:
            symbol = event.instrument_id
            fill_price = event.last_px
            fill_qty = event.last_qty

            if symbol not in self.symbol_states:
                return

            state = self.symbol_states[symbol]

            # Entry fill
            if event.client_order_id == state.pending_entry_order_id:
                state.entry_price = fill_price
                state.entry_time = event.timestamp
                state.highest_price_since_entry = fill_price

                # Get position from engine
                if self._engine:
                    state.position = self._engine.account.get_position(symbol)

                self.log.info(f"ENTRY FILLED: {symbol}", LogColor.GREEN)
                self.log.info(f"   Price: ${fill_price:.4f}", LogColor.GREEN)
                self.log.info(f"   Qty: {fill_qty:.6f}", LogColor.GREEN)
                self.log.info(f"   Score: {state.entry_score} | Pattern: {state.entry_pattern}", LogColor.GREEN)

                # Calculate TP/SL prices using per-symbol parameters
                tp_pct = state.target_profit_pct
                sl_pct = state.stop_loss_pct
                tp_price = fill_price * (1 + tp_pct / 100)
                sl_price = fill_price * (1 - sl_pct / 100)

                self.log.info(f"   Take Profit: ${tp_price:.4f} (+{tp_pct}%)", LogColor.BLUE)
                self.log.info(f"   Stop Loss: ${sl_price:.4f} (-{sl_pct}%)", LogColor.BLUE)
                self.log.info(f"   Risk/Reward: {tp_pct/sl_pct:.1f}:1", LogColor.BLUE)

                state.pending_entry_order_id = None

        except Exception as e:
            self.log.error(f"Error in on_order_filled(): {e}", LogColor.RED)
            import traceback
            traceback.print_exc()

    def on_position_closed(self, position: Position) -> None:
        """Handle position closed."""
        symbol = position.symbol

        if symbol in self.symbol_states:
            state = self.symbol_states[symbol]
            state.reset_position_state()

            if position.realized_pnl:
                pnl = float(position.realized_pnl)
                self.log.info(f"Position Closed: {symbol} P&L=${pnl:.2f}",
                            LogColor.GREEN if pnl > 0 else LogColor.RED)

    def on_stop(self) -> None:
        """Strategy shutdown."""
        self.log.info("STOPPING CRYPTO SCALPING STRATEGY V6", LogColor.YELLOW)

        # Close all open positions
        for symbol, state in self.symbol_states.items():
            if state.position is not None:
                self.log.info(f"Closing position: {symbol}", LogColor.YELLOW)
                self.close_all_positions(symbol)

        win_rate = (self.wins_today / self.trades_today * 100) if self.trades_today > 0 else 0

        self.log.info("=" * 80, LogColor.YELLOW)
        self.log.info("FINAL DAILY STATS:", LogColor.YELLOW)
        self.log.info(f"   Total Trades: {self.trades_today}", LogColor.CYAN)
        self.log.info(f"   Wins: {self.wins_today} | Losses: {self.losses_today}", LogColor.CYAN)
        self.log.info(f"   Win Rate: {win_rate:.1f}%", LogColor.CYAN)
        self.log.info(f"   Daily P&L: ${self.daily_pnl:.2f}", LogColor.CYAN)
        self.log.info("STRATEGY V6 STOPPED", LogColor.YELLOW)

    def on_reset(self) -> None:
        """Reset strategy state."""
        for state in self.symbol_states.values():
            state.reset_position_state()
            state.rsi.reset()
            state.bb.reset()
            state.vwap.reset()
            state.volume_ma.reset()
            state.adx.reset()
            state.stoch.reset()
            state.fast_ema.reset()
            state.slow_ema.reset()
            state.trend_ema.reset()
            state.macd.reset()
            state.atr.reset()
            state.bar_history.clear()
            state.price_history.clear()
            state.consecutive_losses = 0

        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None
        self.daily_trading_stopped = False

    # ========================================================================
    # DUAL SIGNAL VALIDATION - Price Action Analysis (5-MIN BARS)
    # ========================================================================

    @staticmethod
    def calculate_price_action_signal(bars: list) -> dict:
        """
        Calculate trading signal based on PRICE ACTION patterns.

        This is METHOD 2 for dual-confirmation signal validation.
        It analyzes recent candle patterns independent of technical indicators.
        Uses 5-MINUTE bars for better noise filtering on crypto.

        Parameters
        ----------
        bars : list
            List of bar objects with: open, close, high, low attributes

        Returns
        -------
        dict
            {
                'signal': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
                'strength': str,  # 'STRONG' | 'MODERATE' | 'WEAK'
                'bullish_points': int,
                'bearish_points': int,
                'reasons': list
            }
        """
        if not bars or len(bars) < 10:
            return {
                'signal': 'NEUTRAL',
                'strength': 'WEAK',
                'bullish_points': 0,
                'bearish_points': 0,
                'reasons': ['Not enough bars for price action analysis']
            }

        # Use last 10 bars for price action analysis
        recent_bars = bars[-10:]
        closes = [b.close for b in recent_bars]
        opens = [b.open for b in recent_bars]
        highs = [b.high for b in recent_bars]
        lows = [b.low for b in recent_bars]

        bullish_points = 0
        bearish_points = 0
        reasons = []

        # ========== 1. CANDLE COLOR COUNT (last 5 bars) ==========
        last_5 = recent_bars[-5:]
        green_candles = sum(1 for b in last_5 if b.close > b.open)
        red_candles = 5 - green_candles

        if green_candles >= 4:
            bullish_points += 3
            reasons.append(f"{green_candles}/5 bars bullish (green)")
        elif green_candles >= 3:
            bullish_points += 2
            reasons.append(f"{green_candles}/5 bars bullish")
        elif red_candles >= 4:
            bearish_points += 3
            reasons.append(f"{red_candles}/5 bars bearish (red)")
        elif red_candles >= 3:
            bearish_points += 2
            reasons.append(f"{red_candles}/5 bars bearish")

        # ========== 2. HIGHER HIGHS / LOWER LOWS (last 5 bars) ==========
        higher_highs = sum(1 for i in range(1, 5) if highs[-i] > highs[-i-1])
        lower_lows = sum(1 for i in range(1, 5) if lows[-i] < lows[-i-1])
        higher_lows = sum(1 for i in range(1, 5) if lows[-i] > lows[-i-1])
        lower_highs = sum(1 for i in range(1, 5) if highs[-i] < highs[-i-1])

        # Bullish: Higher highs AND higher lows (uptrend)
        if higher_highs >= 3 and higher_lows >= 2:
            bullish_points += 3
            reasons.append(f"Uptrend: {higher_highs} higher highs, {higher_lows} higher lows")
        elif higher_highs >= 2:
            bullish_points += 2
            reasons.append(f"Higher highs pattern ({higher_highs}/4)")

        # Bearish: Lower lows AND lower highs (downtrend)
        if lower_lows >= 3 and lower_highs >= 2:
            bearish_points += 3
            reasons.append(f"Downtrend: {lower_lows} lower lows, {lower_highs} lower highs")
        elif lower_lows >= 2:
            bearish_points += 2
            reasons.append(f"Lower lows pattern ({lower_lows}/4)")

        # ========== 3. PRICE vs 5-BAR AVERAGE ==========
        avg_5 = sum(closes[-5:]) / 5
        current_price = closes[-1]
        price_vs_avg = ((current_price - avg_5) / avg_5) * 100

        if price_vs_avg > 0.1:
            bullish_points += 2
            reasons.append(f"Price above 5-bar avg (+{price_vs_avg:.2f}%)")
        elif price_vs_avg < -0.1:
            bearish_points += 2
            reasons.append(f"Price below 5-bar avg ({price_vs_avg:.2f}%)")

        # ========== 4. MOMENTUM (5-bar change) ==========
        momentum_5 = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0

        if momentum_5 > 0.3:  # Higher threshold for crypto (more volatile)
            bullish_points += 2
            reasons.append(f"Strong 5-bar momentum (+{momentum_5:.2f}%)")
        elif momentum_5 > 0.1:
            bullish_points += 1
            reasons.append(f"Positive momentum (+{momentum_5:.2f}%)")
        elif momentum_5 < -0.3:
            bearish_points += 2
            reasons.append(f"Strong bearish momentum ({momentum_5:.2f}%)")
        elif momentum_5 < -0.1:
            bearish_points += 1
            reasons.append(f"Negative momentum ({momentum_5:.2f}%)")

        # ========== 5. LAST BAR ANALYSIS ==========
        last_bar = recent_bars[-1]
        bar_range = last_bar.high - last_bar.low
        bar_body = abs(last_bar.close - last_bar.open)
        body_ratio = bar_body / bar_range if bar_range > 0 else 0

        # Strong bullish bar: closes near high with good body
        if last_bar.close > last_bar.open and body_ratio > 0.6:
            upper_wick = last_bar.high - last_bar.close
            if upper_wick < bar_body * 0.3:  # Small upper wick
                bullish_points += 2
                reasons.append("Strong bullish last bar (closes near high)")

        # Strong bearish bar: closes near low with good body
        if last_bar.close < last_bar.open and body_ratio > 0.6:
            lower_wick = last_bar.close - last_bar.low
            if lower_wick < bar_body * 0.3:  # Small lower wick
                bearish_points += 2
                reasons.append("Strong bearish last bar (closes near low)")

        # ========== 6. V9: ATR VOLATILITY FILTER ==========
        # Calculate ATR to assess volatility
        # High ATR = good trading conditions (volatility)
        # Very low ATR = skip (no movement expected)
        true_ranges = []
        for i in range(1, len(recent_bars)):
            bar = recent_bars[i]
            prev_close = recent_bars[i-1].close
            tr1 = bar.high - bar.low
            tr2 = abs(bar.high - prev_close)
            tr3 = abs(bar.low - prev_close)
            true_ranges.append(max(tr1, tr2, tr3))

        if true_ranges:
            atr = sum(true_ranges) / len(true_ranges)
            atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

            if atr_pct > 0.2:  # Good volatility (>0.2% per bar)
                bullish_points += 1  # Boost signal confidence
                reasons.append(f"Good volatility (ATR: {atr_pct:.3f}%)")
            elif atr_pct < 0.05:  # Very low volatility
                # Penalize both sides - signal is less reliable
                bullish_points -= 1 if bullish_points > 0 else 0
                bearish_points -= 1 if bearish_points > 0 else 0
                reasons.append(f"Low volatility warning (ATR: {atr_pct:.3f}%)")

        # ========== 7. V9: VWAP DISTANCE (Volume-Weighted Average Price) ==========
        # Calculate VWAP from available bars
        # Price far below VWAP = oversold bounce opportunity
        # Price far above VWAP = potential mean reversion down
        volumes = [getattr(b, 'volume', 1) for b in recent_bars]
        total_volume = sum(volumes)

        if total_volume > 0:
            typical_prices = [(b.high + b.low + b.close) / 3 for b in recent_bars]
            vwap = sum(tp * v for tp, v in zip(typical_prices, volumes)) / total_volume
            vwap_distance_pct = ((current_price - vwap) / vwap) * 100 if vwap > 0 else 0

            if vwap_distance_pct < -0.3:  # Price significantly below VWAP
                bullish_points += 2
                reasons.append(f"Oversold vs VWAP ({vwap_distance_pct:.2f}%)")
            elif vwap_distance_pct < -0.1:
                bullish_points += 1
                reasons.append(f"Below VWAP ({vwap_distance_pct:.2f}%)")
            elif vwap_distance_pct > 0.3:  # Price significantly above VWAP
                bearish_points += 2
                reasons.append(f"Overbought vs VWAP (+{vwap_distance_pct:.2f}%)")
            elif vwap_distance_pct > 0.1:
                bearish_points += 1
                reasons.append(f"Above VWAP (+{vwap_distance_pct:.2f}%)")

        # ========== DECISION ==========
        # Need at least 5 points AND 3 point lead for strong signal
        if bullish_points >= 5 and bullish_points >= bearish_points + 3:
            signal = 'BULLISH'
            strength = 'STRONG'
        elif bearish_points >= 5 and bearish_points >= bullish_points + 3:
            signal = 'BEARISH'
            strength = 'STRONG'
        elif bullish_points >= 4 and bullish_points > bearish_points:
            signal = 'BULLISH'
            strength = 'MODERATE'
        elif bearish_points >= 4 and bearish_points > bullish_points:
            signal = 'BEARISH'
            strength = 'MODERATE'
        else:
            signal = 'NEUTRAL'
            strength = 'WEAK'

        return {
            'signal': signal,
            'strength': strength,
            'bullish_points': bullish_points,
            'bearish_points': bearish_points,
            'reasons': reasons
        }

    @staticmethod
    def calculate_master_trend_signal(bars_15min: list) -> dict:
        """
        Calculate the MASTER TREND signal from 15-minute bars.

        This is M0 (METHOD 0) - the higher timeframe trend filter.
        V8.1: More stable - uses only COMPLETED bars and longer EMA.

        Key changes for stability:
        - Skip the last bar (incomplete/updating)
        - Use EMA50 instead of EMA20 for smoother trend
        - Require stronger confirmation (higher thresholds)
        - Focus on TREND DIRECTION not short-term moves

        Parameters
        ----------
        bars_15min : list
            List of 15-minute bar objects with: open, close, high, low, volume

        Returns
        -------
        dict
            {
                'trend': 'UP' | 'DOWN' | 'NEUTRAL',
                'strength': str,  # 'STRONG' | 'MODERATE' | 'WEAK'
                'score': int,     # Total trend score
                'ema20_slope': float,  # EMA slope percentage
                'price_vs_ema': str,   # 'ABOVE' | 'BELOW'
                'reasons': list
            }
        """
        if not bars_15min or len(bars_15min) < 25:
            return {
                'trend': 'NEUTRAL',
                'strength': 'WEAK',
                'score': 0,
                'ema20_slope': 0.0,
                'price_vs_ema': 'N/A',
                'reasons': ['Not enough 15-min bars (need 25+)']
            }

        # V8.1: Skip the LAST bar (incomplete) - use only completed bars
        # This prevents flip-flopping from the updating current bar
        completed_bars = bars_15min[:-1]
        closes = [b.close for b in completed_bars]
        highs = [b.high for b in completed_bars]
        lows = [b.low for b in completed_bars]
        current_price = closes[-1]  # Last COMPLETED bar's close

        # Calculate EMAs for trend direction
        def calc_ema(prices, period):
            if len(prices) < period:
                return sum(prices) / len(prices)
            multiplier = 2 / (period + 1)
            ema = sum(prices[:period]) / period
            for price in prices[period:]:
                ema = (price - ema) * multiplier + ema
            return ema

        # V8.1: Use EMA20 and compare to 8 bars ago for slope (2 hours of 15-min bars)
        ema20_current = calc_ema(closes, 20)
        ema20_8_bars_ago = calc_ema(closes[:-8], 20) if len(closes) > 28 else ema20_current

        # EMA slope over 8 bars (~2 hours) - more stable
        ema_slope = ((ema20_current - ema20_8_bars_ago) / ema20_8_bars_ago) * 100 if ema20_8_bars_ago > 0 else 0

        bullish_score = 0
        bearish_score = 0
        reasons = []

        # ========== 1. EMA SLOPE (up to 5 points) - MOST IMPORTANT ==========
        # V8.1: Higher thresholds for stability
        if ema_slope > 0.25:
            bullish_score += 5
            reasons.append(f"EMA20 rising strongly (+{ema_slope:.2f}%)")
        elif ema_slope > 0.10:
            bullish_score += 3
            reasons.append(f"EMA20 rising (+{ema_slope:.2f}%)")
        elif ema_slope > 0.03:
            bullish_score += 1
            reasons.append(f"EMA20 slightly up (+{ema_slope:.2f}%)")
        elif ema_slope < -0.25:
            bearish_score += 5
            reasons.append(f"EMA20 falling strongly ({ema_slope:.2f}%)")
        elif ema_slope < -0.10:
            bearish_score += 3
            reasons.append(f"EMA20 falling ({ema_slope:.2f}%)")
        elif ema_slope < -0.03:
            bearish_score += 1
            reasons.append(f"EMA20 slightly down ({ema_slope:.2f}%)")

        # ========== 2. PRICE vs EMA20 (up to 3 points) ==========
        price_vs_ema = 'ABOVE' if current_price > ema20_current else 'BELOW'
        price_distance = ((current_price - ema20_current) / ema20_current) * 100

        if current_price > ema20_current:
            if price_distance > 0.5:
                bullish_score += 3
                reasons.append(f"Price above EMA20 (+{price_distance:.2f}%)")
            else:
                bullish_score += 2
                reasons.append(f"Price near EMA20 (+{price_distance:.2f}%)")
        else:
            if price_distance < -0.5:
                bearish_score += 3
                reasons.append(f"Price below EMA20 ({price_distance:.2f}%)")
            else:
                bearish_score += 2
                reasons.append(f"Price near EMA20 ({price_distance:.2f}%)")

        # ========== 3. COMPLETED CANDLE PATTERN (last 6 completed bars = 1.5 hours) ==========
        # V8.1: Look at more bars for stability
        last_6 = completed_bars[-6:]
        green_candles = sum(1 for b in last_6 if b.close > b.open)
        red_candles = 6 - green_candles

        if green_candles >= 5:
            bullish_score += 3
            reasons.append(f"{green_candles}/6 green bars (strong)")
        elif green_candles >= 4:
            bullish_score += 2
            reasons.append(f"{green_candles}/6 green bars")
        elif red_candles >= 5:
            bearish_score += 3
            reasons.append(f"{red_candles}/6 red bars (strong)")
        elif red_candles >= 4:
            bearish_score += 2
            reasons.append(f"{red_candles}/6 red bars")

        # ========== 4. HIGHER HIGHS / LOWER LOWS (last 6 bars) ==========
        # V8.1: Look at 6 bars for more reliable pattern
        higher_highs = sum(1 for i in range(1, 6) if highs[-i] > highs[-i-1])
        lower_lows = sum(1 for i in range(1, 6) if lows[-i] < lows[-i-1])
        higher_lows = sum(1 for i in range(1, 6) if lows[-i] > lows[-i-1])
        lower_highs = sum(1 for i in range(1, 6) if highs[-i] < highs[-i-1])

        if higher_highs >= 4 and higher_lows >= 3:
            bullish_score += 3
            reasons.append(f"Strong uptrend (HH:{higher_highs} HL:{higher_lows})")
        elif higher_highs >= 3:
            bullish_score += 1
            reasons.append(f"Higher highs ({higher_highs}/5)")

        if lower_lows >= 4 and lower_highs >= 3:
            bearish_score += 3
            reasons.append(f"Strong downtrend (LL:{lower_lows} LH:{lower_highs})")
        elif lower_lows >= 3:
            bearish_score += 1
            reasons.append(f"Lower lows ({lower_lows}/5)")

        # ========== 5. 2-HOUR MOMENTUM (8 bars = 2 hours) ==========
        # V8.1: Look at longer momentum for stability
        momentum_8 = ((closes[-1] - closes[-8]) / closes[-8]) * 100 if len(closes) >= 8 else 0

        if momentum_8 > 0.5:
            bullish_score += 2
            reasons.append(f"Strong 2hr momentum (+{momentum_8:.2f}%)")
        elif momentum_8 > 0.2:
            bullish_score += 1
            reasons.append(f"Positive 2hr momentum (+{momentum_8:.2f}%)")
        elif momentum_8 < -0.5:
            bearish_score += 2
            reasons.append(f"Strong bearish 2hr momentum ({momentum_8:.2f}%)")
        elif momentum_8 < -0.2:
            bearish_score += 1
            reasons.append(f"Negative 2hr momentum ({momentum_8:.2f}%)")

        # ========== 6. V9: RSI-14 TREND ZONE (up to 3 points) ==========
        # RSI > 50 = bullish zone, RSI < 50 = bearish zone
        # This confirms trend direction from a momentum perspective
        def calc_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50.0  # Neutral if not enough data
            gains = []
            losses = []
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                gains.append(max(0, change))
                losses.append(abs(min(0, change)))

            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period

            if avg_loss == 0:
                return 100.0 if avg_gain > 0 else 50.0
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        rsi_15min = calc_rsi(closes, 14)

        if rsi_15min > 60:
            bullish_score += 3
            reasons.append(f"RSI-14 bullish zone ({rsi_15min:.1f})")
        elif rsi_15min > 50:
            bullish_score += 1
            reasons.append(f"RSI-14 above neutral ({rsi_15min:.1f})")
        elif rsi_15min < 40:
            bearish_score += 3
            reasons.append(f"RSI-14 bearish zone ({rsi_15min:.1f})")
        elif rsi_15min < 50:
            bearish_score += 1
            reasons.append(f"RSI-14 below neutral ({rsi_15min:.1f})")

        # ========== 7. V9: ADX-14 TREND STRENGTH (up to 2 points) ==========
        # ADX > 25 = strong trend (confirms direction)
        # ADX < 20 = weak trend (NEUTRAL more likely)
        def calc_adx(highs_list, lows_list, closes_list, period=14):
            if len(highs_list) < period + 1:
                return 20.0  # Default neutral

            # Calculate True Range and Directional Movement
            plus_dm = []
            minus_dm = []
            tr = []

            for i in range(1, len(highs_list)):
                high_diff = highs_list[i] - highs_list[i-1]
                low_diff = lows_list[i-1] - lows_list[i]

                plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
                minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)

                tr1 = highs_list[i] - lows_list[i]
                tr2 = abs(highs_list[i] - closes_list[i-1])
                tr3 = abs(lows_list[i] - closes_list[i-1])
                tr.append(max(tr1, tr2, tr3))

            # Smoothed averages (last 'period' values)
            atr = sum(tr[-period:]) / period if len(tr) >= period else sum(tr) / len(tr)
            plus_di = (sum(plus_dm[-period:]) / period / atr * 100) if atr > 0 else 0
            minus_di = (sum(minus_dm[-period:]) / period / atr * 100) if atr > 0 else 0

            # DX and ADX
            di_sum = plus_di + minus_di
            dx = abs(plus_di - minus_di) / di_sum * 100 if di_sum > 0 else 0
            return dx  # Simplified ADX (single period DX)

        adx_15min = calc_adx(highs, lows, closes, 14)

        if adx_15min > 30:
            # Strong trend - boost the dominant direction
            if bullish_score > bearish_score:
                bullish_score += 2
                reasons.append(f"ADX strong trend ({adx_15min:.1f}) confirms UP")
            elif bearish_score > bullish_score:
                bearish_score += 2
                reasons.append(f"ADX strong trend ({adx_15min:.1f}) confirms DOWN")
        elif adx_15min > 20:
            if bullish_score > bearish_score:
                bullish_score += 1
                reasons.append(f"ADX moderate trend ({adx_15min:.1f})")
            elif bearish_score > bullish_score:
                bearish_score += 1
                reasons.append(f"ADX moderate trend ({adx_15min:.1f})")
        else:
            reasons.append(f"ADX weak/no trend ({adx_15min:.1f})")

        # ========== DECISION ==========
        # V8.1: Higher thresholds for more stable trend detection
        total_score = bullish_score - bearish_score

        if bullish_score >= 9 and bullish_score >= bearish_score + 5:
            trend = 'UP'
            strength = 'STRONG'
        elif bullish_score >= 6 and bullish_score >= bearish_score + 3:
            trend = 'UP'
            strength = 'MODERATE'
        elif bearish_score >= 9 and bearish_score >= bullish_score + 5:
            trend = 'DOWN'
            strength = 'STRONG'
        elif bearish_score >= 6 and bearish_score >= bullish_score + 3:
            trend = 'DOWN'
            strength = 'MODERATE'
        else:
            trend = 'NEUTRAL'
            strength = 'WEAK'

        return {
            'trend': trend,
            'strength': strength,
            'score': total_score,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'ema20_slope': ema_slope,
            'price_vs_ema': price_vs_ema,
            'reasons': reasons
        }

    @staticmethod
    def calculate_macro_context(bars_1h: list) -> dict:
        """
        V10.1: Calculate 1H MACRO CONTEXT using pandas-ta professional indicators.

        Uses pandas-ta for:
        - EMA 50/200 trend direction
        - Supertrend (GOLD for crypto)
        - ADX regime detection
        - RSI momentum
        - VWAP position (if volume available)

        The macro context provides BOOST for good setups, never blocks.

        Parameters
        ----------
        bars_1h : list
            List of 1-hour bar objects with: open, close, high, low, volume

        Returns
        -------
        dict with bias, regime, score_adjustment, etc.
        """
        import pandas as pd

        # Default neutral result if not enough data
        default_result = {
            'bias': 'NEUTRAL',
            'regime': 'UNKNOWN',
            'trend_strength': 0.0,
            'ema50': 0.0,
            'ema200': 0.0,
            'price_vs_ema50': 'N/A',
            'price_vs_ema200': 'N/A',
            'adx': 0.0,
            'supertrend': 'N/A',
            'rsi': 50.0,
            'support': 0.0,
            'resistance': 0.0,
            'score_adjustment': 0,
            'reasons': ['Insufficient 1H data']
        }

        if not bars_1h or len(bars_1h) < 20:
            return default_result

        try:
            # Import pandas-ta
            import pandas_ta as ta

            # Convert bars to DataFrame
            df = pd.DataFrame({
                'open': [b.open for b in bars_1h],
                'high': [b.high for b in bars_1h],
                'low': [b.low for b in bars_1h],
                'close': [b.close for b in bars_1h],
                'volume': [getattr(b, 'volume', 0) for b in bars_1h]
            })

            current_price = df['close'].iloc[-1]
            reasons = []
            score_adjustment = 0

            # ========== 1. EMA TREND (pandas-ta) ==========
            # Use shorter EMAs if limited bars available (24 bars = 24 hours)
            ema_short_len = min(20, len(df) - 5)  # Short EMA
            ema_long_len = min(50, len(df) - 1)   # Long EMA
            df['ema50'] = ta.ema(df['close'], length=ema_short_len)
            df['ema200'] = ta.ema(df['close'], length=ema_long_len)

            ema50 = df['ema50'].iloc[-1]
            ema200 = df['ema200'].iloc[-1]

            price_vs_ema50 = 'ABOVE' if current_price > ema50 else 'BELOW'
            price_vs_ema200 = 'ABOVE' if current_price > ema200 else 'BELOW'

            # ========== 2. SUPERTREND (pandas-ta) - GAME CHANGER ==========
            supertrend_df = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
            if supertrend_df is not None and not supertrend_df.empty:
                # Supertrend direction column is 'SUPERTd_10_3.0': 1 = bullish, -1 = bearish
                direction_col = [c for c in supertrend_df.columns if 'SUPERTd' in c]
                if direction_col:
                    st_direction = supertrend_df[direction_col[0]].iloc[-1]
                    if pd.notna(st_direction):
                        supertrend_signal = 'BULL' if st_direction == 1 else 'BEAR'
                    else:
                        supertrend_signal = 'N/A'
                else:
                    supertrend_signal = 'N/A'
            else:
                supertrend_signal = 'N/A'

            # ========== 3. ADX (pandas-ta) ==========
            adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx_df is not None and not adx_df.empty:
                adx = adx_df['ADX_14'].iloc[-1]
                plus_di = adx_df['DMP_14'].iloc[-1]
                minus_di = adx_df['DMN_14'].iloc[-1]
            else:
                adx = 20.0
                plus_di = 0
                minus_di = 0

            # ========== 4. RSI (pandas-ta) ==========
            rsi_series = ta.rsi(df['close'], length=14)
            rsi = rsi_series.iloc[-1] if rsi_series is not None else 50.0

            # ========== BIAS CALCULATION ==========
            # V10.1: Multi-signal consensus for bias

            bull_signals = 0
            bear_signals = 0

            # Signal 1: Price vs EMA50
            if current_price > ema50:
                bull_signals += 1
                reasons.append(f"Price > EMA50")
            else:
                bear_signals += 1
                reasons.append(f"Price < EMA50")

            # Signal 2: EMA50 vs EMA200 (Golden/Death Cross)
            if ema50 > ema200:
                bull_signals += 1
                reasons.append(f"EMA50 > EMA200 (Golden)")
            else:
                bear_signals += 1
                reasons.append(f"EMA50 < EMA200 (Death)")

            # Signal 3: Supertrend (HEAVY WEIGHT - 2 points)
            if supertrend_signal == 'BULL':
                bull_signals += 2
                reasons.append(f"Supertrend BULLISH")
            elif supertrend_signal == 'BEAR':
                bear_signals += 2
                reasons.append(f"Supertrend BEARISH")

            # Signal 4: RSI momentum
            if rsi > 55:
                bull_signals += 1
                reasons.append(f"RSI bullish ({rsi:.1f})")
            elif rsi < 45:
                bear_signals += 1
                reasons.append(f"RSI bearish ({rsi:.1f})")

            # Signal 5: DI+ vs DI- (trend direction)
            if plus_di > minus_di:
                bull_signals += 1
                reasons.append(f"DI+ > DI- ({plus_di:.1f} vs {minus_di:.1f})")
            elif minus_di > plus_di:
                bear_signals += 1
                reasons.append(f"DI- > DI+ ({minus_di:.1f} vs {plus_di:.1f})")

            # ========== DETERMINE BIAS ==========
            # V10.1: Consensus-based bias (out of 6 possible signals)
            if bull_signals >= 4:
                bias = 'BULLISH'
                score_adjustment = 2
            elif bull_signals >= 3:
                bias = 'BULLISH'
                score_adjustment = 1
            elif bear_signals >= 4:
                bias = 'BEARISH'
                score_adjustment = 0  # Don't penalize, just neutral
            elif bear_signals >= 3:
                bias = 'BEARISH'
                score_adjustment = 0
            else:
                bias = 'NEUTRAL'
                score_adjustment = 0

            reasons.append(f"Consensus: {bull_signals} BULL vs {bear_signals} BEAR")

            # ========== REGIME DETECTION ==========
            if adx > 25:
                regime = 'TRENDING'
                if bias == 'BULLISH':
                    score_adjustment = min(2, score_adjustment + 1)
                reasons.append(f"ADX={adx:.1f}: TRENDING")
            elif adx < 20:
                regime = 'RANGING'
                reasons.append(f"ADX={adx:.1f}: RANGING")
            else:
                regime = 'TRANSITIONING'
                reasons.append(f"ADX={adx:.1f}: TRANSITIONING")

            # ========== SUPPORT/RESISTANCE ==========
            lookback = min(24, len(df))
            resistance = df['high'].iloc[-lookback:].max()
            support = df['low'].iloc[-lookback:].min()

            price_range = resistance - support
            if price_range > 0:
                dist_to_support = (current_price - support) / price_range
                if dist_to_support < 0.15:
                    reasons.append(f"Near support ${support:.2f}")
                    if bias == 'BULLISH':
                        score_adjustment = min(2, score_adjustment + 1)

            # ========== TREND STRENGTH ==========
            trend_strength = min(100, max(0, adx + abs(bull_signals - bear_signals) * 10))

            # Clamp score adjustment
            score_adjustment = max(-2, min(2, score_adjustment))

            return {
                'bias': bias,
                'regime': regime,
                'trend_strength': trend_strength,
                'ema50': ema50,
                'ema200': ema200,
                'price_vs_ema50': price_vs_ema50,
                'price_vs_ema200': price_vs_ema200,
                'adx': adx,
                'supertrend': supertrend_signal,
                'rsi': rsi,
                'support': support,
                'resistance': resistance,
                'score_adjustment': score_adjustment,
                'reasons': reasons
            }

        except Exception as e:
            # Fallback to simple calculation if pandas-ta fails
            return {
                'bias': 'NEUTRAL',
                'regime': 'UNKNOWN',
                'trend_strength': 0.0,
                'ema50': 0.0,
                'ema200': 0.0,
                'price_vs_ema50': 'N/A',
                'price_vs_ema200': 'N/A',
                'adx': 0.0,
                'supertrend': 'N/A',
                'rsi': 50.0,
                'support': 0.0,
                'resistance': 0.0,
                'score_adjustment': 0,
                'reasons': [f'Error: {str(e)}']
            }
