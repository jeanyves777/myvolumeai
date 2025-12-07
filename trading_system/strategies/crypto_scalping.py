"""
Crypto Scalping Strategy V5 - 24/7 Spot Trading on Alpaca

An improved buy-low-sell-high scalping strategy with:
- Candlestick Pattern Recognition (Hammer, Engulfing, Doji)
- Multi-indicator Momentum Confirmation
- Support/Resistance Level Detection
- Trend Filter (only trade with momentum)
- Optimized Risk/Reward Ratio (3.3:1)
- Time-of-Day Filter (avoid low-volume hours)

SUPPORTED ASSETS (Alpaca Spot):
- BTC/USD, ETH/USD, SOL/USD, DOGE/USD, LINK/USD
- AVAX/USD, DOT/USD, LTC/USD, SHIB/USD

STRATEGY LOGIC V5:
==================
Entry Conditions (BUY) - Need 6+ confirmations (RSI mandatory):
1. RSI < 30 (oversold) - MANDATORY
2. Price at or below Lower Bollinger Band
3. Volume spike (current > 1.3x average)
4. Bullish candlestick pattern (Hammer, Engulfing, or Doji reversal)
5. MACD histogram turning positive (momentum shift)
6. Stochastic %K crossing above %D from oversold
7. Price near recent support level
8. Price below VWAP
9. ADX > 20 (trending market)

Exit Conditions (SELL):
1. Take Profit: +2.0% (3.3:1 ratio)
2. Stop Loss: -0.6% (tight but gives room)
3. Trailing Stop after +1.0% profit (locks in gains)
4. RSI > 75 (only after 15min hold and +0.5% profit)
5. Upper BB (only after 15min hold and +0.8% profit)

Time Filter: Only trade 00:00-08:00 UTC and 13:00-21:00 UTC
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
import pytz

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
from ..indicators.ml_ensemble import MLEnsembleIndicator


# Supported crypto symbols on Alpaca (removed UNI - delisted/no data)
ALPACA_CRYPTO_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "LINK/USD",
    "AVAX/USD", "DOT/USD", "LTC/USD", "SHIB/USD"
]


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

    # Profit/Loss targets - IMPROVED RATIO (3.3:1)
    target_profit_pct: float = 2.0      # 2.0% take profit (increased from 1.5%)
    stop_loss_pct: float = 0.6          # 0.6% stop loss (3.3:1 ratio)
    trailing_stop_pct: float = 0.5      # 0.5% trailing stop (slightly wider)
    trailing_stop_activation: float = 1.0  # Activate trailing stop after +1.0% profit
    use_trailing_stop: bool = True

    # Indicator parameters
    rsi_period: int = 14
    rsi_oversold: float = 30.0          # Stricter - only deeply oversold
    rsi_overbought: float = 75.0        # Higher - let winners run more
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
    min_entry_score: int = 6            # Minimum score to enter (RSI mandatory + 5 more confirmations)

    # Minimum holding time (minutes) before technical exits
    min_hold_minutes: int = 15          # Must hold at least 15 minutes before RSI/MACD exits
    min_profit_for_technical_exit: float = 0.5  # Need at least 0.5% profit for technical exits

    # Time-of-day filter (UTC hours) - avoid low-volume periods
    # Peak crypto volume: 13:00-21:00 UTC (US trading hours)
    # Secondary peak: 00:00-08:00 UTC (Asian trading hours)
    use_time_filter: bool = True
    allowed_trading_hours: List[int] = field(default_factory=lambda: list(range(0, 9)) + list(range(13, 22)))  # 00-08 UTC and 13-21 UTC

    # ML Ensemble Configuration
    use_ml_ensemble: bool = True        # Enable ML ensemble indicator
    ml_model_path: str = "models/crypto_scalping_ensemble.pkl"  # Path to trained model
    ml_entry_threshold: float = 0.60    # Min ML probability for entry boost
    ml_strong_threshold: float = 0.70   # Min ML probability for strong boost


class SymbolState:
    """Tracks state for a single symbol with pattern recognition."""

    def __init__(self, symbol: str, config: CryptoScalpingConfig):
        self.symbol = symbol
        self.config = config

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

        # ML Ensemble Indicator
        self.ml_ensemble: Optional[MLEnsembleIndicator] = None
        if config.use_ml_ensemble:
            try:
                self.ml_ensemble = MLEnsembleIndicator(
                    model_path=config.ml_model_path,
                    window=50
                )
                if self.ml_ensemble.is_available:
                    print(f"      ✅ ML Ensemble loaded for {symbol}")
                else:
                    print(f"      ⚠️ ML Ensemble not available for {symbol} (model not loaded)")
            except Exception as e:
                print(f"      ❌ ML Ensemble failed for {symbol}: {e}")
                import traceback
                traceback.print_exc()
                self.ml_ensemble = None

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
                        'macd': self.macd.macd if self.macd.initialized else 0,
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
        self.log.info("INITIALIZING CRYPTO SCALPING STRATEGY V5", LogColor.BLUE)
        self.log.info("=" * 80, LogColor.BLUE)

        # Initialize state for each symbol
        self.symbol_states: Dict[str, SymbolState] = {}
        for symbol in config.symbols:
            self.symbol_states[symbol] = SymbolState(symbol, config)
            self.log.info(f"   Tracking: {symbol}", LogColor.CYAN)

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
        self.log.info("Strategy V5 initialization complete", LogColor.GREEN)

    def on_start(self) -> None:
        """Strategy startup."""
        self.log.info("=" * 80, LogColor.GREEN)
        self.log.info("STARTING CRYPTO SCALPING STRATEGY V5", LogColor.GREEN)
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

            # Calculate entry score and check conditions
            entry_score, pattern, signals = self._calculate_entry_score(state, bar)

            if entry_score >= self.config.min_entry_score:
                state.entry_score = entry_score
                state.entry_pattern = pattern
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

    def _calculate_entry_score(self, state: SymbolState, bar: Bar) -> Tuple[int, str, Dict[str, bool]]:
        """
        Calculate entry quality score based on multiple confirmations.

        Returns:
            score: Number of confirmations (0 if RSI not oversold, 1-8 otherwise)
            pattern: Detected candlestick pattern
            signals: Dict of which signals are active
        """
        price = bar.close
        score = 0
        signals = {}

        # 1. RSI Oversold (MANDATORY - must be oversold to consider entry)
        rsi_oversold = state.rsi.value < self.config.rsi_oversold
        signals['rsi_oversold'] = rsi_oversold

        # If RSI is NOT oversold, return 0 score immediately (no entry allowed)
        if not rsi_oversold:
            signals['below_bb'] = False
            signals['volume_spike'] = False
            signals['pattern'] = CandlePattern.NONE
            signals['macd_positive'] = False
            signals['stoch_oversold'] = False
            signals['near_support'] = False
            signals['below_vwap'] = False
            signals['adx_trending'] = False
            return 0, CandlePattern.NONE, signals

        # RSI is oversold - start counting confirmations
        score += 1
        if state.rsi.value < 25:  # Extra point for deeply oversold
            score += 1

        # 2. Price at/below Lower BB - 1 point (strong oversold confirmation)
        bb_signal = price <= state.bb.lower
        signals['below_bb'] = bb_signal
        if bb_signal:
            score += 1

        # 3. Volume spike - 1 point (capitulation/interest)
        volume_spike = state.volume_ma.is_spike(self.config.volume_spike_multiplier)
        signals['volume_spike'] = volume_spike
        if volume_spike:
            score += 1

        # 4. Bullish candlestick pattern - 1-2 points (reversal confirmation)
        pattern = state.detect_candlestick_pattern()
        signals['pattern'] = pattern
        if pattern != CandlePattern.NONE:
            score += 1
            if pattern in [CandlePattern.BULLISH_ENGULFING, CandlePattern.MORNING_STAR]:
                score += 1  # Extra point for strong patterns

        # 5. MACD momentum positive or improving - 1 point
        macd_positive = state.is_macd_momentum_positive()
        signals['macd_positive'] = macd_positive
        if macd_positive:
            score += 1

        # 6. Stochastic oversold or bullish crossover - 1 point
        stoch_signal = state.is_stoch_oversold() or state.is_stoch_bullish_crossover()
        signals['stoch_oversold'] = stoch_signal
        if stoch_signal:
            score += 1

        # 7. Near support level - 1 point
        near_support = state.is_near_support(price)
        signals['near_support'] = near_support
        if near_support:
            score += 1

        # 8. Price below VWAP - 1 point (discount to average)
        below_vwap = state.vwap.is_price_below(price)
        signals['below_vwap'] = below_vwap
        if below_vwap:
            score += 1

        # 9. ADX trending - 1 point (confirms momentum exists)
        adx_trending = state.adx.is_trending(self.config.adx_trend_threshold)
        signals['adx_trending'] = adx_trending
        if adx_trending:
            score += 1

        # 10. ML Ensemble Prediction - 0 to 2 bonus points (HYBRID BOOST)
        ml_score = 0
        ml_probability = 0.5
        ml_signal = "UNAVAILABLE"
        ml_confidence = "N/A"
        
        if state.ml_ensemble is not None and state.ml_ensemble.is_available:
            try:
                ml_probability = state.ml_ensemble.value
                ml_signal = state.ml_ensemble.signal
                ml_confidence = state.ml_ensemble.confidence
                
                # Add ML score based on probability
                if ml_probability >= self.config.ml_strong_threshold:
                    ml_score = 2  # Very strong ML confirmation
                elif ml_probability >= self.config.ml_entry_threshold:
                    ml_score = 1  # Moderate ML confirmation
                elif ml_probability >= 0.50:
                    ml_score = 0.5  # Weak ML confirmation
                elif ml_probability < 0.45:
                    ml_score = -1  # ML says NO - reduce score
                
                score += ml_score
            except:
                pass
        
        signals['ml_probability'] = ml_probability
        signals['ml_signal'] = ml_signal
        signals['ml_confidence'] = ml_confidence
        signals['ml_score'] = ml_score

        return score, pattern, signals

    def _enter_position(self, state: SymbolState, bar: Bar, bar_time: datetime, signals: Dict[str, bool]) -> None:
        """Enter a long position with detailed logging."""
        symbol = state.symbol
        price = bar.close

        self.log.info("=" * 60, LogColor.GREEN)
        self.log.info(f"ENTRY SIGNAL: {symbol} (Score: {state.entry_score})", LogColor.GREEN)
        self.log.info("=" * 60, LogColor.GREEN)

        # Log all signal details
        self.log.info(f"   Price: ${price:.4f}", LogColor.CYAN)
        self.log.info(f"   Pattern: {state.entry_pattern}", LogColor.CYAN)
        self.log.info(f"   RSI: {state.rsi.value:.1f} {'[X]' if signals.get('rsi_oversold') else '[ ]'}", LogColor.CYAN)
        self.log.info(f"   BB Lower: ${state.bb.lower:.4f} {'[X]' if signals.get('below_bb') else '[ ]'}", LogColor.CYAN)
        self.log.info(f"   Volume: {state.volume_ma.volume_ratio:.1f}x {'[X]' if signals.get('volume_spike') else '[ ]'}", LogColor.CYAN)
        self.log.info(f"   MACD: {state.macd.histogram:.4f} {'[X]' if signals.get('macd_positive') else '[ ]'}", LogColor.CYAN)
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

        # Create and submit entry order
        entry_order = self.order_factory.market(
            instrument_id=symbol,
            order_side=OrderSide.BUY,
            quantity=quantity,
            time_in_force=TimeInForce.GTC,
        )

        self.submit_order(entry_order)

        # Track state
        state.pending_entry_order_id = entry_order.client_order_id
        state.last_trade_time = bar_time
        self.trades_today += 1

        self.log.info(f"   Order ID: {entry_order.client_order_id}", LogColor.GREEN)
        self.log.info("ENTRY ORDER SUBMITTED", LogColor.GREEN)

    def _manage_position(self, state: SymbolState, bar: Bar, bar_time: datetime) -> None:
        """Manage an open position - check exit conditions."""
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

        # Check exit conditions in priority order

        # 1. STOP LOSS - Always check first (no hold time requirement)
        if pnl_pct <= -self.config.stop_loss_pct:
            self.log.info(f"SL HIT: {state.symbol} {pnl_pct:.2f}%", LogColor.RED)
            self._close_position(state, bar, "Stop Loss")
            state.consecutive_losses += 1
            self.losses_today += 1
            return

        # 2. TAKE PROFIT (no hold time requirement)
        if pnl_pct >= self.config.target_profit_pct:
            self.log.info(f"TP HIT: {state.symbol} +{pnl_pct:.2f}%", LogColor.GREEN)
            self._close_position(state, bar, "Take Profit")
            state.consecutive_losses = 0
            self.wins_today += 1
            return

        # 3. TRAILING STOP (only if in sufficient profit)
        if self.config.use_trailing_stop and state.highest_price_since_entry is not None:
            current_profit = ((state.highest_price_since_entry - entry_price) / entry_price) * 100

            # Only activate trailing stop after reaching activation threshold
            if current_profit >= self.config.trailing_stop_activation:
                trailing_stop_price = state.highest_price_since_entry * (1 - self.config.trailing_stop_pct / 100)
                if price <= trailing_stop_price:
                    self.log.info(f"TRAILING STOP: {state.symbol} +{pnl_pct:.2f}%", LogColor.YELLOW)
                    self._close_position(state, bar, "Trailing Stop")
                    if pnl_pct > 0:
                        state.consecutive_losses = 0
                        self.wins_today += 1
                    else:
                        state.consecutive_losses += 1
                        self.losses_today += 1
                    return

        # === TECHNICAL EXITS - REQUIRE MINIMUM HOLD TIME AND PROFIT ===
        # Don't allow technical exits too early or with insufficient profit
        can_technical_exit = (
            hold_minutes >= self.config.min_hold_minutes and
            pnl_pct >= self.config.min_profit_for_technical_exit
        )

        if not can_technical_exit:
            return  # Only SL/TP/Trailing can exit early

        # 4. RSI OVERBOUGHT (requires min hold time and min profit)
        if state.rsi.value >= self.config.rsi_overbought:
            self.log.info(f"RSI OVERBOUGHT: {state.symbol} RSI={state.rsi.value:.1f} +{pnl_pct:.2f}% (held {hold_minutes:.0f}m)", LogColor.YELLOW)
            self._close_position(state, bar, "RSI Overbought")
            state.consecutive_losses = 0
            self.wins_today += 1
            return

        # 5. MACD MOMENTUM REVERSAL (requires min hold and profit)
        if pnl_pct > 0.5 and state.macd_hist_prev is not None:
            # Exit if MACD histogram turns negative from positive
            if state.macd_hist_prev > 0 and state.macd.histogram < 0:
                self.log.info(f"MACD REVERSAL: {state.symbol} +{pnl_pct:.2f}% (held {hold_minutes:.0f}m)", LogColor.YELLOW)
                self._close_position(state, bar, "MACD Reversal")
                state.consecutive_losses = 0
                self.wins_today += 1
                return

        # 6. UPPER BB (requires min hold and good profit)
        if pnl_pct >= 0.8 and price >= state.bb.upper:
            self.log.info(f"UPPER BB: {state.symbol} +{pnl_pct:.2f}% (held {hold_minutes:.0f}m)", LogColor.YELLOW)
            self._close_position(state, bar, "Upper BB")
            state.consecutive_losses = 0
            self.wins_today += 1
            return

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

                # Calculate TP/SL prices
                tp_price = fill_price * (1 + self.config.target_profit_pct / 100)
                sl_price = fill_price * (1 - self.config.stop_loss_pct / 100)

                self.log.info(f"   Take Profit: ${tp_price:.4f} (+{self.config.target_profit_pct}%)", LogColor.BLUE)
                self.log.info(f"   Stop Loss: ${sl_price:.4f} (-{self.config.stop_loss_pct}%)", LogColor.BLUE)

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
        self.log.info("STOPPING CRYPTO SCALPING STRATEGY V2", LogColor.YELLOW)

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
        self.log.info("STRATEGY V2 STOPPED", LogColor.YELLOW)

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
