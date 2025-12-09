"""
ETH Margin Scalping Strategy V10 - Kraken Futures

Single-asset margin trading strategy for ETH perpetual (PI_ETHUSD) on Kraken Futures.

VERSION 10 SIGNAL HIERARCHY (same as crypto spot):
===================================================
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
  - ATR volatility filter

MARGIN TRADING SPECIFICS:
- Uses leverage (default 5x, max 10x for demo)
- Supports LONG positions only (matches spot strategy)
- Tighter risk management due to margin risk
- Same V10 signal hierarchy as crypto spot

Exit Conditions:
1. Take Profit: +1.5%
2. Stop Loss: -1.0%
3. Trailing Stop after +0.8% profit
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
import pytz

# Import indicators
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

# UTC timezone
UTC = pytz.UTC

# ETH-specific risk parameters (adapted from crypto spot)
ETH_RISK_PARAMS = {
    "target_profit_pct": 1.3,    # Slightly higher than BTC
    "stop_loss_pct": 0.9,
    "trailing_stop_pct": 0.45,
    "trailing_activation": 0.7,
    "min_entry_score": 4,        # V10.2: Lowered since M0+M2 already filtering
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
class BarData:
    """Simple bar data structure for Kraken."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime


@dataclass
class ETHMarginConfig:
    """Configuration for ETH Margin Scalping Strategy."""

    # Position sizing
    position_value_usd: float = 500.0  # USD value per trade
    leverage: float = 5.0  # Leverage multiplier

    # Profit/Loss targets - V6: More realistic for crypto volatility
    target_profit_pct: float = 1.3      # 1.3% take profit (ETH specific)
    stop_loss_pct: float = 0.9          # 0.9% stop loss
    trailing_stop_pct: float = 0.45     # 0.45% trailing stop
    trailing_stop_activation: float = 0.7  # Activate after +0.7% profit
    use_trailing_stop: bool = True

    # Indicator parameters
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 70.0
    bb_period: int = 20
    bb_std_dev: float = 2.0
    vwap_period: int = 50
    volume_ma_period: int = 20
    volume_spike_multiplier: float = 1.3
    adx_period: int = 14
    adx_trend_threshold: float = 20.0
    fast_ema_period: int = 9
    slow_ema_period: int = 21
    trend_ema_period: int = 50
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0

    # Risk controls
    max_trades_per_hour: int = 5
    min_time_between_trades: int = 120  # 2 minute cooldown

    # Entry quality score
    min_entry_score: int = 4  # V10.2: Lower threshold with M0+M2 filtering

    # Time filter
    use_time_filter: bool = True
    allowed_trading_hours: List[int] = field(
        default_factory=lambda: list(range(0, 9)) + list(range(13, 22))
    )


class ETHMarginState:
    """Tracks state for ETH margin trading."""

    def __init__(self, config: ETHMarginConfig):
        self.symbol = "PI_ETHUSD"
        self.config = config

        # Risk parameters
        self.target_profit_pct = config.target_profit_pct
        self.stop_loss_pct = config.stop_loss_pct
        self.trailing_stop_pct = config.trailing_stop_pct
        self.trailing_stop_activation = config.trailing_stop_activation
        self.min_entry_score = config.min_entry_score

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

        # Bar history for pattern recognition
        self.bar_history: deque = deque(maxlen=100)
        self.price_history: deque = deque(maxlen=50)

        # MACD and Stochastic history for crossover detection
        self.macd_hist_prev: Optional[float] = None
        self.stoch_k_prev: Optional[float] = None
        self.stoch_d_prev: Optional[float] = None

        # Position tracking
        self.has_position: bool = False
        self.position_side: Optional[str] = None  # 'long' or 'short'
        self.position_size: float = 0.0
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[datetime] = None
        self.highest_price_since_entry: Optional[float] = None
        self.lowest_price_since_entry: Optional[float] = None
        self.entry_score: int = 0
        self.entry_pattern: str = CandlePattern.NONE

        # Order tracking
        self.pending_entry_order_id: Optional[str] = None
        self.active_sl_order_id: Optional[str] = None
        self.active_tp_order_id: Optional[str] = None

        # Cooldown
        self.last_trade_time: Optional[datetime] = None
        self.consecutive_losses: int = 0

        # V10.5: Multi-timeframe bar storage
        self.bars_5min: Optional[list] = None   # For M2 (Price Action)
        self.bars_15min: Optional[list] = None  # For M0 (Master Trend)
        self.bars_1h: Optional[list] = None     # For MACRO context

        # V10.5: Cached layer results
        self.last_m0_result: Optional[dict] = None
        self.last_m2_result: Optional[dict] = None
        self.last_macro_result: Optional[dict] = None

    def update_indicators(self, bar: BarData) -> None:
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
        self.adx.update(bar.high, bar.low, bar.close)
        self.stoch.update(bar.high, bar.low, bar.close)
        self.fast_ema.update(bar.close)
        self.slow_ema.update(bar.close)
        self.trend_ema.update(bar.close)
        self.macd.update(bar.close)
        self.atr.update(bar.high, bar.low, bar.close)

        # Store bar and price history
        self.bar_history.append(bar)
        self.price_history.append(bar.low)

        # Update trailing stop tracking
        if self.has_position and self.entry_price is not None:
            if self.position_side == 'long':
                if self.highest_price_since_entry is None:
                    self.highest_price_since_entry = bar.close
                else:
                    self.highest_price_since_entry = max(self.highest_price_since_entry, bar.close)
            else:  # short
                if self.lowest_price_since_entry is None:
                    self.lowest_price_since_entry = bar.close
                else:
                    self.lowest_price_since_entry = min(self.lowest_price_since_entry, bar.close)

    def indicators_ready(self) -> bool:
        """Check if all indicators are initialized."""
        return all([
            self.rsi.initialized,
            self.bb.initialized,
            self.adx.initialized,
            self.fast_ema.initialized,
            self.slow_ema.initialized,
            self.trend_ema.initialized,
            self.macd.initialized,
            self.stoch.initialized,
            len(self.bar_history) >= 3,
        ])

    def reset_position_state(self) -> None:
        """Reset position-related state."""
        self.has_position = False
        self.position_side = None
        self.position_size = 0.0
        self.entry_price = None
        self.entry_time = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
        self.entry_score = 0
        self.entry_pattern = CandlePattern.NONE
        self.pending_entry_order_id = None
        self.active_sl_order_id = None
        self.active_tp_order_id = None

    def check_v10_entry_conditions(self, m1_score: int) -> Tuple[bool, str, dict]:
        """
        V10.5: Check ALL V10 signal hierarchy layers for entry.

        Returns:
            (can_enter, reason, context_dict)
        """
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
            macro_result = ETHMarginStrategy.calculate_macro_context(self.bars_1h)
            context['macro_result'] = macro_result
            self.last_macro_result = macro_result

            score_adjustment = macro_result.get('score_adjustment', 0)
            adjusted_score = m1_score + score_adjustment
            context['adjusted_score'] = adjusted_score
        else:
            adjusted_score = m1_score
            context['adjusted_score'] = adjusted_score

        # ===== CHECK M0: Master Trend (15-min) - MUST BE UP =====
        if self.bars_15min and len(self.bars_15min) >= 25:
            m0_result = ETHMarginStrategy.calculate_master_trend_signal(self.bars_15min)
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
            m2_result = ETHMarginStrategy.calculate_price_action_signal(self.bars_5min)
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


class ETHMarginStrategy:
    """
    ETH Margin Scalping Strategy V10 Implementation.

    Uses V10 signal hierarchy for entry with leverage.
    """

    def __init__(self, config: ETHMarginConfig):
        self.config = config
        self.state = ETHMarginState(config)

        # Daily tracking
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.daily_pnl = 0.0
        self.last_trade_date: Optional[datetime] = None

    def on_bar(self, bar: BarData) -> Optional[Dict[str, Any]]:
        """
        Process a new bar and return trading action if any.

        Returns:
            Dict with action details or None:
            {
                'action': 'enter_long' | 'exit' | None,
                'reason': str,
                'score': int,
                'context': dict
            }
        """
        # Get bar time in UTC
        bar_time_utc = bar.timestamp if bar.timestamp.tzinfo else UTC.localize(bar.timestamp)
        current_date = bar_time_utc.date()

        # Check for new trading day
        if self.last_trade_date is None:
            self.last_trade_date = current_date
        elif current_date > self.last_trade_date:
            self._reset_daily_stats(current_date)

        # Update indicators
        self.state.update_indicators(bar)

        # Check if indicators are ready
        if not self.state.indicators_ready():
            return None

        # Time filter
        if self.config.use_time_filter:
            if bar_time_utc.hour not in self.config.allowed_trading_hours:
                return {'action': None, 'reason': 'Outside trading hours'}

        # POSITION MANAGEMENT
        if self.state.has_position:
            return self._check_exit_conditions(bar, bar_time_utc)
        else:
            return self._check_entry_conditions(bar, bar_time_utc)

    def _check_entry_conditions(
        self, bar: BarData, bar_time_utc: datetime
    ) -> Optional[Dict[str, Any]]:
        """Check for entry conditions."""
        # Cooldown check
        if self.state.last_trade_time:
            seconds_since_last = (bar_time_utc - self.state.last_trade_time).total_seconds()
            if seconds_since_last < self.config.min_time_between_trades:
                return {'action': None, 'reason': 'Cooldown period'}

        # Calculate M1 entry score
        entry_score, pattern, signals = self._calculate_entry_score(bar)

        # V10.5: Check ALL V10 layers
        can_enter, reason, v10_context = self.state.check_v10_entry_conditions(entry_score)

        if can_enter:
            adjusted_score = v10_context.get('adjusted_score', entry_score)
            return {
                'action': 'enter_long',
                'reason': f"V10 ENTRY: {reason}",
                'score': adjusted_score,
                'pattern': pattern,
                'signals': signals,
                'context': v10_context,
                'price': bar.close
            }
        else:
            return {
                'action': None,
                'reason': reason,
                'score': entry_score,
                'context': v10_context
            }

    def _check_exit_conditions(
        self, bar: BarData, bar_time_utc: datetime
    ) -> Optional[Dict[str, Any]]:
        """Check for exit conditions on open position."""
        if not self.state.entry_price:
            return None

        current_price = bar.close
        entry_price = self.state.entry_price

        # Calculate P&L percentage (for long positions)
        pnl_pct = ((current_price - entry_price) / entry_price) * 100

        # 1. STOP LOSS
        if pnl_pct <= -self.state.stop_loss_pct:
            return {
                'action': 'exit',
                'reason': f"STOP_LOSS: {pnl_pct:.2f}%",
                'pnl_pct': pnl_pct,
                'price': current_price
            }

        # 2. TAKE PROFIT
        if pnl_pct >= self.state.target_profit_pct:
            return {
                'action': 'exit',
                'reason': f"TAKE_PROFIT: {pnl_pct:.2f}%",
                'pnl_pct': pnl_pct,
                'price': current_price
            }

        # 3. TRAILING STOP
        if self.config.use_trailing_stop and self.state.highest_price_since_entry:
            highest = self.state.highest_price_since_entry
            profit_from_high = ((highest - entry_price) / entry_price) * 100

            # Only activate trailing stop after reaching activation threshold
            if profit_from_high >= self.state.trailing_stop_activation:
                drawdown_from_high = ((highest - current_price) / highest) * 100

                if drawdown_from_high >= self.state.trailing_stop_pct:
                    return {
                        'action': 'exit',
                        'reason': f"TRAILING_STOP: Peak={profit_from_high:.2f}%, Drawdown={drawdown_from_high:.2f}%",
                        'pnl_pct': pnl_pct,
                        'price': current_price
                    }

        return None

    def _calculate_entry_score(self, bar: BarData) -> Tuple[int, str, List[str]]:
        """
        Calculate M1 entry score based on technical indicators.

        Returns:
            (score, pattern, signal_list)
        """
        score = 0
        signals = []
        pattern = CandlePattern.NONE

        # 1. RSI OVERSOLD
        if self.state.rsi.initialized:
            rsi_val = self.state.rsi.value
            if rsi_val < self.config.rsi_oversold:
                score += 2
                signals.append(f"RSI={rsi_val:.1f} (oversold)")
            elif rsi_val < 45:
                score += 1
                signals.append(f"RSI={rsi_val:.1f} (low)")

        # 2. MACD BULLISH
        if self.state.macd.initialized:
            if self.state.macd.histogram > 0:
                score += 1
                signals.append("MACD bullish")

            # MACD crossover (histogram just turned positive)
            if self.state.macd_hist_prev is not None:
                if self.state.macd_hist_prev <= 0 and self.state.macd.histogram > 0:
                    score += 1
                    signals.append("MACD crossover")

        # 3. TREND EMA
        if self.state.trend_ema.initialized:
            if bar.close > self.state.trend_ema.value:
                score += 1
                signals.append("Above Trend EMA")

        # 4. BOLLINGER BANDS
        if self.state.bb.initialized:
            if bar.close < self.state.bb.lower:
                score += 2
                signals.append("Below BB lower")
            elif bar.close < self.state.bb.middle:
                score += 1
                signals.append("Below BB middle")

        # 5. STOCHASTIC
        if self.state.stoch.initialized:
            if self.state.stoch.k < self.config.stoch_oversold:
                score += 1
                signals.append(f"Stoch K={self.state.stoch.k:.1f} (oversold)")

            # Stochastic crossover
            if self.state.stoch_k_prev is not None and self.state.stoch_d_prev is not None:
                if (self.state.stoch_k_prev < self.state.stoch_d_prev and
                    self.state.stoch.k > self.state.stoch.d):
                    score += 1
                    signals.append("Stoch bullish crossover")

        # 6. ADX TREND STRENGTH
        if self.state.adx.initialized:
            if self.state.adx.value > self.config.adx_trend_threshold:
                score += 1
                signals.append(f"ADX={self.state.adx.value:.1f} (trending)")

        # 7. CANDLESTICK PATTERN
        pattern = self._detect_candlestick_pattern()
        if pattern != CandlePattern.NONE:
            score += 1
            signals.append(f"Pattern: {pattern}")

        return score, pattern, signals

    def _detect_candlestick_pattern(self) -> str:
        """Detect bullish candlestick reversal patterns."""
        if len(self.state.bar_history) < 3:
            return CandlePattern.NONE

        current = self.state.bar_history[-1]
        prev = self.state.bar_history[-2]

        body = abs(current.close - current.open)
        upper_wick = current.high - max(current.open, current.close)
        lower_wick = min(current.open, current.close) - current.low
        total_range = current.high - current.low

        if total_range == 0:
            return CandlePattern.NONE

        body_ratio = body / total_range

        # HAMMER: Small body at top, long lower wick
        if (lower_wick > 2 * body and
            upper_wick < body * 0.5 and
            current.close > current.open):
            return CandlePattern.HAMMER

        # BULLISH ENGULFING
        if (prev.close < prev.open and  # Previous was bearish
            current.close > current.open and  # Current is bullish
            current.open < prev.close and  # Opens below previous close
            current.close > prev.open):  # Closes above previous open
            return CandlePattern.BULLISH_ENGULFING

        # DOJI
        if body_ratio < 0.1:
            return CandlePattern.DOJI

        return CandlePattern.NONE

    def _reset_daily_stats(self, new_date):
        """Reset daily tracking stats."""
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.daily_pnl = 0.0
        self.last_trade_date = new_date

    def record_trade(self, pnl: float, is_win: bool):
        """Record a completed trade."""
        self.trades_today += 1
        self.daily_pnl += pnl
        if is_win:
            self.wins_today += 1
            self.state.consecutive_losses = 0
        else:
            self.losses_today += 1
            self.state.consecutive_losses += 1

    # ==================== V10 STATIC METHODS ====================

    @staticmethod
    def calculate_price_action_signal(bars: list) -> dict:
        """
        Calculate trading signal based on PRICE ACTION patterns (M2 layer).
        Uses 5-MINUTE bars.
        """
        if not bars or len(bars) < 10:
            return {
                'signal': 'NEUTRAL',
                'strength': 'WEAK',
                'bullish_points': 0,
                'bearish_points': 0,
                'reasons': ['Not enough bars']
            }

        recent_bars = bars[-10:]
        closes = [b.close for b in recent_bars]
        opens = [b.open for b in recent_bars]
        highs = [b.high for b in recent_bars]
        lows = [b.low for b in recent_bars]

        bullish_points = 0
        bearish_points = 0
        reasons = []

        # 1. CANDLE COLOR COUNT (last 5 bars)
        last_5 = recent_bars[-5:]
        green_candles = sum(1 for b in last_5 if b.close > b.open)
        red_candles = 5 - green_candles

        if green_candles >= 4:
            bullish_points += 3
            reasons.append(f"{green_candles}/5 bars bullish")
        elif green_candles >= 3:
            bullish_points += 2
        elif red_candles >= 4:
            bearish_points += 3
            reasons.append(f"{red_candles}/5 bars bearish")
        elif red_candles >= 3:
            bearish_points += 2

        # 2. HIGHER HIGHS / LOWER LOWS
        higher_highs = sum(1 for i in range(1, 5) if highs[-i] > highs[-i-1])
        higher_lows = sum(1 for i in range(1, 5) if lows[-i] > lows[-i-1])

        if higher_highs >= 3 and higher_lows >= 2:
            bullish_points += 3
            reasons.append("Uptrend pattern")
        elif higher_highs >= 2:
            bullish_points += 2

        # 3. PRICE vs 5-BAR AVERAGE
        avg_5 = sum(closes[-5:]) / 5
        current_price = closes[-1]
        price_vs_avg = ((current_price - avg_5) / avg_5) * 100

        if price_vs_avg > 0.1:
            bullish_points += 2
            reasons.append(f"Above 5-bar avg (+{price_vs_avg:.2f}%)")
        elif price_vs_avg < -0.1:
            bearish_points += 2

        # 4. MOMENTUM
        momentum_5 = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0

        if momentum_5 > 0.3:
            bullish_points += 2
            reasons.append(f"Strong momentum (+{momentum_5:.2f}%)")
        elif momentum_5 > 0.1:
            bullish_points += 1
        elif momentum_5 < -0.3:
            bearish_points += 2
        elif momentum_5 < -0.1:
            bearish_points += 1

        # Determine signal
        net_score = bullish_points - bearish_points

        if net_score >= 5:
            signal = 'BULLISH'
            strength = 'STRONG'
        elif net_score >= 3:
            signal = 'BULLISH'
            strength = 'MODERATE'
        elif net_score <= -5:
            signal = 'BEARISH'
            strength = 'STRONG'
        elif net_score <= -3:
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
        Calculate master trend signal (M0 layer).
        Uses 15-MINUTE bars.
        """
        if not bars_15min or len(bars_15min) < 25:
            return {
                'trend': 'NEUTRAL',
                'strength': 0,
                'ema20': 0,
                'ema20_slope': 0,
                'reasons': ['Not enough 15-min bars']
            }

        closes = [b.close for b in bars_15min]
        highs = [b.high for b in bars_15min]
        lows = [b.low for b in bars_15min]

        # Calculate EMA20
        ema_period = 20
        multiplier = 2 / (ema_period + 1)
        ema20 = sum(closes[:ema_period]) / ema_period

        for price in closes[ema_period:]:
            ema20 = (price - ema20) * multiplier + ema20

        # EMA slope (compare to 5 bars ago)
        if len(closes) >= 25:
            old_ema = sum(closes[-25:-5]) / 20
            for price in closes[-5:-1]:
                old_ema = (price - old_ema) * multiplier + old_ema
            ema_slope = ((ema20 - old_ema) / old_ema) * 100 if old_ema > 0 else 0
        else:
            ema_slope = 0

        # Price vs EMA
        current_price = closes[-1]
        price_vs_ema = ((current_price - ema20) / ema20) * 100 if ema20 > 0 else 0

        # Higher highs/lows analysis
        higher_highs = sum(1 for i in range(-4, 0) if highs[i] > highs[i-1])
        higher_lows = sum(1 for i in range(-4, 0) if lows[i] > lows[i-1])

        # Calculate trend score
        trend_score = 0
        reasons = []

        # EMA slope contribution
        if ema_slope > 0.1:
            trend_score += 2
            reasons.append(f"EMA20 rising (+{ema_slope:.2f}%)")
        elif ema_slope > 0:
            trend_score += 1
        elif ema_slope < -0.1:
            trend_score -= 2
            reasons.append(f"EMA20 falling ({ema_slope:.2f}%)")
        elif ema_slope < 0:
            trend_score -= 1

        # Price vs EMA contribution
        if price_vs_ema > 0.2:
            trend_score += 2
            reasons.append(f"Price above EMA (+{price_vs_ema:.2f}%)")
        elif price_vs_ema > 0:
            trend_score += 1
        elif price_vs_ema < -0.2:
            trend_score -= 2
        elif price_vs_ema < 0:
            trend_score -= 1

        # HH/HL contribution
        if higher_highs >= 3 and higher_lows >= 2:
            trend_score += 2
            reasons.append("Higher highs/lows pattern")
        elif higher_highs >= 2:
            trend_score += 1

        # Determine trend
        if trend_score >= 4:
            trend = 'UP'
        elif trend_score <= -4:
            trend = 'DOWN'
        else:
            trend = 'NEUTRAL'

        return {
            'trend': trend,
            'strength': abs(trend_score),
            'ema20': ema20,
            'ema20_slope': ema_slope,
            'price_vs_ema': price_vs_ema,
            'reasons': reasons
        }

    @staticmethod
    def calculate_macro_context(bars_1h: list) -> dict:
        """
        Calculate macro context (1H layer).
        Provides score adjustment, NOT a hard filter.
        """
        if not bars_1h or len(bars_1h) < 20:
            return {
                'bias': 'NEUTRAL',
                'score_adjustment': 0,
                'reasons': ['Not enough 1H bars']
            }

        closes = [b.close for b in bars_1h]

        # Calculate EMAs
        ema50 = sum(closes[-50:]) / min(50, len(closes))
        ema200 = sum(closes[-200:]) / min(200, len(closes)) if len(closes) >= 50 else ema50

        # Simple EMA calculations
        mult_50 = 2 / 51
        mult_200 = 2 / 201

        for price in closes[-20:]:
            ema50 = (price - ema50) * mult_50 + ema50
            ema200 = (price - ema200) * mult_200 + ema200

        current_price = closes[-1]
        score_adjustment = 0
        reasons = []

        # Price above/below EMAs
        if current_price > ema50 > ema200:
            score_adjustment += 2
            reasons.append("Bullish EMA alignment")
        elif current_price > ema50:
            score_adjustment += 1
            reasons.append("Above EMA50")
        elif current_price < ema50 < ema200:
            score_adjustment -= 2
            reasons.append("Bearish EMA alignment")
        elif current_price < ema50:
            score_adjustment -= 1
            reasons.append("Below EMA50")

        # Determine bias
        if score_adjustment >= 2:
            bias = 'BULLISH'
        elif score_adjustment <= -2:
            bias = 'BEARISH'
        else:
            bias = 'NEUTRAL'

        return {
            'bias': bias,
            'score_adjustment': score_adjustment,
            'ema50': ema50,
            'ema200': ema200,
            'reasons': reasons
        }
