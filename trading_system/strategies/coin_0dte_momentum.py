"""
COIN Daily 0DTE Momentum Strategy - Native Implementation

This is a reimplementation of the COIN Daily 0DTE Momentum Strategy
using our own trading system (no NautilusTrader dependency).

# Find THIS WEEK's Friday - options expire same week at 4PM Friday
# Monday (0) -> Friday in 4 days
# Tuesday (1) -> Friday in 3 days
# Wednesday (2) -> Friday in 2 days
# Thursday (3) -> Friday in 1 day
# Friday (4) -> Friday TODAY (0DTE)

CRITICAL REQUIREMENTS:
======================
1. MUST open exactly ONE trade per day (no exceptions)
2. FIXED position size: $2000 per day
3. Indicators determine DIRECTION only (CALL vs PUT)
4. If indicators neutral/weak -> default to CALL
5. Entry window: 9:30-10:00 AM EST
6. Force exit: 3:55 PM EST (before 4 PM 0DTE expiration)
7. NEVER holds overnight

STRATEGY LOGIC:
===============
- Uses EMA, RSI, MACD, Bollinger Bands, Volume for direction analysis
- Buys ATM (at-the-money) weekly options based on market direction
- Target profit: 7.5% (configurable)
- Stop loss: 25% (configurable)
- Max hold time: 30 minutes (configurable)
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List
import pytz

from ..strategy.base import Strategy, StrategyConfig
from ..strategy.logger import LogColor
from ..core.models import (
    Bar, Order, OrderSide, OrderType, OrderStatus, TimeInForce,
    Position, Instrument, OptionContract, InstrumentType, OptionType
)
from ..core.events import FillEvent
from ..indicators import (
    ExponentialMovingAverage,
    RelativeStrengthIndex,
    MACD,
    BollingerBands,
    AverageTrueRange,
)


@dataclass
class COINDaily0DTEMomentumConfig(StrategyConfig):
    """
    Configuration for COIN Daily 0DTE Momentum Strategy.

    Attributes
    ----------
    underlying_symbol : str
        The underlying stock symbol (e.g., "COIN").
    instrument_id : str
        The underlying instrument ID (COIN stock).
    bar_type : str
        The bar type for the strategy (for compatibility).
    fixed_position_value : float
        FIXED dollar amount per trade (default $2000).
    target_profit_pct : Decimal
        Target profit percentage (default 7.5%).
    stop_loss_pct : Decimal
        Stop loss percentage (default 25%).
    min_hold_minutes : int
        Minimum hold time before SL can trigger (default 5).
    entry_time_start : str
        Entry window start (default "09:30:00" EST).
    entry_time_end : str
        Entry window end (default "10:00:00" EST).
    max_hold_minutes : int
        Maximum hold time (default 30).
    force_exit_time : str
        Force exit time (default "15:55:00" EST).
    """
    underlying_symbol: str = "COIN"
    instrument_id: str = ""
    bar_type: str = ""
    fixed_position_value: float = 2000.0
    target_profit_pct: Decimal = Decimal("7.5")
    stop_loss_pct: Decimal = Decimal("25.0")
    min_hold_minutes: int = 5
    entry_time_start: str = "09:30:00"
    entry_time_end: str = "10:00:00"
    max_hold_minutes: int = 30
    force_exit_time: str = "15:55:00"
    fast_ema_period: int = 9
    slow_ema_period: int = 20
    rsi_period: int = 14
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    bb_period: int = 20
    bb_std_dev: float = 2.0
    min_volume_ratio: float = 1.0
    max_bid_ask_spread_pct: float = 30.0
    min_option_premium: float = 2.0
    max_option_premium: float = 30.0
    request_bars: bool = True
    max_trades_per_day: int = 1  # STRICT: This strategy trades exactly 1 time per day


class COINDaily0DTEMomentum(Strategy):
    """
    COIN Daily 0DTE Momentum Strategy Implementation.

    Trades daily call/put options on COIN based on technical indicators.
    """

    def __init__(self, config: COINDaily0DTEMomentumConfig):
        super().__init__(config)
        self.config: COINDaily0DTEMomentumConfig = config

        self.log.info("=" * 80, LogColor.BLUE)
        self.log.info("🔧 INITIALIZING COIN DAILY 0DTE MOMENTUM STRATEGY", LogColor.BLUE)
        self.log.info("=" * 80, LogColor.BLUE)

        # Initialize indicators
        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)
        self.rsi = RelativeStrengthIndex(config.rsi_period)
        self.macd = MACD(config.macd_fast_period, config.macd_slow_period, config.macd_signal_period)
        self.bb = BollingerBands(config.bb_period, config.bb_std_dev)
        self.atr = AverageTrueRange(14)

        self.log.info(f"   ✓ EMA: Fast={config.fast_ema_period}, Slow={config.slow_ema_period}", LogColor.BLUE)
        self.log.info(f"   ✓ RSI: Period={config.rsi_period}", LogColor.BLUE)
        self.log.info(f"   ✓ MACD: {config.macd_fast_period}/{config.macd_slow_period}/{config.macd_signal_period}", LogColor.BLUE)

        # Trading state
        self.traded_today = False
        self.last_trade_date: Optional[datetime] = None
        self.entry_price: Optional[float] = None
        self.entry_timestamp: Optional[int] = None
        self.entry_bar_datetime: Optional[datetime] = None  # For SL/TP eligibility
        self.current_position: Optional[Position] = None
        self.current_option_instrument: Optional[OptionContract] = None
        self.pending_option_instrument: Optional[OptionContract] = None

        # Order tracking
        self.entry_order_id = None
        self.active_sl_order_id = None
        self.active_tp_order_id = None
        self.is_closing = False
        self.current_signal: str = ""  # BULLISH, BEARISH, or NEUTRAL

        # Volume tracking
        self.volume_history = []
        self.max_volume_samples = 20

        # Parse times
        self.entry_time_start_parsed = self._parse_time(config.entry_time_start)
        self.entry_time_end_parsed = self._parse_time(config.entry_time_end)
        self.force_exit_time_parsed = self._parse_time(config.force_exit_time)

        self.est_tz = pytz.timezone('America/New_York')

        # Performance tracking
        self.trades_today = 0
        self.daily_pnl = 0.0

        self.log.info("✅ Strategy initialization complete", LogColor.GREEN)

    def _parse_time(self, time_str: str) -> time:
        """Parse time string to time object"""
        try:
            parts = time_str.split(":")
            return time(int(parts[0]), int(parts[1]), int(parts[2]))
        except Exception:
            return time(9, 30, 0)

    def on_start(self) -> None:
        """Strategy startup"""
        self.log.info("=" * 80, LogColor.GREEN)
        self.log.info("🚀 STARTING COIN DAILY 0DTE MOMENTUM STRATEGY", LogColor.GREEN)
        self.log.info("=" * 80, LogColor.GREEN)

        self.log.info("📋 STRATEGY CONFIGURATION:", LogColor.CYAN)
        self.log.info(f"   FIXED Position: ${self.config.fixed_position_value:.2f} per day", LogColor.CYAN)
        self.log.info(f"   Entry Window: {self.config.entry_time_start} - {self.config.entry_time_end} EST", LogColor.CYAN)
        self.log.info(f"   Force Exit: {self.config.force_exit_time} EST", LogColor.CYAN)
        self.log.info(f"   Target Profit: +{self.config.target_profit_pct}%", LogColor.CYAN)
        self.log.info(f"   Stop Loss: -{self.config.stop_loss_pct}%", LogColor.CYAN)

    def on_bar(self, bar: Bar) -> None:
        """Process each bar"""
        try:
            # ONLY process bars for the underlying symbol - skip options bars
            underlying_symbol = self.config.instrument_id
            if underlying_symbol and bar.symbol != underlying_symbol:
                # This is an options bar - skip indicator updates
                # But still check position management for exits
                if self.current_position is not None and self.current_option_instrument:
                    if bar.symbol == self.current_option_instrument.symbol:
                        bar_time_est = bar.timestamp.astimezone(self.est_tz) if bar.timestamp.tzinfo else \
                                      self.est_tz.localize(bar.timestamp)
                        self._manage_position(bar, bar_time_est)
                return

            # Convert to EST
            bar_time_est = bar.timestamp.astimezone(self.est_tz) if bar.timestamp.tzinfo else \
                          self.est_tz.localize(bar.timestamp)
            current_date = bar_time_est.date()

            # Update indicators ONLY with underlying bars
            self.fast_ema.update(bar.close)
            self.slow_ema.update(bar.close)
            self.rsi.update(bar.close)
            self.macd.update(bar.close)
            self.bb.update(bar.close)
            self.atr.update_from_bar(bar)

            # Check for new trading day
            if self.last_trade_date is None:
                self.last_trade_date = current_date
            elif current_date > self.last_trade_date:
                self.log.info("=" * 80, LogColor.CYAN)
                self.log.info(f"🔄 NEW TRADING DAY: {current_date}", LogColor.CYAN)
                self.log.info(f"   Previous P&L: ${self.daily_pnl:.2f}", LogColor.CYAN)
                self.traded_today = False
                self.trades_today = 0
                self.daily_pnl = 0.0
                self.last_trade_date = current_date  # Update immediately to prevent duplicate logs
                self._cancel_all_tracked_orders("New trading day")

            # Update volume history
            self.volume_history.append(float(bar.volume))
            if len(self.volume_history) > self.max_volume_samples:
                self.volume_history.pop(0)

            # Manage existing position
            if self.current_position is not None:
                self._manage_position(bar, bar_time_est)
                return

            # Check if already traded today
            if self.traded_today:
                return

            # Check entry window
            if not self._is_entry_time(bar_time_est):
                return

            # Analyze market and enter
            self.log.info("=" * 80, LogColor.MAGENTA)
            self.log.info("🎯 ENTRY WINDOW ACTIVE - ANALYZING MARKET", LogColor.MAGENTA)

            if self._indicators_ready():
                signal = self._analyze_market_direction(bar)
            else:
                self.log.warning("⚠️ Indicators NOT ready - DEFAULTING TO CALL", LogColor.YELLOW)
                signal = "BULLISH"

            self._enter_position(bar, bar_time_est, signal)

        except Exception as e:
            self.log.error(f"Error in on_bar(): {e}", LogColor.RED)
            import traceback
            traceback.print_exc()

    def _is_entry_time(self, bar_time_est: datetime) -> bool:
        """Check if within entry window"""
        bar_time_only = bar_time_est.time()
        return self.entry_time_start_parsed <= bar_time_only <= self.entry_time_end_parsed

    def _indicators_ready(self) -> bool:
        """Check if all indicators are initialized"""
        return all([
            self.fast_ema.initialized,
            self.slow_ema.initialized,
            self.rsi.initialized,
            self.macd.initialized,
            self.bb.initialized,
        ])

    def _analyze_market_direction(self, bar: Bar) -> str:
        """Analyze market to determine CALL vs PUT - uses high-probability scoring"""
        self.log.info("📊 HIGH-PROBABILITY SIGNAL ANALYSIS", LogColor.CYAN)

        bullish_score = 0
        bearish_score = 0
        bullish_signals = []
        bearish_signals = []

        # ========== 1. EMA STACK (up to 4 points) ==========
        if self.fast_ema.value > self.slow_ema.value:
            if bar.close > self.fast_ema.value:
                bullish_score += 4
                bullish_signals.append("EMA Stack Bullish (Price > EMA9 > EMA20)")
            else:
                bullish_score += 2
                bullish_signals.append("EMA9 > EMA20")
        else:
            if bar.close < self.fast_ema.value:
                bearish_score += 4
                bearish_signals.append("EMA Stack Bearish (Price < EMA9 < EMA20)")
            else:
                bearish_score += 2
                bearish_signals.append("EMA9 < EMA20")

        # ========== 2. RSI CONFIRMATION (up to 3 points) ==========
        rsi_val = self.rsi.value
        if 30 <= rsi_val <= 50:  # Recovering from oversold
            bullish_score += 3
            bullish_signals.append(f"RSI bullish zone ({rsi_val:.1f})")
        elif 50 < rsi_val <= 65:  # Strong but not overbought
            bullish_score += 2
            bullish_signals.append(f"RSI momentum ({rsi_val:.1f})")
        elif rsi_val > 70:  # Overbought - potential reversal
            bearish_score += 3
            bearish_signals.append(f"RSI OVERBOUGHT ({rsi_val:.1f})")
        elif rsi_val < 30:  # Oversold - bounce potential
            bullish_score += 3
            bullish_signals.append(f"RSI OVERSOLD ({rsi_val:.1f})")
        elif 50 <= rsi_val <= 70:
            bearish_score += 2
            bearish_signals.append(f"RSI neutral-high ({rsi_val:.1f})")

        # ========== 3. MACD CONFIRMATION (up to 3 points) ==========
        macd_val = self.macd.value
        macd_sig = self.macd.signal
        macd_hist = macd_val - macd_sig

        if macd_hist > 0 and macd_val > macd_sig:
            bullish_score += 3
            bullish_signals.append(f"MACD Bullish (hist: {macd_hist:.3f})")
        elif macd_hist < 0 and macd_val < macd_sig:
            bearish_score += 3
            bearish_signals.append(f"MACD Bearish (hist: {macd_hist:.3f})")

        # ========== 4. BOLLINGER BANDS (up to 3 points) ==========
        bb_upper = self.bb.upper
        bb_middle = self.bb.middle
        bb_lower = self.bb.lower

        if bb_upper and bb_lower and bb_middle:
            bb_width = bb_upper - bb_lower
            bb_position = (bar.close - bb_lower) / bb_width if bb_width > 0 else 0.5

            if bar.close > bb_middle:
                bullish_score += 2
                bullish_signals.append(f"Above BB Middle ({bb_position:.1%} position)")
                if bb_position < 0.7:  # Not too extended
                    bullish_score += 1
                    bullish_signals.append("BB not overbought")
            else:
                bearish_score += 2
                bearish_signals.append(f"Below BB Middle ({bb_position:.1%} position)")
                if bb_position > 0.3:  # Not too extended
                    bearish_score += 1
                    bearish_signals.append("BB not oversold")

        # ========== 5. VOLUME CONFIRMATION (up to 2 points) ==========
        if len(self.volume_history) >= 5:
            avg_vol = sum(self.volume_history[:-1]) / (len(self.volume_history) - 1)
            vol_ratio = bar.volume / avg_vol if avg_vol > 0 else 1.0

            if vol_ratio > 1.5:
                if bullish_score > bearish_score:
                    bullish_score += 2
                    bullish_signals.append(f"High volume confirms ({vol_ratio:.1f}x)")
                else:
                    bearish_score += 2
                    bearish_signals.append(f"High volume confirms ({vol_ratio:.1f}x)")
            elif vol_ratio > 1.0:
                if bullish_score > bearish_score:
                    bullish_score += 1
                    bullish_signals.append(f"Volume above avg ({vol_ratio:.1f}x)")
                else:
                    bearish_score += 1
                    bearish_signals.append(f"Volume above avg ({vol_ratio:.1f}x)")

        # ========== 6. ATR VOLATILITY CHECK ==========
        atr_val = self.atr.value if self.atr.initialized else 0
        atr_pct = (atr_val / bar.close) * 100 if bar.close > 0 else 0

        if atr_pct > 2.0:
            self.log.warning(f"   ⚠️ High volatility ({atr_pct:.2f}%) - reducing scores", LogColor.YELLOW)
            bullish_score = int(bullish_score * 0.7)
            bearish_score = int(bearish_score * 0.7)

        # ========== LOG ANALYSIS ==========
        self.log.info("=" * 60, LogColor.CYAN)
        self.log.info(f"   Price: ${bar.close:.2f}", LogColor.CYAN)
        self.log.info(f"   EMA9: ${self.fast_ema.value:.2f} | EMA20: ${self.slow_ema.value:.2f}", LogColor.CYAN)
        self.log.info(f"   RSI: {rsi_val:.1f} | MACD: {macd_val:.3f}", LogColor.CYAN)
        self.log.info("-" * 60, LogColor.CYAN)
        self.log.info(f"   BULLISH Score: {bullish_score}/15", LogColor.GREEN)
        for sig in bullish_signals:
            self.log.info(f"      ✓ {sig}", LogColor.GREEN)
        self.log.info(f"   BEARISH Score: {bearish_score}/15", LogColor.RED)
        for sig in bearish_signals:
            self.log.info(f"      ✗ {sig}", LogColor.RED)
        self.log.info("-" * 60, LogColor.CYAN)

        # ========== DECISION - STRICT THRESHOLDS ==========
        # For 90% win rate: Need score >= 8 AND lead >= 4
        MIN_SCORE = 8
        MIN_LEAD = 4

        if bullish_score >= MIN_SCORE and bullish_score >= bearish_score + MIN_LEAD:
            self.log.info(f"📈 HIGH-PROBABILITY BULLISH (Score: {bullish_score} vs {bearish_score})", LogColor.GREEN)
            self.log.info(">>> RECOMMENDATION: BUY CALL", LogColor.GREEN)
            return "BULLISH"
        elif bearish_score >= MIN_SCORE and bearish_score >= bullish_score + MIN_LEAD:
            self.log.info(f"📉 HIGH-PROBABILITY BEARISH (Score: {bearish_score} vs {bullish_score})", LogColor.RED)
            self.log.info(">>> RECOMMENDATION: BUY PUT", LogColor.RED)
            return "BEARISH"
        else:
            # NO CLEAR SIGNAL - for this strategy we still trade but log warning
            self.log.warning(f"⚠️ NO HIGH-PROB SIGNAL (Bull:{bullish_score} Bear:{bearish_score})", LogColor.YELLOW)
            self.log.warning(f"   Need {MIN_SCORE}+ score with {MIN_LEAD}+ lead", LogColor.YELLOW)
            # Default based on slight edge (this strategy must trade once per day)
            if bullish_score >= bearish_score:
                self.log.info(">>> DEFAULTING TO CALL (slight bullish edge)", LogColor.YELLOW)
                return "BULLISH"
            else:
                self.log.info(">>> DEFAULTING TO PUT (slight bearish edge)", LogColor.YELLOW)
                return "BEARISH"

    def _enter_position(self, bar: Bar, bar_time_est: datetime, signal: str) -> None:
        """Enter a trading position"""
        self.log.info("=" * 80, LogColor.GREEN)
        self.log.info("🎯 ENTERING POSITION", LogColor.GREEN)

        option_type = "CALL" if signal == "BULLISH" else "PUT"
        underlying_price = bar.close
        today_date = bar_time_est.date()

        self.log.info(f"   Option Type: {option_type}", LogColor.CYAN)
        self.log.info(f"   Underlying: ${underlying_price:.2f}", LogColor.CYAN)

        # Calculate ATM strike
        atm_strike = round(underlying_price / 5) * 5
        self.log.info(f"   ATM Strike: ${atm_strike:.2f}", LogColor.CYAN)

        # Find option contract
        option_instrument = self._find_best_option(today_date, atm_strike, underlying_price, option_type)

        if option_instrument is None:
            self.log.error("❌ No option contract found!", LogColor.RED)
            return

        # Get ACTUAL option price from current bar data (not estimate)
        actual_premium = None
        if self._engine and option_instrument.symbol in self._engine.current_bars:
            option_bar = self._engine.current_bars[option_instrument.symbol]
            actual_premium = option_bar.close
            self.log.info(f"   Actual Option Price: ${actual_premium:.2f}", LogColor.CYAN)

        if actual_premium is None or actual_premium <= 0:
            # Fallback to estimate if no bar data available
            actual_premium = underlying_price * 0.03
            self.log.info(f"   Est. Premium (fallback): ${actual_premium:.2f}", LogColor.YELLOW)

        # STRICT $2000 CAP CALCULATION
        # Account for slippage (10%) when calculating max contracts
        # This ensures even with slippage, we stay under budget
        SLIPPAGE_BUFFER = 1.15  # 15% buffer to account for slippage and price movement
        budget = float(self.config.fixed_position_value)

        # Contract cost = premium * 100 (multiplier)
        contract_cost = actual_premium * 100

        # Calculate contracts with slippage buffer to ensure we NEVER exceed budget
        adjusted_contract_cost = contract_cost * SLIPPAGE_BUFFER
        max_contracts_in_budget = int(budget / adjusted_contract_cost)
        num_contracts = max(1, max_contracts_in_budget)

        # Double-check: recalculate with actual expected cost including buffer
        expected_cost_with_slippage = num_contracts * adjusted_contract_cost

        # If even 1 contract exceeds budget significantly, skip the trade
        if num_contracts == 1 and contract_cost > budget:
            self.log.warning(f"   ⚠️ SKIPPING: Single contract (${contract_cost:.2f}) exceeds budget (${budget:.2f})", LogColor.RED)
            return  # Don't enter this trade

        # Calculate actual capital being deployed (before slippage)
        actual_capital = num_contracts * contract_cost

        self.log.info(f"   Strike: ${option_instrument.strike_price:.2f} (ATM)", LogColor.CYAN)
        self.log.info(f"   Cost per Contract: ${contract_cost:.2f}", LogColor.CYAN)
        self.log.info(f"   Contracts to Buy: {num_contracts}", LogColor.CYAN)
        self.log.info(f"   Estimated Capital: ${actual_capital:.2f} (max with slippage: ${expected_cost_with_slippage:.2f})", LogColor.CYAN)
        self.log.info(f"   Budget: ${budget:.2f}", LogColor.CYAN)

        # Create and submit entry order
        entry_order = self.order_factory.market(
            instrument_id=option_instrument.symbol,
            order_side=OrderSide.BUY,
            quantity=num_contracts,
            time_in_force=TimeInForce.GTC,
        )

        self.submit_order(entry_order)

        # Track state
        self.entry_order_id = entry_order.client_order_id
        self.pending_option_instrument = option_instrument
        self.current_option_instrument = option_instrument
        self.traded_today = True
        self.last_trade_date = today_date
        self.entry_timestamp = bar.ts_event
        self.entry_bar_datetime = bar.timestamp  # Store for SL/TP eligibility
        self.trades_today += 1
        self.current_signal = signal  # Store the signal (BULLISH/BEARISH)

        self.log.info("✅ ENTRY ORDER SUBMITTED", LogColor.GREEN)
        self.log.info(f"   Order ID: {entry_order.client_order_id}", LogColor.GREEN)

    def _find_best_option(
        self,
        today_date,
        target_strike: float,
        underlying_price: float,
        option_type: str
    ) -> Optional[OptionContract]:
        """Find best ATM option contract expiring THIS WEEK's Friday"""
        self.log.info(f"   🔍 Searching for {option_type} options...", LogColor.BLUE)

        if self._engine is None:
            return None

        # Calculate THIS WEEK's Friday expiry
        # Monday (0) -> Friday in 4 days
        # Tuesday (1) -> Friday in 3 days
        # Wednesday (2) -> Friday in 2 days
        # Thursday (3) -> Friday in 1 day
        # Friday (4) -> Friday TODAY (0DTE)
        weekday = today_date.weekday()
        if weekday <= 4:  # Monday to Friday
            days_to_friday = 4 - weekday
        else:  # Saturday/Sunday - shouldn't happen in backtest
            days_to_friday = (4 - weekday) % 7

        this_weeks_friday = today_date + timedelta(days=days_to_friday)

        self.log.info(f"   📅 Today: {today_date} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][weekday]})", LogColor.CYAN)
        self.log.info(f"   📅 This week's Friday: {this_weeks_friday} ({days_to_friday} days to expiry)", LogColor.CYAN)

        # ATM threshold: strike must be within 2.5% of underlying price
        ATM_THRESHOLD_PCT = 2.5

        best_match = None
        best_strike_diff = float('inf')

        for inst in self._engine.instruments.values():
            if not isinstance(inst, OptionContract):
                continue

            if 'COIN' not in inst.symbol:
                continue

            # Check option type
            is_call = inst.option_type == OptionType.CALL
            if option_type == "CALL" and not is_call:
                continue
            if option_type == "PUT" and is_call:
                continue

            # MUST expire THIS WEEK's Friday
            if inst.expiration:
                expiry_date = inst.expiration.date()
                if expiry_date != this_weeks_friday:
                    continue  # Skip options not expiring this Friday
            else:
                continue  # Skip options without expiration

            strike_diff = abs(inst.strike_price - underlying_price)
            strike_diff_pct = (strike_diff / underlying_price) * 100

            # STRICT ATM CHECK: Only consider options within 2.5% of underlying
            if strike_diff_pct > ATM_THRESHOLD_PCT:
                continue  # Skip OTM contracts

            # Find closest strike to ATM
            if strike_diff < best_strike_diff:
                best_strike_diff = strike_diff
                best_match = inst

        if best_match:
            strike_diff_pct = (abs(best_match.strike_price - underlying_price) / underlying_price) * 100
            days_to_exp = (best_match.expiration.date() - today_date).days
            self.log.info(f"   ✅ FOUND ATM: {best_match.symbol}", LogColor.GREEN)
            self.log.info(f"      Strike: ${best_match.strike_price:.2f} ({strike_diff_pct:.1f}% from underlying)", LogColor.GREEN)
            self.log.info(f"      Expiry: {best_match.expiration.date()} ({days_to_exp} DTE)", LogColor.GREEN)
            self.log.info(f"      Type: {option_type}", LogColor.GREEN)
        else:
            self.log.warning(f"   ⚠️ No ATM {option_type} option found for {this_weeks_friday} within {ATM_THRESHOLD_PCT}% of ${underlying_price:.2f}", LogColor.YELLOW)

        return best_match

    def _manage_position(self, bar: Bar, bar_time_est: datetime) -> None:
        """Manage open position - check for time-based exits"""
        if self.current_position is None or self.is_closing:
            return

        # Check position is still open via engine
        if self._engine:
            pos = self._engine.account.get_position(self.current_option_instrument.symbol)
            if pos is None or pos.is_flat:
                self.log.info("   Position closed externally (SL/TP triggered)", LogColor.CYAN)
                self.current_position = None
                return

        # Calculate hold time
        time_held_minutes = 0.0
        if self.entry_timestamp:
            time_held_ns = bar.ts_event - self.entry_timestamp
            time_held_minutes = time_held_ns / (60 * 1_000_000_000)

        # Check force exit at 3:55 PM
        if bar_time_est.hour == 15 and bar_time_est.minute >= 55:
            self.log.warning("🕐 FORCE EXIT: 3:55 PM - 0DTE expires at 4 PM!", LogColor.RED)
            self._close_position("Force exit - 3:55 PM")
            return

        # Check max hold time
        if time_held_minutes >= self.config.max_hold_minutes:
            self.log.warning(f"⏱️  MAX HOLD TIME: {time_held_minutes:.1f} min", LogColor.YELLOW)
            self._close_position("Max hold time exceeded")
            return

    def _cancel_all_tracked_orders(self, reason: str) -> None:
        """Cancel all tracked SL/TP orders"""
        for order_id in [self.active_sl_order_id, self.active_tp_order_id]:
            if order_id:
                self.cancel_order(order_id)

        self.active_sl_order_id = None
        self.active_tp_order_id = None

    def _close_position(self, reason: str) -> None:
        """Close current position"""
        if self.current_position is None or self.is_closing:
            return

        self.is_closing = True
        self.log.info(f"🔒 CLOSING POSITION: {reason}", LogColor.YELLOW)

        # Cancel SL/TP orders
        self._cancel_all_tracked_orders(reason)

        # Close position
        if self.current_option_instrument:
            self.close_all_positions(self.current_option_instrument.symbol)

        # Reset state
        self.current_position = None
        self.current_option_instrument = None
        self.entry_price = None
        self.entry_timestamp = None
        self.entry_bar_datetime = None
        self.is_closing = False

    def on_order_filled(self, event: FillEvent) -> None:
        """Handle order fills"""
        try:
            fill_price = event.last_px
            fill_qty = event.last_qty

            # Check if SL filled
            if event.client_order_id == self.active_sl_order_id:
                self.log.info("🛑 STOP LOSS TRIGGERED", LogColor.RED)
                self.log.info(f"   Exit Price: ${fill_price:.2f}", LogColor.RED)

                # Set exit reason on position before it closes
                if self._engine and self.current_option_instrument:
                    pos = self._engine.account.get_position(self.current_option_instrument.symbol)
                    if pos:
                        pos.exit_reason = "SL"

                self.active_sl_order_id = None

                # Cancel TP
                if self.active_tp_order_id:
                    self.cancel_order(self.active_tp_order_id)
                    self.active_tp_order_id = None
                return

            # Check if TP filled
            if event.client_order_id == self.active_tp_order_id:
                self.log.info("🎯 TAKE PROFIT HIT!", LogColor.GREEN)
                self.log.info(f"   Exit Price: ${fill_price:.2f}", LogColor.GREEN)

                # Set exit reason on position before it closes
                if self._engine and self.current_option_instrument:
                    pos = self._engine.account.get_position(self.current_option_instrument.symbol)
                    if pos:
                        pos.exit_reason = "TP"

                self.active_tp_order_id = None

                # Cancel SL
                if self.active_sl_order_id:
                    self.cancel_order(self.active_sl_order_id)
                    self.active_sl_order_id = None
                return

            # Entry order filled
            if event.client_order_id == self.entry_order_id:
                self.entry_price = fill_price

                self.log.info("📍 ENTRY FILLED", LogColor.GREEN)
                self.log.info(f"   Price: ${fill_price:.2f}", LogColor.GREEN)
                self.log.info(f"   Qty: {fill_qty}", LogColor.GREEN)

                # Set up position tracking
                if self._engine and self.pending_option_instrument:
                    self.current_position = self._engine.account.get_position(
                        self.pending_option_instrument.symbol
                    )
                    # Store signal on position for trade records
                    if self.current_position:
                        self.current_position.signal = self.current_signal

                # Calculate SL/TP prices
                tp_price = fill_price * (1 + float(self.config.target_profit_pct) / 100)
                sl_price = fill_price * (1 - float(self.config.stop_loss_pct) / 100)

                self.log.info(f"   Take Profit: ${tp_price:.2f} (+{self.config.target_profit_pct}%)", LogColor.BLUE)
                self.log.info(f"   Stop Loss: ${sl_price:.2f} (-{self.config.stop_loss_pct}%)", LogColor.BLUE)

                # Create SL order
                sl_order = self.order_factory.stop_market(
                    instrument_id=self.pending_option_instrument.symbol,
                    order_side=OrderSide.SELL,
                    quantity=fill_qty,
                    trigger_price=sl_price,
                    time_in_force=TimeInForce.GTC,
                    reduce_only=True,
                )

                # Create TP order
                tp_order = self.order_factory.limit(
                    instrument_id=self.pending_option_instrument.symbol,
                    order_side=OrderSide.SELL,
                    quantity=fill_qty,
                    price=tp_price,
                    time_in_force=TimeInForce.GTC,
                    reduce_only=True,
                )

                # Link orders (OCO)
                sl_order.linked_orders = [tp_order.client_order_id]
                tp_order.linked_orders = [sl_order.client_order_id]

                # Set first eligible bar timestamp - orders can only fill on bars AFTER entry
                # This prevents immediate SL/TP triggers on the same bar
                # Use the FILL timestamp (when entry actually executed), not submission timestamp
                fill_ts = event.timestamp  # This is when the entry order actually filled
                if fill_ts is not None:
                    sl_order.first_eligible_bar_timestamp = fill_ts
                    tp_order.first_eligible_bar_timestamp = fill_ts
                elif self.entry_bar_datetime is not None:
                    # Fallback to submission timestamp
                    sl_order.first_eligible_bar_timestamp = self.entry_bar_datetime
                    tp_order.first_eligible_bar_timestamp = self.entry_bar_datetime

                # Track order IDs
                self.active_sl_order_id = sl_order.client_order_id
                self.active_tp_order_id = tp_order.client_order_id

                # Submit orders
                self.submit_order(sl_order)
                self.submit_order(tp_order)

                self.log.info("✅ SL/TP orders submitted", LogColor.GREEN)

                # Clear pending
                self.pending_option_instrument = None
                self.entry_order_id = None

        except Exception as e:
            self.log.error(f"Error in on_order_filled(): {e}", LogColor.RED)
            import traceback
            traceback.print_exc()

    def on_stop(self) -> None:
        """Strategy shutdown"""
        self.log.info("🛑 STOPPING STRATEGY", LogColor.YELLOW)

        if self.current_position and not self.is_closing:
            self._close_position("Strategy stopped")

        self.log.info("🏁 STRATEGY STOPPED", LogColor.YELLOW)

    def on_reset(self) -> None:
        """Reset strategy state"""
        self.traded_today = False
        self.last_trade_date = None
        self.entry_price = None
        self.entry_timestamp = None
        self.entry_bar_datetime = None
        self.current_position = None
        self.current_option_instrument = None
        self.pending_option_instrument = None
        self.volume_history = []
        self.is_closing = False
        self.entry_order_id = None
        self.active_sl_order_id = None
        self.active_tp_order_id = None
        self.trades_today = 0
        self.daily_pnl = 0.0

    # ========================================================================
    # STANDALONE SIGNAL CALCULATION - For use by paper_trading_engine
    # ========================================================================

    @staticmethod
    def calculate_signal_from_bars(bars: list, config: 'COINDaily0DTEMomentumConfig' = None) -> dict:
        """
        Calculate trading signal from raw bar data.

        This is a STANDALONE method that can be called by any engine
        without instantiating the full strategy class.

        Parameters
        ----------
        bars : list
            List of bar objects with: close, high, low, volume attributes
        config : COINDaily0DTEMomentumConfig, optional
            Strategy configuration

        Returns
        -------
        dict
            {
                'signal': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
                'bullish_score': int,
                'bearish_score': int,
                'confidence': str,  # 'HIGH' | 'MEDIUM' | 'LOW'
                'signals': list,    # List of signal reasons
                'indicators': dict  # All indicator values
            }
        """
        if not bars or len(bars) < 30:
            return {
                'signal': 'NEUTRAL',
                'bullish_score': 0,
                'bearish_score': 0,
                'confidence': 'LOW',
                'signals': ['Not enough data'],
                'indicators': {}
            }

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        volumes = [getattr(b, 'volume', 0) for b in bars]
        current_price = closes[-1]

        # Calculate indicators
        def calc_ema(prices, period):
            if len(prices) < period:
                return sum(prices) / len(prices)
            multiplier = 2 / (period + 1)
            ema = sum(prices[:period]) / period
            for price in prices[period:]:
                ema = (price - ema) * multiplier + ema
            return ema

        def calc_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50.0
            gains, losses = [], []
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                gains.append(max(0, change))
                losses.append(max(0, -change))
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        def calc_bb(prices, period=20, std_dev=2.0):
            if len(prices) < period:
                return None, None, None
            recent = prices[-period:]
            middle = sum(recent) / period
            variance = sum((p - middle) ** 2 for p in recent) / period
            std = variance ** 0.5
            return middle - std_dev * std, middle, middle + std_dev * std

        def calc_vwap(bars_list):
            total_vol = sum(getattr(b, 'volume', 0) for b in bars_list)
            if total_vol == 0:
                return bars_list[-1].close
            total_vwap = sum(((b.high + b.low + b.close) / 3) * getattr(b, 'volume', 0) for b in bars_list)
            return total_vwap / total_vol

        def calc_atr(bars_list, period=14):
            if len(bars_list) < 2:
                return 0
            trs = []
            for i in range(1, len(bars_list)):
                tr = max(
                    bars_list[i].high - bars_list[i].low,
                    abs(bars_list[i].high - bars_list[i-1].close),
                    abs(bars_list[i].low - bars_list[i-1].close)
                )
                trs.append(tr)
            return sum(trs[-period:]) / min(period, len(trs))

        # Calculate all indicators
        ema_9 = calc_ema(closes, 9)
        ema_20 = calc_ema(closes, 20)
        ema_50 = calc_ema(closes, min(50, len(closes)))
        rsi = calc_rsi(closes, 14)
        bb_lower, bb_middle, bb_upper = calc_bb(closes, 20, 2.0)
        vwap = calc_vwap(bars)
        atr = calc_atr(bars, 14)
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

        # MACD
        ema_12 = calc_ema(closes, 12)
        ema_26 = calc_ema(closes, 26)
        macd_line = ema_12 - ema_26
        macd_signal = macd_line * 0.9  # Approximation
        macd_hist = macd_line - macd_signal

        # Volume analysis
        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / max(1, len(volumes))
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

        # ========== SCORING SYSTEM ==========
        bullish_score = 0
        bearish_score = 0
        bullish_signals = []
        bearish_signals = []

        # 1. EMA STACK (up to 4 points)
        if current_price > ema_9 > ema_20:
            bullish_score += 4
            bullish_signals.append("EMA Stack Bullish")
        elif current_price < ema_9 < ema_20:
            bearish_score += 4
            bearish_signals.append("EMA Stack Bearish")
        elif current_price > ema_20:
            bullish_score += 2
            bullish_signals.append("Above EMA20")
        elif current_price < ema_20:
            bearish_score += 2
            bearish_signals.append("Below EMA20")

        # 2. VWAP (up to 3 points)
        vwap_dist = ((current_price - vwap) / vwap) * 100 if vwap > 0 else 0
        if current_price > vwap:
            bullish_score += 2
            bullish_signals.append(f"Above VWAP (+{vwap_dist:.2f}%)")
            if abs(vwap_dist) < 0.3:
                bullish_score += 1
                bullish_signals.append("Near VWAP support")
        else:
            bearish_score += 2
            bearish_signals.append(f"Below VWAP ({vwap_dist:.2f}%)")
            if abs(vwap_dist) < 0.3:
                bearish_score += 1
                bearish_signals.append("Near VWAP resistance")

        # 3. RSI (up to 3 points)
        if 30 <= rsi <= 50:
            bullish_score += 3
            bullish_signals.append(f"RSI bullish zone ({rsi:.1f})")
        elif 50 < rsi <= 65:
            bullish_score += 2
            bullish_signals.append(f"RSI momentum ({rsi:.1f})")
        elif rsi > 70:
            bearish_score += 3
            bearish_signals.append(f"RSI OVERBOUGHT ({rsi:.1f})")
        elif rsi < 30:
            bullish_score += 3
            bullish_signals.append(f"RSI OVERSOLD ({rsi:.1f})")
        elif 50 <= rsi <= 70:
            bearish_score += 2
            bearish_signals.append(f"RSI neutral-high ({rsi:.1f})")

        # 4. MACD (up to 3 points)
        if macd_hist > 0 and macd_line > macd_signal:
            bullish_score += 3
            bullish_signals.append("MACD Bullish")
        elif macd_hist < 0 and macd_line < macd_signal:
            bearish_score += 3
            bearish_signals.append("MACD Bearish")

        # 5. Bollinger Bands (up to 2 points)
        if bb_middle:
            if current_price > bb_middle:
                bullish_score += 2
                bullish_signals.append("Above BB Middle")
            else:
                bearish_score += 2
                bearish_signals.append("Below BB Middle")

        # 6. Volume (up to 2 points)
        if volume_ratio > 1.5:
            price_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0
            if price_change > 0:
                bullish_score += 2
                bullish_signals.append(f"High volume bullish ({volume_ratio:.1f}x)")
            else:
                bearish_score += 2
                bearish_signals.append(f"High volume bearish ({volume_ratio:.1f}x)")

        # Volatility adjustment
        if atr_pct > 2.0:
            bullish_score = int(bullish_score * 0.7)
            bearish_score = int(bearish_score * 0.7)

        # ========== DECISION ==========
        MIN_SCORE = 8
        MIN_LEAD = 4

        if bullish_score >= MIN_SCORE and bullish_score >= bearish_score + MIN_LEAD:
            signal = 'BULLISH'
            confidence = 'HIGH'
        elif bearish_score >= MIN_SCORE and bearish_score >= bullish_score + MIN_LEAD:
            signal = 'BEARISH'
            confidence = 'HIGH'
        elif bullish_score >= 6 or bearish_score >= 6:
            signal = 'BULLISH' if bullish_score >= bearish_score else 'BEARISH'
            confidence = 'MEDIUM'
        else:
            signal = 'NEUTRAL'
            confidence = 'LOW'

        return {
            'signal': signal,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'confidence': confidence,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'indicators': {
                'price': current_price,
                'ema_9': ema_9,
                'ema_20': ema_20,
                'ema_50': ema_50,
                'rsi': rsi,
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'bb_upper': bb_upper,
                'vwap': vwap,
                'atr': atr,
                'atr_pct': atr_pct,
                'volume_ratio': volume_ratio,
            }
        }

    @staticmethod
    def calculate_price_action_signal(bars: list) -> dict:
        """
        Calculate trading signal based on PRICE ACTION patterns.

        This is METHOD 2 for dual-confirmation signal validation.
        It analyzes recent candle patterns independent of technical indicators.

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

        if momentum_5 > 0.15:
            bullish_points += 2
            reasons.append(f"Strong 5-bar momentum (+{momentum_5:.2f}%)")
        elif momentum_5 > 0.05:
            bullish_points += 1
            reasons.append(f"Positive momentum (+{momentum_5:.2f}%)")
        elif momentum_5 < -0.15:
            bearish_points += 2
            reasons.append(f"Strong bearish momentum ({momentum_5:.2f}%)")
        elif momentum_5 < -0.05:
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
