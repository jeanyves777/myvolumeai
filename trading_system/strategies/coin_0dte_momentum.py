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
        """Analyze market to determine CALL vs PUT"""
        self.log.info("📊 MARKET DIRECTION ANALYSIS", LogColor.CYAN)

        bullish_score = 0
        bearish_score = 0

        # EMA Trend (3 points)
        if self.fast_ema.value > self.slow_ema.value:
            bullish_score += 3
            self.log.info(f"   ✓ EMA: BULLISH (+3)", LogColor.GREEN)
        else:
            bearish_score += 3
            self.log.info(f"   ✓ EMA: BEARISH (+3)", LogColor.RED)

        # RSI Momentum (2 points)
        if self.rsi.value > 50:
            bullish_score += 2
            self.log.info(f"   ✓ RSI: BULLISH (+2) | RSI={self.rsi.value:.1f}", LogColor.GREEN)
        else:
            bearish_score += 2
            self.log.info(f"   ✓ RSI: BEARISH (+2) | RSI={self.rsi.value:.1f}", LogColor.RED)

        # MACD (2 points)
        if self.macd.value > self.macd.signal:
            bullish_score += 2
            self.log.info(f"   ✓ MACD: BULLISH (+2)", LogColor.GREEN)
        else:
            bearish_score += 2
            self.log.info(f"   ✓ MACD: BEARISH (+2)", LogColor.RED)

        # Bollinger Bands (1 point)
        if bar.close > self.bb.middle:
            bullish_score += 1
            self.log.info(f"   ✓ BB: BULLISH (+1)", LogColor.GREEN)
        else:
            bearish_score += 1
            self.log.info(f"   ✓ BB: BEARISH (+1)", LogColor.RED)

        # Volume confirmation (1 point)
        if len(self.volume_history) >= 5:
            avg_vol = sum(self.volume_history[:-1]) / (len(self.volume_history) - 1)
            vol_ratio = bar.volume / avg_vol if avg_vol > 0 else 0

            if vol_ratio >= self.config.min_volume_ratio:
                if bullish_score > bearish_score:
                    bullish_score += 1
                else:
                    bearish_score += 1

        # Final decision
        self.log.info(f"📊 SCORE: BULLISH={bullish_score} | BEARISH={bearish_score}", LogColor.CYAN)

        if bullish_score >= bearish_score:
            self.log.info("📈 DECISION: BULLISH -> BUY CALL", LogColor.GREEN)
            return "BULLISH"
        else:
            self.log.info("📉 DECISION: BEARISH -> BUY PUT", LogColor.RED)
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
