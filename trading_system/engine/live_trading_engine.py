"""
Live Trading Engine

REAL MONEY trading execution engine with:
- Multiple safety checks and confirmations
- Daily loss limits
- Trade logging
- Position monitoring
"""

import asyncio
import signal
import sys
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import pytz
import threading

from ..config.live_trading_config import (
    LiveTradingConfig,
    log_trade,
    mask_api_key,
)
from .alpaca_client import AlpacaClient, Quote, Bar, ALPACA_AVAILABLE
from ..strategies import COINDaily0DTEMomentum, COINDaily0DTEMomentumConfig
from ..analytics.options_trade_logger import OptionsTradeLogger, get_options_trade_logger
from ..data.options_market_collector import OptionsMarketCollector, get_options_market_collector


EST = pytz.timezone('America/New_York')


@dataclass
class LivePosition:
    """Tracks an open live trading position."""
    symbol: str  # OCC option symbol
    underlying: str
    qty: int
    side: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    option_type: str  # 'CALL' or 'PUT'
    strike: float
    expiration: datetime
    signal: str  # 'BULLISH' or 'BEARISH'

    # SL/TP tracking
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0

    # Order IDs
    entry_order_id: str = ""
    sl_order_id: str = ""
    tp_order_id: str = ""


@dataclass
class LiveTradingSession:
    """Tracks current live trading session state."""
    date: datetime = field(default_factory=lambda: datetime.now(EST).date())
    trades_today: int = 0
    pnl_today: float = 0.0
    position: Optional[LivePosition] = None
    has_traded_today: bool = False
    stopped_by_limit: bool = False
    stop_reason: str = ""


class LiveTradingEngine:
    """
    LIVE trading execution engine with REAL MONEY.

    Includes multiple safety checks:
    - Daily loss limits
    - Max trades per day
    - Position size limits
    - Optional trade confirmation
    - Complete trade logging
    """

    def __init__(self, config: LiveTradingConfig):
        """
        Initialize live trading engine.

        Args:
            config: Live trading configuration
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py package required. Run: pip install alpaca-py")

        self.config = config

        # IMPORTANT: use_paper MUST be False for live trading
        if config.use_paper:
            raise ValueError("Live trading config has use_paper=True. This is incorrect.")

        self.client = AlpacaClient(
            api_key=config.api_key,
            api_secret=config.api_secret,
            paper=False  # LIVE TRADING
        )

        # Strategy config (for max_trades_per_day and other strategy-specific settings)
        self.strategy_config = COINDaily0DTEMomentumConfig(
            underlying_symbol=config.underlying_symbol,
            fixed_position_value=config.fixed_position_value,
            target_profit_pct=config.target_profit_pct,
            stop_loss_pct=config.stop_loss_pct,
            max_hold_minutes=config.max_hold_minutes,
            entry_time_start=config.entry_time_start,
            entry_time_end=config.entry_time_end,
            force_exit_time=config.force_exit_time,
        )

        # Get max_trades_per_day from strategy (if config value is 0, use strategy default)
        self.max_trades_per_day = (
            config.max_trades_per_day if config.max_trades_per_day > 0
            else self.strategy_config.max_trades_per_day
        )

        # Session state
        self.session = LiveTradingSession()
        self.running = False
        self._stop_event = threading.Event()

        # Market data
        self.latest_underlying_quote: Optional[Quote] = None
        self.latest_underlying_bar: Optional[Bar] = None
        self.latest_option_quote: Optional[Quote] = None

        # Parse trading times
        self.entry_start = datetime.strptime(config.entry_time_start, "%H:%M:%S").time()
        self.entry_end = datetime.strptime(config.entry_time_end, "%H:%M:%S").time()
        self.force_exit = datetime.strptime(config.force_exit_time, "%H:%M:%S").time()

        # Trade logging and market data collection
        self.trade_logger: OptionsTradeLogger = get_options_trade_logger()
        self.market_collector: OptionsMarketCollector = get_options_market_collector()

    def _log(self, msg: str, level: str = "INFO"):
        """Log message with timestamp."""
        now = datetime.now(EST)
        prefix = ""
        if level == "WARN":
            prefix = "[WARNING] "
        elif level == "ERROR":
            prefix = "[ERROR] "
        elif level == "TRADE":
            prefix = "[$$$ TRADE $$$] "
        elif level == "DANGER":
            prefix = "[!!! DANGER !!!] "

        print(f"[{now.strftime('%H:%M:%S')}] {prefix}{msg}")

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(EST)

        # Check if weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        # Market hours: 9:30 AM - 4:00 PM EST
        market_open = time(9, 30)
        market_close = time(16, 0)

        return market_open <= now.time() <= market_close

    def _is_entry_window(self) -> bool:
        """Check if within entry time window."""
        now = datetime.now(EST).time()
        return self.entry_start <= now <= self.entry_end

    def _should_force_exit(self) -> bool:
        """Check if should force exit position."""
        now = datetime.now(EST).time()
        return now >= self.force_exit

    def _check_safety_limits(self) -> tuple[bool, str]:
        """
        Check all safety limits before trading.

        Returns:
            Tuple of (can_trade: bool, reason: str)
        """
        # Check config limits (pass effective max_trades_per_day from strategy)
        can_trade, reason = self.config.can_trade(self.max_trades_per_day)
        if not can_trade:
            return False, reason

        # Additional session checks
        if self.session.stopped_by_limit:
            return False, self.session.stop_reason

        return True, "OK"

    def _get_this_weeks_friday(self) -> datetime:
        """Get this week's Friday for 0DTE expiry."""
        now = datetime.now(EST)
        weekday = now.weekday()

        if weekday <= 4:  # Monday to Friday
            days_to_friday = 4 - weekday
        else:  # Saturday/Sunday
            days_to_friday = (4 - weekday) % 7

        friday = now + timedelta(days=days_to_friday)
        return friday.replace(hour=16, minute=0, second=0, microsecond=0)

    def _calculate_signal(self) -> str:
        """
        Calculate trading signal using DUAL CONFIRMATION from strategy.

        METHOD 1: Technical Scoring (EMA, VWAP, RSI, MACD, BB, Volume) - 1-MIN BARS
        METHOD 2: Price Action (candle patterns, higher highs/lows, momentum) - 5-MIN BARS

        Only trades when BOTH methods agree on the same direction.

        Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        # Get 1-MINUTE bars for Technical Scoring (Method 1)
        start_time = datetime.now(EST) - timedelta(hours=3)
        bars_1min = self.client.get_stock_bars(
            self.config.underlying_symbol,
            timeframe='1Min',
            start=start_time,
            limit=180
        )

        if not bars_1min or len(bars_1min) < 30:
            bars_count = len(bars_1min) if bars_1min else 0
            self._log(f"Not enough 1-min bar data for analysis ({bars_count} bars, need 30) - SKIPPING TRADE", "WARN")
            return 'NEUTRAL'

        # Use last 60 bars for 1-min
        bars_1min = bars_1min[-60:]

        # Get 5-MINUTE bars for Price Action (Method 2)
        # Need 60 5-min bars = 5 hours of data, fetch 6 hours to be safe
        start_time_5min = datetime.now(EST) - timedelta(hours=6)
        bars_5min = self.client.get_stock_bars(
            self.config.underlying_symbol,
            timeframe='5Min',
            start=start_time_5min,
            limit=100
        )

        if not bars_5min or len(bars_5min) < 12:
            bars_count = len(bars_5min) if bars_5min else 0
            self._log(f"Not enough 5-min bar data for Price Action ({bars_count} bars, need 12) - SKIPPING TRADE", "WARN")
            return 'NEUTRAL'

        # Use last 30 bars for 5-min (2.5 hours of data)
        bars_5min = bars_5min[-30:]

        # ============================================================
        # DUAL SIGNAL VALIDATION
        # ============================================================
        self._log("=" * 65, "INFO")
        self._log("  DUAL SIGNAL VALIDATION", "INFO")
        self._log("=" * 65, "INFO")

        # ========== METHOD 1: Technical Scoring (1-MIN BARS) ==========
        tech_result = COINDaily0DTEMomentum.calculate_signal_from_bars(bars_1min)
        tech_signal = tech_result['signal']
        tech_confidence = tech_result['confidence']
        indicators = tech_result.get('indicators', {})

        self._log("  METHOD 1 - Technical Scoring (1-MIN):", "INFO")
        if indicators:
            self._log(f"    Price: ${indicators.get('price', 0):.2f} | VWAP: ${indicators.get('vwap', 0):.2f}", "INFO")
            self._log(f"    EMA9: ${indicators.get('ema_9', 0):.2f} | EMA20: ${indicators.get('ema_20', 0):.2f}", "INFO")
            self._log(f"    RSI: {indicators.get('rsi', 0):.1f} | MACD: {indicators.get('macd_line', 0):.3f}", "INFO")

        self._log(f"    Signal: {tech_signal} | Score: {tech_result['bullish_score']}/17 vs {tech_result['bearish_score']}/17 | Confidence: {tech_confidence}", "INFO")

        # Log bullish signals
        for sig in tech_result.get('bullish_signals', []):
            self._log(f"      + {sig}", "TRADE")
        # Log bearish signals
        for sig in tech_result.get('bearish_signals', []):
            self._log(f"      - {sig}", "WARN")

        self._log("-" * 65, "INFO")

        # ========== METHOD 2: Price Action (5-MIN BARS) ==========
        pa_result = COINDaily0DTEMomentum.calculate_price_action_signal(bars_5min)
        pa_signal = pa_result['signal']
        pa_strength = pa_result['strength']

        self._log("  METHOD 2 - Price Action (5-MIN):", "INFO")
        self._log(f"    Signal: {pa_signal} | Strength: {pa_strength} | Points: {pa_result['bullish_points']} bull vs {pa_result['bearish_points']} bear", "INFO")

        for reason in pa_result.get('reasons', []):
            if 'bullish' in reason.lower() or 'green' in reason.lower() or 'higher' in reason.lower() or 'above' in reason.lower() or 'uptrend' in reason.lower() or '+' in reason:
                self._log(f"      + {reason}", "TRADE")
            elif 'bearish' in reason.lower() or 'red' in reason.lower() or 'lower' in reason.lower() or 'below' in reason.lower() or 'downtrend' in reason.lower():
                self._log(f"      - {reason}", "WARN")
            else:
                self._log(f"      * {reason}", "INFO")

        self._log("-" * 65, "INFO")

        # ========== FINAL DECISION ==========
        self._log("  FINAL DECISION:", "INFO")

        # Both must agree for confirmed signal
        if tech_signal == 'BULLISH' and pa_signal == 'BULLISH':
            final_signal = 'BULLISH'
            self._log(f"    CONFIRMED BULLISH - BOTH METHODS AGREE", "TRADE")
            self._log(f"    Technical: {tech_signal} ({tech_confidence}) | Price Action: {pa_signal} ({pa_strength})", "TRADE")
            self._log(f"    >>> EXECUTING: BUY CALLS", "TRADE")

        elif tech_signal == 'BEARISH' and pa_signal == 'BEARISH':
            final_signal = 'BEARISH'
            self._log(f"    CONFIRMED BEARISH - BOTH METHODS AGREE", "DANGER")
            self._log(f"    Technical: {tech_signal} ({tech_confidence}) | Price Action: {pa_signal} ({pa_strength})", "DANGER")
            self._log(f"    >>> EXECUTING: BUY PUTS", "DANGER")

        elif tech_signal == 'NEUTRAL' or pa_signal == 'NEUTRAL':
            final_signal = 'NEUTRAL'
            self._log(f"    NO TRADE - One or both methods neutral", "INFO")
            self._log(f"    Technical: {tech_signal} | Price Action: {pa_signal}", "INFO")
            self._log(f"    >>> SKIPPING TRADE", "INFO")

        else:
            # Conflicting signals
            final_signal = 'NEUTRAL'
            self._log(f"    CONFLICTING SIGNALS - NO TRADE", "WARN")
            self._log(f"    Technical: {tech_signal} ({tech_confidence}) | Price Action: {pa_signal} ({pa_strength})", "WARN")
            self._log(f"    >>> SKIPPING TRADE (signals disagree)", "WARN")

        self._log("=" * 65, "INFO")

        return final_signal

    def _find_atm_option(self, option_type: str) -> Optional[str]:
        """Find ATM option contract for the underlying."""
        if not self.latest_underlying_quote:
            return None

        underlying_price = self.latest_underlying_quote.mid
        expiry = self._get_this_weeks_friday()

        # Round to nearest $5 strike for ATM
        strike = round(underlying_price / 5) * 5

        occ_symbol = self.client.format_occ_symbol(
            underlying=self.config.underlying_symbol,
            expiration=expiry,
            strike=strike,
            option_type=option_type
        )

        return occ_symbol

    def _calculate_position_size(self, option_price: float) -> int:
        """Calculate number of contracts based on fixed position value."""
        if option_price <= 0:
            return 0

        contract_value = option_price * 100
        contracts = int(self.config.fixed_position_value / contract_value)

        return max(1, contracts)

    def _confirm_trade(self, action: str, symbol: str, qty: int, price: float) -> bool:
        """
        Get user confirmation for a trade.

        Returns True if confirmed, False otherwise.
        """
        if not self.config.require_confirmation:
            return True

        print("\n" + "=" * 50)
        print("  TRADE CONFIRMATION REQUIRED")
        print("=" * 50)
        print(f"  Action: {action}")
        print(f"  Symbol: {symbol}")
        print(f"  Quantity: {qty} contracts")
        print(f"  Price: ${price:.2f}")
        print(f"  Total Value: ${price * qty * 100:.2f}")
        print("=" * 50)

        try:
            response = input("  Execute this trade? (yes/no): ").strip().lower()
            return response in ['yes', 'y']
        except (EOFError, KeyboardInterrupt):
            return False

    def _enter_position(self, signal: str):
        """Enter a new position based on signal."""
        if self.session.position is not None:
            self._log("Already in position, skipping entry", "WARN")
            return

        if self.session.has_traded_today:
            self._log("Already traded today, skipping entry", "INFO")
            return

        # Check safety limits
        can_trade, reason = self._check_safety_limits()
        if not can_trade:
            self._log(f"Cannot trade: {reason}", "WARN")
            self.session.stopped_by_limit = True
            self.session.stop_reason = reason
            return

        option_type = 'C' if signal == 'BULLISH' else 'P'
        occ_symbol = self._find_atm_option(option_type)

        if not occ_symbol:
            self._log("Could not find ATM option", "ERROR")
            return

        # Get option quote
        option_quote = self.client.get_latest_option_quote(occ_symbol)
        if not option_quote:
            self._log(f"Could not get quote for {occ_symbol}", "ERROR")
            return

        # Calculate position size
        qty = self._calculate_position_size(option_quote.ask)

        if qty == 0:
            self._log("Position size would be 0, skipping", "WARN")
            return

        # Check position value against max
        position_value = option_quote.ask * qty * 100
        if position_value > self.config.max_position_value:
            self._log(f"Position value ${position_value:.2f} exceeds max ${self.config.max_position_value:.2f}", "WARN")
            return

        # CONFIRMATION REQUIRED FOR LIVE TRADING
        self._log(f"LIVE TRADE SIGNAL: {signal} - {qty}x {occ_symbol} @ ${option_quote.ask:.2f}", "TRADE")

        if not self._confirm_trade("BUY", occ_symbol, qty, option_quote.ask):
            self._log("Trade cancelled by user", "WARN")
            return

        # EXECUTE LIVE TRADE
        self._log(f"EXECUTING LIVE TRADE: BUY {qty}x {occ_symbol}", "DANGER")

        try:
            order = self.client.submit_market_order(
                symbol=occ_symbol,
                qty=qty,
                side='buy'
            )

            option_details = self.client.parse_occ_symbol(occ_symbol)

            self.session.position = LivePosition(
                symbol=occ_symbol,
                underlying=self.config.underlying_symbol,
                qty=qty,
                side='long',
                entry_price=option_quote.ask,
                entry_time=datetime.now(EST),
                option_type='CALL' if option_type == 'C' else 'PUT',
                strike=option_details['strike'],
                expiration=option_details['expiration'],
                signal=signal,
                entry_order_id=order['id'],
            )

            # Calculate SL/TP prices
            self.session.position.stop_loss_price = (
                self.session.position.entry_price *
                (1 - self.config.stop_loss_pct / 100)
            )
            self.session.position.take_profit_price = (
                self.session.position.entry_price *
                (1 + self.config.target_profit_pct / 100)
            )

            self._log(f"LIVE POSITION OPENED: Entry=${self.session.position.entry_price:.2f}, "
                     f"TP=${self.session.position.take_profit_price:.2f}, "
                     f"SL=${self.session.position.stop_loss_price:.2f}", "TRADE")

            # Place TAKE PROFIT limit order on the exchange
            # NOTE: Alpaca only supports MARKET and LIMIT orders for options (no STOP orders)
            # So we place TP as limit order, and monitor SL internally
            self._log(f"    Submitting TAKE PROFIT limit order to Alpaca...", "TRADE")
            tp_placed = False
            max_retries = 3
            tp_price = self.session.position.take_profit_price

            for attempt in range(1, max_retries + 1):
                try:
                    tp_order = self.client.submit_option_limit_order(
                        symbol=occ_symbol,
                        qty=qty,
                        side='sell',
                        limit_price=round(tp_price, 2),
                    )
                    tp_order_id = tp_order.get('id', 'N/A')
                    tp_order_status = tp_order.get('status', 'unknown')
                    self.session.position.tp_order_id = tp_order_id
                    tp_placed = True
                    self._log(f"    TP limit order placed: {tp_order_id} @ ${tp_price:.2f} (status: {tp_order_status})", "TRADE")
                    break
                except Exception as tp_err:
                    self._log(f"    TP order attempt {attempt}/{max_retries} failed: {tp_err}", "WARN")
                    if attempt < max_retries:
                        import time
                        time.sleep(1)

            if not tp_placed:
                self._log("    WARNING: TP limit order failed - will monitor TP internally", "WARN")

            self._log(f"    SL will be monitored internally @ ${self.session.position.stop_loss_price:.2f}", "TRADE")

            self.session.has_traded_today = True
            self.session.trades_today += 1

            # Log trade (original simple logging)
            log_trade({
                'type': 'ENTRY',
                'symbol': occ_symbol,
                'underlying': self.config.underlying_symbol,
                'signal': signal,
                'option_type': 'CALL' if option_type == 'C' else 'PUT',
                'qty': qty,
                'entry_price': option_quote.ask,
                'position_value': position_value,
                'order_id': order['id'],
                'timestamp': datetime.now(EST).isoformat(),
            })

            # Log trade entry to comprehensive logger
            try:
                trade_id = f"LIVE_{occ_symbol}_{datetime.now(EST).strftime('%Y%m%d_%H%M%S')}"
                expiry_str = self.session.position.expiration.strftime("%Y-%m-%d") if isinstance(self.session.position.expiration, datetime) else str(self.session.position.expiration)[:10]

                self.trade_logger.log_entry(
                    trade_id=trade_id,
                    underlying_symbol=self.config.underlying_symbol,
                    option_symbol=occ_symbol,
                    option_type='call' if option_type == 'C' else 'put',
                    strike_price=self.session.position.strike,
                    expiration_date=expiry_str,
                    entry_time=datetime.now(EST),
                    entry_price=option_quote.ask,
                    entry_qty=qty,
                    entry_order_id=order['id'],
                    entry_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0.0,
                    target_profit_pct=self.config.target_profit_pct,
                    stop_loss_pct=self.config.stop_loss_pct,
                    notes="LIVE TRADING"
                )
                # Store trade_id for exit logging
                self.session.position.entry_order_id = trade_id
            except Exception as log_err:
                self._log(f"Error logging trade entry: {log_err}", "WARN")

        except Exception as e:
            self._log(f"ERROR EXECUTING TRADE: {e}", "ERROR")

    def _check_exit_conditions(self):
        """Check if position should be exited."""
        if self.session.position is None:
            return

        pos = self.session.position

        # FIRST: Check if TP order has filled (Alpaca limit order)
        if pos.tp_order_id:
            try:
                tp_order = self.client.get_order(pos.tp_order_id)
                if tp_order and tp_order.get('status') == 'filled':
                    # TP order filled! Log the exit and clear position
                    fill_price = float(tp_order.get('filled_avg_price', pos.take_profit_price))
                    pnl_dollars = (fill_price - pos.entry_price) * pos.qty * 100
                    self._log(f"TP LIMIT ORDER FILLED @ ${fill_price:.2f}!", "TRADE")
                    self._handle_tp_filled(fill_price, pnl_dollars, tp_order.get('id', ''))
                    return
            except Exception as e:
                self._log(f"Error checking TP order status: {e}", "WARN")

        # Get current option price
        option_quote = self.client.get_latest_option_quote(pos.symbol)
        if not option_quote:
            self._log(f"Could not get quote for position {pos.symbol}", "WARN")
            return

        current_price = option_quote.mid
        self.latest_option_quote = option_quote

        # Calculate P&L
        pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
        pnl_dollars = (current_price - pos.entry_price) * pos.qty * 100

        exit_reason = None

        # Check take profit (TP limit order may have filled on exchange)
        # NOTE: If TP order filled, this is redundant but ensures we handle it
        if current_price >= pos.take_profit_price:
            exit_reason = "TAKE_PROFIT"

        # Check stop loss (monitored internally - no stop orders for options on Alpaca)
        elif current_price <= pos.stop_loss_price:
            exit_reason = "STOP_LOSS"

        # Check force exit time (3:45 PM EST) - TIME_EXIT removed per strategy
        elif self._should_force_exit():
            exit_reason = "FORCE_EXIT"

        if exit_reason:
            self._exit_position(exit_reason, current_price, pnl_dollars)

    def _handle_tp_filled(self, fill_price: float, pnl: float, order_id: str):
        """Handle when TP limit order fills on exchange - no need to submit sell order."""
        if self.session.position is None:
            return

        pos = self.session.position
        hold_time = (datetime.now(EST) - pos.entry_time).total_seconds() / 60
        pnl_pct = ((fill_price - pos.entry_price) / pos.entry_price) * 100

        self._log("=" * 50, "TRADE")
        self._log("TAKE PROFIT FILLED (Limit Order on Exchange)", "TRADE")
        self._log("=" * 50, "TRADE")
        self._log(f"Option: {pos.symbol}", "TRADE")
        self._log(f"Entry Price: ${pos.entry_price:.2f}", "TRADE")
        self._log(f"Exit Price: ${fill_price:.2f} (TP LIMIT FILL)", "TRADE")
        self._log(f"Hold Time: {hold_time:.1f} minutes", "TRADE")
        self._log(f"P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)", "TRADE")

        # Update session and config
        self.session.pnl_today += pnl
        self.config.record_trade(pnl)

        self._log(f"Session P&L: ${self.session.pnl_today:.2f}", "TRADE")
        self._log("=" * 50, "TRADE")

        # Log trade (original simple logging)
        log_trade({
            'type': 'EXIT',
            'symbol': pos.symbol,
            'reason': 'TAKE_PROFIT',
            'qty': pos.qty,
            'entry_price': pos.entry_price,
            'exit_price': fill_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'hold_time_minutes': hold_time,
            'order_id': order_id,
            'timestamp': datetime.now(EST).isoformat(),
        })

        # Log trade exit to comprehensive logger
        try:
            self.trade_logger.log_exit(
                trade_id=pos.entry_order_id,
                exit_time=datetime.now(EST),
                exit_price=fill_price,
                exit_qty=pos.qty,
                exit_order_id=order_id,
                exit_reason="TAKE_PROFIT",
                exit_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0.0,
                notes=f"LIVE TRADING - TP Limit Order Filled - Hold time: {hold_time:.1f}m"
            )
        except Exception as log_err:
            self._log(f"Error logging trade exit: {log_err}", "WARN")

        # Check if we hit daily loss limit
        if self.session.pnl_today <= -self.config.max_daily_loss:
            self._log(f"DAILY LOSS LIMIT REACHED: ${self.session.pnl_today:.2f}", "DANGER")
            self.session.stopped_by_limit = True
            self.session.stop_reason = f"Daily loss limit (${self.config.max_daily_loss:.2f})"

        self.session.position = None

    def _exit_position(self, reason: str, exit_price: float, pnl: float):
        """Exit the current position."""
        if self.session.position is None:
            return

        pos = self.session.position

        self._log(f"LIVE EXIT SIGNAL: {reason} @ ${exit_price:.2f} (P&L: ${pnl:.2f})", "TRADE")

        # Cancel the take profit order before exiting (unless exit is due to TP filling)
        # We have TP limit order on exchange, SL is monitored internally
        if pos.tp_order_id and reason != "TAKE_PROFIT":
            self._log(f"    Cancelling take profit order {pos.tp_order_id}...", "TRADE")
            try:
                cancelled = self.client.cancel_order(pos.tp_order_id)
                if cancelled:
                    self._log(f"    TP order cancelled successfully", "TRADE")
                else:
                    self._log(f"    TP order cancel returned False (may already be filled)", "WARN")
            except Exception as cancel_err:
                self._log(f"    Error cancelling TP order: {cancel_err}", "WARN")

        # Confirmation for exits too (if enabled)
        if self.config.require_confirmation and reason not in ["STOP_LOSS", "FORCE_EXIT"]:
            if not self._confirm_trade("SELL", pos.symbol, pos.qty, exit_price):
                self._log("Exit cancelled by user", "WARN")
                return

        self._log(f"EXECUTING LIVE EXIT: SELL {pos.qty}x {pos.symbol}", "DANGER")

        try:
            order = self.client.submit_market_order(
                symbol=pos.symbol,
                qty=pos.qty,
                side='sell'
            )

            # Update session and config
            self.session.pnl_today += pnl
            self.config.record_trade(pnl)

            hold_time = (datetime.now(EST) - pos.entry_time).total_seconds() / 60

            self._log(f"LIVE POSITION CLOSED: P&L=${pnl:.2f}, Hold={hold_time:.1f}m, "
                     f"Session P&L=${self.session.pnl_today:.2f}", "TRADE")

            # Log trade (original simple logging)
            log_trade({
                'type': 'EXIT',
                'symbol': pos.symbol,
                'reason': reason,
                'qty': pos.qty,
                'entry_price': pos.entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': ((exit_price - pos.entry_price) / pos.entry_price) * 100,
                'hold_time_minutes': hold_time,
                'order_id': order['id'],
                'timestamp': datetime.now(EST).isoformat(),
            })

            # Log trade exit to comprehensive logger
            try:
                self.trade_logger.log_exit(
                    trade_id=pos.entry_order_id,  # We stored trade_id here during entry
                    exit_time=datetime.now(EST),
                    exit_price=exit_price,
                    exit_qty=pos.qty,
                    exit_order_id=order['id'],
                    exit_reason=reason,
                    exit_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0.0,
                    notes=f"LIVE TRADING - Hold time: {hold_time:.1f}m"
                )
            except Exception as log_err:
                self._log(f"Error logging trade exit: {log_err}", "WARN")

            # Check if we hit daily loss limit
            if self.session.pnl_today <= -self.config.max_daily_loss:
                self._log(f"DAILY LOSS LIMIT REACHED: ${self.session.pnl_today:.2f}", "DANGER")
                self.session.stopped_by_limit = True
                self.session.stop_reason = f"Daily loss limit (${self.config.max_daily_loss:.2f})"

            self.session.position = None

        except Exception as e:
            self._log(f"ERROR EXITING POSITION: {e}", "ERROR")

    def _update_status(self):
        """Print current status."""
        now = datetime.now(EST)

        status = f"[LIVE] Time: {now.strftime('%H:%M:%S')} | "
        status += f"Market: {'OPEN' if self._is_market_open() else 'CLOSED'} | "

        if self.latest_underlying_quote:
            status += f"{self.config.underlying_symbol}: ${self.latest_underlying_quote.mid:.2f} | "

        if self.session.position:
            if self.latest_option_quote:
                current = self.latest_option_quote.mid
                entry = self.session.position.entry_price
                pnl_pct = ((current - entry) / entry) * 100
                status += f"POS: {self.session.position.option_type} @ ${current:.2f} ({pnl_pct:+.1f}%)"
            else:
                status += f"POS: {self.session.position.option_type}"
        else:
            status += "POS: None"

        status += f" | Day P&L: ${self.session.pnl_today:.2f}"

        if self.session.stopped_by_limit:
            status += f" | STOPPED: {self.session.stop_reason}"

        print(f"\r{status}", end='', flush=True)

    def run(self):
        """Main live trading loop."""
        print("\n" + "!" * 60)
        print("  !!! LIVE TRADING ENGINE - REAL MONEY !!!")
        print("!" * 60)

        self._log("=" * 50)
        self._log("LIVE TRADING CONFIGURATION")
        self._log("=" * 50)
        self._log(f"Symbol: {self.config.underlying_symbol}")
        self._log(f"Position Size: ${self.config.fixed_position_value:,.2f}")
        self._log(f"Max Position: ${self.config.max_position_value:,.2f}")
        self._log(f"Max Daily Loss: ${self.config.max_daily_loss:,.2f}")
        self._log(f"Max Trades/Day: {self.max_trades_per_day} (from strategy)")
        self._log(f"Take Profit: {self.config.target_profit_pct}% (LIMIT ORDER on exchange)")
        self._log(f"Stop Loss: {self.config.stop_loss_pct}% (monitored internally)")
        self._log(f"Entry Window: {self.config.entry_time_start} - {self.config.entry_time_end} EST")
        self._log(f"Confirm Trades: {'Yes' if self.config.require_confirmation else 'No'}")
        self._log("=" * 50)

        # Test connection and verify LIVE account
        try:
            account = self.client.get_account()
            self._log(f"Connected to Alpaca LIVE Trading", "DANGER")
            self._log(f"Account ID: {account['id']}")
            self._log(f"Cash: ${account['cash']:,.2f}")
            self._log(f"Buying Power: ${account['buying_power']:,.2f}")

            # Final confirmation before starting
            print("\n" + "!" * 60)
            print("  !!! THIS IS LIVE TRADING WITH REAL MONEY !!!")
            print("!" * 60)
            response = input("\nType 'START' to begin live trading (or anything else to cancel): ")
            if response.upper() != 'START':
                self._log("Live trading cancelled by user")
                return

        except Exception as e:
            self._log(f"Failed to connect to Alpaca: {e}", "ERROR")
            return

        self.running = True
        self._log("LIVE TRADING STARTED... (Ctrl+C to stop)", "DANGER")

        try:
            while self.running and not self._stop_event.is_set():
                # Check if new trading day
                today = datetime.now(EST).date()
                if today != self.session.date:
                    self._log(f"New trading day: {today}")
                    self.session = LiveTradingSession(date=today)

                # Update market data
                self.latest_underlying_quote = self.client.get_latest_stock_quote(
                    self.config.underlying_symbol
                )

                if not self._is_market_open():
                    self._update_status()
                    self._stop_event.wait(timeout=60)
                    continue

                # Check if stopped by limit
                if self.session.stopped_by_limit:
                    self._update_status()
                    self._stop_event.wait(timeout=60)
                    continue

                # If in position, check exit conditions
                if self.session.position:
                    self._check_exit_conditions()

                # If not in position and within entry window, look for entry
                elif self._is_entry_window() and not self.session.has_traded_today:
                    signal = self._calculate_signal()
                    if signal in ['BULLISH', 'BEARISH']:
                        self._enter_position(signal)

                # Force exit if needed
                if self.session.position and self._should_force_exit():
                    self._check_exit_conditions()

                self._update_status()

                # Sleep between iterations
                self._stop_event.wait(timeout=30)  # Faster polling for live

        except KeyboardInterrupt:
            self._log("\nShutdown requested...")
        finally:
            self.running = False

            # Close any open positions
            if self.session.position:
                self._log("CLOSING OPEN POSITION BEFORE SHUTDOWN...", "DANGER")
                option_quote = self.client.get_latest_option_quote(
                    self.session.position.symbol
                )
                if option_quote:
                    pnl = (option_quote.mid - self.session.position.entry_price) * \
                          self.session.position.qty * 100
                    self._exit_position("SHUTDOWN", option_quote.mid, pnl)

            self._log("Live trading engine stopped.")
            self._print_session_summary()

    def stop(self):
        """Signal the engine to stop."""
        self.running = False
        self._stop_event.set()

    def _print_session_summary(self):
        """Print session summary."""
        print("\n" + "=" * 60)
        print("  LIVE TRADING SESSION SUMMARY")
        print("=" * 60)
        print(f"  Date: {self.session.date}")
        print(f"  Trades: {self.session.trades_today}")
        print(f"  P&L: ${self.session.pnl_today:.2f}")
        if self.session.stopped_by_limit:
            print(f"  Stopped: {self.session.stop_reason}")
        print("=" * 60)

        # Print trade logger summary
        try:
            self.trade_logger.print_summary()
        except Exception as e:
            self._log(f"Error printing trade summary: {e}", "WARN")


def run_live_trading(config: LiveTradingConfig):
    """Run the live trading engine with given config."""
    engine = LiveTradingEngine(config)

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        engine.stop()

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    engine.run()
