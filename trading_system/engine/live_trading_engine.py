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
        """Calculate trading signal based on current market conditions."""
        if not self.latest_underlying_bar:
            return 'NEUTRAL'

        bar = self.latest_underlying_bar

        if bar.close > bar.open:
            return 'BULLISH'
        elif bar.close < bar.open:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

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

            self.session.has_traded_today = True
            self.session.trades_today += 1

            # Log trade
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

        except Exception as e:
            self._log(f"ERROR EXECUTING TRADE: {e}", "ERROR")

    def _check_exit_conditions(self):
        """Check if position should be exited."""
        if self.session.position is None:
            return

        pos = self.session.position

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

        # Check take profit
        if current_price >= pos.take_profit_price:
            exit_reason = "TAKE_PROFIT"

        # Check stop loss
        elif current_price <= pos.stop_loss_price:
            exit_reason = "STOP_LOSS"

        # Check max hold time
        elif (datetime.now(EST) - pos.entry_time).total_seconds() / 60 >= self.config.max_hold_minutes:
            exit_reason = "TIME_EXIT"

        # Check force exit time
        elif self._should_force_exit():
            exit_reason = "FORCE_EXIT"

        if exit_reason:
            self._exit_position(exit_reason, current_price, pnl_dollars)

    def _exit_position(self, reason: str, exit_price: float, pnl: float):
        """Exit the current position."""
        if self.session.position is None:
            return

        pos = self.session.position

        self._log(f"LIVE EXIT SIGNAL: {reason} @ ${exit_price:.2f} (P&L: ${pnl:.2f})", "TRADE")

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

            # Log trade
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
        self._log(f"Take Profit: {self.config.target_profit_pct}%")
        self._log(f"Stop Loss: {self.config.stop_loss_pct}%")
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
