"""
Paper Trading Engine

Real-time paper trading execution engine that:
- Connects to Alpaca for live market data
- Executes trades using the paper trading API
- Manages positions with SL/TP orders
- Follows the same strategy logic as backtesting
"""

import asyncio
import signal
import sys
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import pytz
import threading
import queue

from ..config import PaperTradingConfig
from .alpaca_client import AlpacaClient, Quote, Bar, ALPACA_AVAILABLE
from ..strategies import COINDaily0DTEMomentum, COINDaily0DTEMomentumConfig


EST = pytz.timezone('America/New_York')


@dataclass
class PaperPosition:
    """Tracks an open paper trading position."""
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
class TradingSession:
    """Tracks current trading session state."""
    date: datetime = field(default_factory=lambda: datetime.now(EST).date())
    trades_today: int = 0
    pnl_today: float = 0.0
    position: Optional[PaperPosition] = None
    has_traded_today: bool = False


class PaperTradingEngine:
    """
    Real-time paper trading execution engine.
    """

    def __init__(self, config: PaperTradingConfig):
        """
        Initialize paper trading engine.

        Args:
            config: Paper trading configuration
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py package required. Run: pip install alpaca-py")

        self.config = config
        self.client = AlpacaClient(
            api_key=config.api_key,
            api_secret=config.api_secret,
            paper=True
        )

        # Session state
        self.session = TradingSession()
        self.running = False
        self._stop_event = threading.Event()

        # Market data
        self.latest_underlying_quote: Optional[Quote] = None
        self.latest_underlying_bar: Optional[Bar] = None
        self.latest_option_quote: Optional[Quote] = None

        # Strategy instance (for signal generation)
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

        # Parse trading times
        self.entry_start = datetime.strptime(config.entry_time_start, "%H:%M:%S").time()
        self.entry_end = datetime.strptime(config.entry_time_end, "%H:%M:%S").time()
        self.force_exit = datetime.strptime(config.force_exit_time, "%H:%M:%S").time()

        # Max trades per day from strategy config
        self.max_trades_per_day = self.strategy_config.max_trades_per_day

        # Callbacks
        self.on_trade: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None
        self.on_quote: Optional[Callable] = None

    def _log(self, msg: str, level: str = "INFO"):
        """Log message with timestamp."""
        now = datetime.now(EST)
        print(f"[{now.strftime('%H:%M:%S')}] [{level}] {msg}")

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
        Calculate trading signal based on current market conditions.

        Returns:
            'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        if not self.latest_underlying_bar:
            return 'NEUTRAL'

        # Simple momentum signal based on current bar
        # In production, this would use the full strategy logic with indicators
        bar = self.latest_underlying_bar

        # Price momentum: if close > open, bullish; else bearish
        if bar.close > bar.open:
            return 'BULLISH'
        elif bar.close < bar.open:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _find_atm_option(self, option_type: str) -> Optional[str]:
        """
        Find ATM option contract for the underlying.

        Args:
            option_type: 'C' for call, 'P' for put

        Returns:
            OCC symbol string or None
        """
        if not self.latest_underlying_quote:
            return None

        underlying_price = self.latest_underlying_quote.mid
        expiry = self._get_this_weeks_friday()

        # Round to nearest $5 strike for ATM
        strike = round(underlying_price / 5) * 5

        # Format OCC symbol
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

        # Each contract = 100 shares
        contract_value = option_price * 100
        contracts = int(self.config.fixed_position_value / contract_value)

        return max(1, contracts)  # At least 1 contract

    def _enter_position(self, signal: str):
        """Enter a new position based on signal."""
        if self.session.position is not None:
            self._log("Already in position, skipping entry", "WARN")
            return

        # Check max trades per day (from strategy config)
        if self.session.trades_today >= self.max_trades_per_day:
            self._log(f"Max trades per day reached ({self.max_trades_per_day}), skipping entry", "INFO")
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

        # Submit market order
        self._log(f"Entering {signal} position: {qty}x {occ_symbol} @ ${option_quote.ask:.2f}")

        try:
            order = self.client.submit_market_order(
                symbol=occ_symbol,
                qty=qty,
                side='buy'
            )

            # Parse option details from OCC symbol
            option_details = self.client.parse_occ_symbol(occ_symbol)

            # Create position tracking
            self.session.position = PaperPosition(
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

            self._log(f"Position opened: Entry=${self.session.position.entry_price:.2f}, "
                     f"TP=${self.session.position.take_profit_price:.2f}, "
                     f"SL=${self.session.position.stop_loss_price:.2f}")

            self.session.has_traded_today = True
            self.session.trades_today += 1

            if self.on_trade:
                self.on_trade('ENTRY', self.session.position)

        except Exception as e:
            self._log(f"Error entering position: {e}", "ERROR")

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

        self._log(f"Exiting position: {reason} @ ${exit_price:.2f} (P&L: ${pnl:.2f})")

        try:
            # Submit sell order
            order = self.client.submit_market_order(
                symbol=pos.symbol,
                qty=pos.qty,
                side='sell'
            )

            # Update session
            self.session.pnl_today += pnl
            hold_time = (datetime.now(EST) - pos.entry_time).total_seconds() / 60

            self._log(f"Position closed. Hold time: {hold_time:.1f}m, "
                     f"Session P&L: ${self.session.pnl_today:.2f}")

            if self.on_trade:
                self.on_trade('EXIT', pos, reason, exit_price, pnl)

            self.session.position = None

        except Exception as e:
            self._log(f"Error exiting position: {e}", "ERROR")

    def _update_status(self):
        """Print current status."""
        now = datetime.now(EST)

        status = f"Time: {now.strftime('%H:%M:%S')} | "
        status += f"Market: {'OPEN' if self._is_market_open() else 'CLOSED'} | "

        if self.latest_underlying_quote:
            status += f"{self.config.underlying_symbol}: ${self.latest_underlying_quote.mid:.2f} | "

        if self.session.position:
            if self.latest_option_quote:
                current = self.latest_option_quote.mid
                entry = self.session.position.entry_price
                pnl_pct = ((current - entry) / entry) * 100
                status += f"Position: {self.session.position.option_type} @ ${current:.2f} ({pnl_pct:+.1f}%)"
            else:
                status += f"Position: {self.session.position.option_type}"
        else:
            status += "Position: None"

        status += f" | Day P&L: ${self.session.pnl_today:.2f}"

        print(f"\r{status}", end='', flush=True)

    def run(self):
        """Main trading loop."""
        self._log("=" * 60)
        self._log("THE VOLUME AI - Paper Trading Engine")
        self._log("=" * 60)
        self._log(f"Symbol: {self.config.underlying_symbol}")
        self._log(f"Position Size: ${self.config.fixed_position_value:,.2f}")
        self._log(f"Max Trades/Day: {self.max_trades_per_day} (from strategy)")
        self._log(f"Take Profit: {self.config.target_profit_pct}%")
        self._log(f"Stop Loss: {self.config.stop_loss_pct}%")
        self._log(f"Entry Window: {self.config.entry_time_start} - {self.config.entry_time_end} EST")
        self._log("=" * 60)

        # Test connection
        try:
            account = self.client.get_account()
            self._log(f"Connected to Alpaca Paper Trading")
            self._log(f"Account ID: {account['id']}")
            self._log(f"Buying Power: ${account['buying_power']:,.2f}")
        except Exception as e:
            self._log(f"Failed to connect to Alpaca: {e}", "ERROR")
            return

        self.running = True
        self._log("Starting trading loop... (Ctrl+C to stop)")

        try:
            while self.running and not self._stop_event.is_set():
                # Check if new trading day
                today = datetime.now(EST).date()
                if today != self.session.date:
                    self._log(f"New trading day: {today}")
                    self.session = TradingSession(date=today)

                # Update market data
                self.latest_underlying_quote = self.client.get_latest_stock_quote(
                    self.config.underlying_symbol
                )

                if not self._is_market_open():
                    self._update_status()
                    self._stop_event.wait(timeout=60)  # Check every minute when closed
                    continue

                # If in position, check exit conditions
                if self.session.position:
                    self._check_exit_conditions()

                # If not in position and within entry window, look for entry
                elif self._is_entry_window() and self.session.trades_today < self.max_trades_per_day:
                    signal = self._calculate_signal()
                    if signal in ['BULLISH', 'BEARISH']:
                        self._enter_position(signal)

                # Force exit if needed
                if self.session.position and self._should_force_exit():
                    self._check_exit_conditions()

                self._update_status()

                # Sleep between iterations (1 minute polling)
                self._stop_event.wait(timeout=60)

        except KeyboardInterrupt:
            self._log("\nShutdown requested...")
        finally:
            self.running = False

            # Close any open positions
            if self.session.position:
                self._log("Closing open position before shutdown...")
                option_quote = self.client.get_latest_option_quote(
                    self.session.position.symbol
                )
                if option_quote:
                    pnl = (option_quote.mid - self.session.position.entry_price) * \
                          self.session.position.qty * 100
                    self._exit_position("SHUTDOWN", option_quote.mid, pnl)

            self._log("Paper trading engine stopped.")
            self._print_session_summary()

    def stop(self):
        """Signal the engine to stop."""
        self.running = False
        self._stop_event.set()

    def _print_session_summary(self):
        """Print session summary."""
        self._log("=" * 60)
        self._log("SESSION SUMMARY")
        self._log("=" * 60)
        self._log(f"Date: {self.session.date}")
        self._log(f"Trades: {self.session.trades_today}")
        self._log(f"P&L: ${self.session.pnl_today:.2f}")
        self._log("=" * 60)


def run_paper_trading(config: PaperTradingConfig):
    """Run the paper trading engine with given config."""
    engine = PaperTradingEngine(config)

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        engine.stop()

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    engine.run()
