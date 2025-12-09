"""
Interactive Brokers Paper Trading Engine for MARA Options

This engine:
1. Connects to IB TWS/Gateway (paper trading account)
2. Fetches real-time MARA data
3. Calculates signals using MARA 0DTE Momentum strategy
4. Executes trades via IB API
5. Manages positions with TP/SL
"""

import time
import json
import signal
import sys
from datetime import datetime, time as dtime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pytz

from .ib_client import IBClient, Quote, Bar, OptionGreeks
from ..strategies.mara_0dte_momentum import MARADaily0DTEMomentum, MARADaily0DTEMomentumConfig
from ..analytics.options_trade_logger import OptionsTradeLogger, get_options_trade_logger

EST = pytz.timezone('America/New_York')


@dataclass
class IBPaperTradingConfig:
    """Configuration for IB Paper Trading."""
    # IB Connection
    ib_host: str = '127.0.0.1'
    ib_port: int = 7497  # Paper trading port (7496 for live)
    ib_client_id: int = 1

    # Symbol
    underlying_symbol: str = "MARA"

    # Trading parameters
    fixed_position_value: float = 2000.0
    target_profit_pct: float = 7.5
    stop_loss_pct: float = 25.0

    # Time windows
    entry_time_start: str = "09:30:00"
    entry_time_end: str = "15:45:00"
    force_exit_time: str = "15:50:00"
    max_hold_minutes: int = 30

    # Polling interval
    poll_interval_seconds: int = 5


@dataclass
class Position:
    """Represents an open options position."""
    symbol: str
    underlying_symbol: str
    expiry: str
    strike: float
    right: str  # 'C' or 'P'
    qty: int
    entry_price: float
    entry_time: datetime
    entry_order_id: str
    tp_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None


@dataclass
class TradingSession:
    """Tracks the current trading session state."""
    position: Optional[Position] = None
    has_traded_today: bool = False
    trades_today: int = 0
    wins: int = 0
    losses: int = 0
    pnl_today: float = 0.0
    last_signal: str = ""


class IBPaperTradingEngine:
    """
    Paper trading engine for MARA options using Interactive Brokers.

    Features:
    - Real-time IB market data
    - Dual-signal validation (Technical + Price Action)
    - Automatic TP/SL order management
    - Trade logging
    """

    def __init__(self, config: IBPaperTradingConfig):
        self.config = config
        self.ib_client: Optional[IBClient] = None
        self.running = False

        # Strategy
        self.strategy_config = MARADaily0DTEMomentumConfig(
            underlying_symbol=config.underlying_symbol,
            fixed_position_value=config.fixed_position_value,
            target_profit_pct=config.target_profit_pct,
            stop_loss_pct=config.stop_loss_pct,
            entry_time_start=config.entry_time_start,
            entry_time_end=config.entry_time_end,
            force_exit_time=config.force_exit_time,
            max_hold_minutes=config.max_hold_minutes
        )

        # Session state
        self.session = TradingSession()

        # Trade logger
        self.trade_logger: OptionsTradeLogger = get_options_trade_logger()

        # Data caches
        self.latest_quote: Optional[Quote] = None
        self.bars_1min: list = []
        self.bars_5min: list = []

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n\nShutdown signal received...")
        self.stop()

    def _log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        now = datetime.now(EST)
        prefix = {
            "INFO": "[INFO]",
            "WARN": "[WARN]",
            "ERROR": "[ERROR]",
            "TRADE": "[TRADE]",
            "SUCCESS": "[SUCCESS]"
        }.get(level, "[INFO]")
        print(f"{now.strftime('%H:%M:%S')} {prefix} {message}")

    def _parse_time(self, time_str: str) -> dtime:
        """Parse time string to time object."""
        parts = time_str.split(":")
        return dtime(int(parts[0]), int(parts[1]), int(parts[2]))

    def start(self):
        """Start the paper trading engine."""
        self._log("=" * 70)
        self._log("STARTING IB PAPER TRADING ENGINE - MARA 0DTE MOMENTUM")
        self._log("=" * 70)

        # Initialize IB client
        self.ib_client = IBClient(
            host=self.config.ib_host,
            port=self.config.ib_port,
            client_id=self.config.ib_client_id
        )

        # Connect to IB
        if not self.ib_client.connect():
            self._log("Failed to connect to Interactive Brokers!", "ERROR")
            return

        # Get account info
        account_info = self.ib_client.get_account_info()
        self._log(f"Account: ${account_info.get('NetLiquidation', 0):,.2f}")
        self._log(f"Buying Power: ${account_info.get('BuyingPower', 0):,.2f}")

        self._log(f"Symbol: {self.config.underlying_symbol}")
        self._log(f"Position Size: ${self.config.fixed_position_value:,.2f}")
        self._log(f"TP: +{self.config.target_profit_pct}% | SL: -{self.config.stop_loss_pct}%")
        self._log(f"Entry Window: {self.config.entry_time_start} - {self.config.entry_time_end} EST")
        self._log("=" * 70)

        self.running = True
        self._run_loop()

    def stop(self):
        """Stop the paper trading engine."""
        self._log("Stopping engine...")
        self.running = False

        # Close any open positions
        if self.session.position:
            self._close_position("Engine shutdown")

        # Disconnect from IB
        if self.ib_client:
            self.ib_client.disconnect()

        # Print summary
        self._log("=" * 70)
        self._log("SESSION SUMMARY")
        self._log(f"Trades: {self.session.trades_today}")
        self._log(f"Wins: {self.session.wins} | Losses: {self.session.losses}")
        win_rate = (self.session.wins / max(1, self.session.wins + self.session.losses)) * 100
        self._log(f"Win Rate: {win_rate:.1f}%")
        self._log(f"P&L: ${self.session.pnl_today:+.2f}")
        self._log("=" * 70)

        self.trade_logger.print_summary()

    def _run_loop(self):
        """Main trading loop."""
        last_check_date = None

        while self.running:
            try:
                now = datetime.now(EST)
                current_date = now.date()

                # Reset session on new day
                if last_check_date != current_date:
                    self._log(f"New trading day: {current_date}")
                    self.session.has_traded_today = False
                    self.session.trades_today = 0
                    last_check_date = current_date

                # Check market hours (9:30 AM - 4:00 PM EST)
                market_open = dtime(9, 30)
                market_close = dtime(16, 0)

                if not (market_open <= now.time() <= market_close):
                    if now.time() < market_open:
                        self._log("Waiting for market open...", "INFO")
                    else:
                        self._log("Market closed. Stopping engine.", "INFO")
                        self.stop()
                        return
                    time.sleep(60)
                    continue

                # Fetch latest data
                self._fetch_market_data()

                # Check for position management
                if self.session.position:
                    self._manage_position()
                else:
                    # Check for entry signals
                    if not self.session.has_traded_today:
                        self._check_entry_signal()

                # Sleep between iterations
                time.sleep(self.config.poll_interval_seconds)

            except Exception as e:
                self._log(f"Error in main loop: {e}", "ERROR")
                import traceback
                traceback.print_exc()
                time.sleep(10)

    def _fetch_market_data(self):
        """Fetch latest market data from IB."""
        try:
            # Get latest quote
            self.latest_quote = self.ib_client.get_latest_stock_quote(self.config.underlying_symbol)

            if self.latest_quote:
                self._log(f"{self.config.underlying_symbol}: ${self.latest_quote.mid:.2f} (Bid: ${self.latest_quote.bid:.2f} / Ask: ${self.latest_quote.ask:.2f})")

            # Get historical bars for signal calculation
            bars = self.ib_client.get_stock_bars(
                self.config.underlying_symbol,
                timeframe='1 min',
                duration='1 D'
            )

            if bars:
                self.bars_1min = bars[-60:]  # Last 60 1-min bars

            # Get 5-min bars
            bars_5min = self.ib_client.get_stock_bars(
                self.config.underlying_symbol,
                timeframe='5 mins',
                duration='1 D'
            )

            if bars_5min:
                self.bars_5min = bars_5min[-30:]  # Last 30 5-min bars

        except Exception as e:
            self._log(f"Error fetching market data: {e}", "ERROR")

    def _check_entry_signal(self):
        """Check if we should enter a position."""
        now = datetime.now(EST)

        # Check entry window
        entry_start = self._parse_time(self.config.entry_time_start)
        entry_end = self._parse_time(self.config.entry_time_end)

        if not (entry_start <= now.time() <= entry_end):
            return

        if len(self.bars_1min) < 30:
            return

        # Calculate signals
        self._log("=" * 70)
        self._log("ANALYZING ENTRY SIGNALS")

        # METHOD 1: Technical indicators (1-min bars)
        signal_1 = MARADaily0DTEMomentum.calculate_signal_from_bars(self.bars_1min, self.strategy_config)
        self._log(f"METHOD 1 (Technical): {signal_1['signal']}, Score: {signal_1['bullish_score']}/{signal_1['bearish_score']}, {signal_1['confidence']} confidence")

        # METHOD 2: Price action (5-min bars)
        signal_2 = MARADaily0DTEMomentum.calculate_price_action_signal(self.bars_5min)
        self._log(f"METHOD 2 (Price Action): {signal_2['signal']}, {signal_2['strength']}, {signal_2['bullish_points']} bull vs {signal_2['bearish_points']} bear")

        # Both methods must agree for entry
        if signal_1['signal'] == signal_2['signal'] and signal_1['signal'] != 'NEUTRAL':
            self._log(f"DUAL CONFIRMATION: {signal_1['signal']}", "SUCCESS")
            self._enter_position(signal_1['signal'])
        else:
            self._log(f"No dual confirmation (Method1: {signal_1['signal']}, Method2: {signal_2['signal']})", "WARN")

        self._log("=" * 70)

    def _enter_position(self, signal: str):
        """Enter a new position."""
        if not self.latest_quote:
            self._log("No quote available for entry", "ERROR")
            return

        self._log("=" * 70)
        self._log("ENTERING POSITION", "TRADE")

        underlying_price = self.latest_quote.mid
        option_type = 'C' if signal == 'BULLISH' else 'P'

        # Calculate expiry (this week's Friday)
        today = datetime.now(EST).date()
        weekday = today.weekday()
        days_to_friday = (4 - weekday) % 7
        if days_to_friday == 0 and datetime.now(EST).time() > dtime(15, 0):
            days_to_friday = 7
        expiry_date = today + timedelta(days=days_to_friday)
        expiry_str = expiry_date.strftime('%Y%m%d')

        # Find ATM strike
        atm_strike = round(underlying_price)  # MARA options have $1 strikes

        self._log(f"Signal: {signal} -> {option_type}")
        self._log(f"Underlying: ${underlying_price:.2f}")
        self._log(f"Strike: ${atm_strike:.2f}")
        self._log(f"Expiry: {expiry_str}")

        # Get option quote
        option_quote = self.ib_client.get_latest_option_quote(
            self.config.underlying_symbol,
            expiry_str,
            atm_strike,
            option_type
        )

        if not option_quote or option_quote.ask <= 0:
            self._log("Could not get option quote", "ERROR")
            return

        option_price = option_quote.ask
        contract_cost = option_price * 100  # 100 shares per contract

        # Calculate position size
        qty = max(1, int(self.config.fixed_position_value / contract_cost))
        total_cost = qty * contract_cost

        self._log(f"Option Price: ${option_price:.2f}")
        self._log(f"Contracts: {qty}")
        self._log(f"Total Cost: ${total_cost:.2f}")

        # Submit market order
        order_id = self.ib_client.submit_market_order(
            symbol=self.config.underlying_symbol,
            quantity=qty,
            side='BUY',
            is_option=True,
            expiry=expiry_str,
            strike=atm_strike,
            right=option_type
        )

        if not order_id:
            self._log("Failed to submit order", "ERROR")
            return

        # Create position
        option_symbol = f"{self.config.underlying_symbol}{expiry_str}{option_type}{int(atm_strike*1000):08d}"
        self.session.position = Position(
            symbol=option_symbol,
            underlying_symbol=self.config.underlying_symbol,
            expiry=expiry_str,
            strike=atm_strike,
            right=option_type,
            qty=qty,
            entry_price=option_price,
            entry_time=datetime.now(EST),
            entry_order_id=order_id
        )

        self.session.has_traded_today = True
        self.session.trades_today += 1
        self.session.last_signal = signal

        # Calculate TP/SL prices
        tp_price = option_price * (1 + self.config.target_profit_pct / 100)
        sl_price = option_price * (1 - self.config.stop_loss_pct / 100)

        self._log(f"TP Price: ${tp_price:.2f} (+{self.config.target_profit_pct}%)")
        self._log(f"SL Price: ${sl_price:.2f} (-{self.config.stop_loss_pct}%)")

        # Submit TP limit order
        tp_order_id = self.ib_client.submit_limit_order(
            symbol=self.config.underlying_symbol,
            quantity=qty,
            side='SELL',
            limit_price=tp_price,
            is_option=True,
            expiry=expiry_str,
            strike=atm_strike,
            right=option_type
        )
        self.session.position.tp_order_id = tp_order_id

        # Note: IB handles SL differently - we'll monitor internally
        self._log(f"ENTRY COMPLETE - Order ID: {order_id}", "SUCCESS")

        # Log trade
        try:
            trade_id = f"IB_{option_symbol}_{datetime.now(EST).strftime('%Y%m%d_%H%M%S')}"

            # Get Greeks
            greeks = self.ib_client.get_option_greeks(
                self.config.underlying_symbol,
                expiry_str,
                atm_strike,
                option_type
            )
            greeks_dict = {}
            if greeks:
                greeks_dict = {
                    'delta': greeks.delta,
                    'gamma': greeks.gamma,
                    'theta': greeks.theta,
                    'vega': greeks.vega,
                    'iv': greeks.implied_volatility
                }
                self._log(f"Greeks: Δ={greeks.delta:.3f} Γ={greeks.gamma:.4f} Θ={greeks.theta:.3f} IV={greeks.implied_volatility:.1%}")

            self.trade_logger.log_entry(
                trade_id=trade_id,
                underlying_symbol=self.config.underlying_symbol,
                option_symbol=option_symbol,
                option_type='call' if option_type == 'C' else 'put',
                strike_price=atm_strike,
                expiration_date=expiry_date.strftime('%Y-%m-%d'),
                entry_time=datetime.now(EST),
                entry_price=option_price,
                entry_qty=qty,
                entry_order_id=order_id,
                entry_underlying_price=underlying_price,
                greeks=greeks_dict,
                target_profit_pct=self.config.target_profit_pct,
                stop_loss_pct=self.config.stop_loss_pct,
                notes="IB PAPER TRADING"
            )
            self.session.position.entry_order_id = trade_id
        except Exception as e:
            self._log(f"Error logging trade: {e}", "WARN")

        self._log("=" * 70)

    def _manage_position(self):
        """Manage open position - check for exits."""
        if not self.session.position:
            return

        pos = self.session.position
        now = datetime.now(EST)

        # Check force exit time
        force_exit = self._parse_time(self.config.force_exit_time)
        if now.time() >= force_exit:
            self._log("FORCE EXIT: End of day", "TRADE")
            self._close_position("Force exit - EOD")
            return

        # Check hold time
        hold_minutes = (now - pos.entry_time).total_seconds() / 60
        if hold_minutes >= self.config.max_hold_minutes:
            self._log(f"MAX HOLD TIME: {hold_minutes:.1f} minutes", "TRADE")
            self._close_position("Max hold time exceeded")
            return

        # Check TP order status
        if pos.tp_order_id:
            status = self.ib_client.get_order_status(pos.tp_order_id)
            if status == 'Filled':
                self._log("TAKE PROFIT HIT!", "SUCCESS")
                self._handle_exit("TAKE_PROFIT", pos.entry_price * (1 + self.config.target_profit_pct / 100))
                return

        # Check SL (internal monitoring)
        option_quote = self.ib_client.get_latest_option_quote(
            pos.underlying_symbol,
            pos.expiry,
            pos.strike,
            pos.right
        )

        if option_quote and option_quote.bid > 0:
            current_price = option_quote.bid
            pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100

            if pnl_pct <= -self.config.stop_loss_pct:
                self._log(f"STOP LOSS HIT: {pnl_pct:.1f}%", "TRADE")
                self._close_position("STOP_LOSS")

    def _close_position(self, reason: str):
        """Close the current position."""
        if not self.session.position:
            return

        pos = self.session.position

        # Cancel TP order if exists
        if pos.tp_order_id:
            self.ib_client.cancel_order(pos.tp_order_id)

        # Submit market sell order
        order_id = self.ib_client.submit_market_order(
            symbol=pos.underlying_symbol,
            quantity=pos.qty,
            side='SELL',
            is_option=True,
            expiry=pos.expiry,
            strike=pos.strike,
            right=pos.right
        )

        # Get exit price
        option_quote = self.ib_client.get_latest_option_quote(
            pos.underlying_symbol,
            pos.expiry,
            pos.strike,
            pos.right
        )
        exit_price = option_quote.bid if option_quote else pos.entry_price

        self._handle_exit(reason, exit_price, order_id)

    def _handle_exit(self, reason: str, exit_price: float, exit_order_id: str = None):
        """Handle position exit."""
        pos = self.session.position
        if not pos:
            return

        # Calculate P&L
        pnl = (exit_price - pos.entry_price) * pos.qty * 100
        pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100
        hold_time = (datetime.now(EST) - pos.entry_time).total_seconds() / 60

        self._log("=" * 70, "TRADE")
        self._log("POSITION CLOSED", "TRADE")
        self._log(f"Entry: ${pos.entry_price:.2f} -> Exit: ${exit_price:.2f}", "TRADE")
        self._log(f"P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)", "SUCCESS" if pnl >= 0 else "WARN")
        self._log(f"Hold Time: {hold_time:.1f} minutes", "TRADE")
        self._log(f"Reason: {reason}", "TRADE")
        self._log("=" * 70, "TRADE")

        # Update session stats
        self.session.pnl_today += pnl
        if pnl >= 0:
            self.session.wins += 1
        else:
            self.session.losses += 1

        # Log trade exit
        try:
            self.trade_logger.log_exit(
                trade_id=pos.entry_order_id,
                exit_time=datetime.now(EST),
                exit_price=exit_price,
                exit_qty=pos.qty,
                exit_order_id=exit_order_id or "",
                exit_reason=reason,
                exit_underlying_price=self.latest_quote.mid if self.latest_quote else 0.0,
                notes=f"IB PAPER TRADING - Hold time: {hold_time:.1f}m"
            )
        except Exception as e:
            self._log(f"Error logging exit: {e}", "WARN")

        # Clear position
        self.session.position = None


def load_ib_config(config_path: str) -> IBPaperTradingConfig:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        data = json.load(f)

    return IBPaperTradingConfig(
        ib_host=data.get('ib_host', '127.0.0.1'),
        ib_port=data.get('ib_port', 7497),
        ib_client_id=data.get('ib_client_id', 1),
        underlying_symbol=data.get('underlying_symbol', 'MARA'),
        fixed_position_value=data.get('fixed_position_value', 2000.0),
        target_profit_pct=data.get('target_profit_pct', 7.5),
        stop_loss_pct=data.get('stop_loss_pct', 25.0),
        entry_time_start=data.get('entry_time_start', '09:30:00'),
        entry_time_end=data.get('entry_time_end', '15:45:00'),
        force_exit_time=data.get('force_exit_time', '15:50:00'),
        max_hold_minutes=data.get('max_hold_minutes', 30),
        poll_interval_seconds=data.get('poll_interval_seconds', 5)
    )
