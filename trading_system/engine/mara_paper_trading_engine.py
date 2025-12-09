"""
MARA Paper Trading Engine for 0DTE Options

Real-time paper trading execution engine for MARA (Marathon Digital) options:
- Connects to Alpaca for live market data
- Executes trades using the paper trading API
- Manages positions with TP LIMIT order on exchange, SL monitored internally
- Uses MARA 0DTE momentum strategy with dual signal validation
"""

import os
import signal
import sys
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import pytz
import threading

from .alpaca_client import AlpacaClient, Quote, Bar, ALPACA_AVAILABLE
from ..strategies.mara_0dte_momentum import MARADaily0DTEMomentum, MARADaily0DTEMomentumConfig
from ..analytics.options_trade_logger import OptionsTradeLogger, get_options_trade_logger


EST = pytz.timezone('America/New_York')


@dataclass
class MARAPaperPosition:
    """Tracks an open MARA paper trading position."""
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
    highest_price_since_entry: float = 0.0

    # Order IDs
    entry_order_id: str = ""
    sl_order_id: str = ""
    tp_order_id: str = ""


@dataclass
class MARATradingSession:
    """Tracks current MARA trading session state."""
    date: datetime = field(default_factory=lambda: datetime.now(EST).date())
    trades_today: int = 0
    wins: int = 0
    losses: int = 0
    pnl_today: float = 0.0
    position: Optional[MARAPaperPosition] = None
    has_traded_today: bool = False


class MARAPaperTradingEngine:
    """
    Real-time MARA paper trading execution engine.
    """

    def __init__(self, config: MARADaily0DTEMomentumConfig, api_key: str, api_secret: str):
        """Initialize MARA paper trading engine."""
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py package required. Run: pip install alpaca-py")

        self.config = config
        self.client = AlpacaClient(
            api_key=api_key,
            api_secret=api_secret,
            paper=True
        )

        # Session state
        self.session = MARATradingSession()
        self.running = False
        self._stop_event = threading.Event()
        self._status_counter = 0

        # Market data
        self.latest_underlying_quote: Optional[Quote] = None
        self.latest_underlying_bar: Optional[Bar] = None
        self.latest_option_quote: Optional[Quote] = None

        # Parse trading times
        self.entry_start = datetime.strptime(config.entry_time_start, "%H:%M:%S").time()
        self.entry_end = datetime.strptime(config.entry_time_end, "%H:%M:%S").time()
        self.force_exit = datetime.strptime(config.force_exit_time, "%H:%M:%S").time()

        # Trade logging
        self.trade_logger: OptionsTradeLogger = get_options_trade_logger()

    def _log(self, msg: str, level: str = "INFO"):
        """Log message with timestamp."""
        now = datetime.now(EST)
        color = ""
        reset = ""

        if sys.platform != 'win32' or 'TERM' in os.environ:
            if level == "ERROR":
                color = "\033[91m"
            elif level == "WARN":
                color = "\033[93m"
            elif level == "SUCCESS":
                color = "\033[92m"
            elif level == "TRADE":
                color = "\033[96m"
            reset = "\033[0m"

        print(f"{color}[{now.strftime('%H:%M:%S')}] [{level}] {msg}{reset}")

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(EST)
        if now.weekday() >= 5:
            return False
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
        if weekday <= 4:
            days_to_friday = 4 - weekday
        else:
            days_to_friday = (4 - weekday) % 7
        friday = now + timedelta(days=days_to_friday)
        return friday.replace(hour=16, minute=0, second=0, microsecond=0)

    def _recover_existing_positions(self):
        """Check for existing MARA positions in Alpaca on startup."""
        self._log("Checking for existing MARA positions in Alpaca...", "INFO")

        import time

        alpaca_positions = []
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                alpaca_positions = self.client.get_options_positions()
                if alpaca_positions:
                    break
                if attempt < max_retries:
                    time.sleep(2)
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(2)

        # Check open orders as backup
        if not alpaca_positions:
            try:
                open_orders = self.client.get_open_orders("MARA")
                for order in open_orders:
                    symbol = order.get('symbol', '')
                    side = order.get('side', '')
                    if side == 'sell' and 'MARA' in symbol:
                        self._log(f"Found existing TP order for MARA: {symbol}", "WARN")
                        # Recover position from TP order
                        qty = int(float(order.get('qty', 0)))
                        tp_price = float(order.get('limit_price', 0))
                        estimated_entry = tp_price / (1 + self.config.target_profit_pct / 100)

                        try:
                            option_type = 'CALL' if 'C' in symbol[6:13] else 'PUT'
                            strike_str = symbol[-8:]
                            strike = float(strike_str) / 1000
                            expiry_str = symbol[4:10]
                            expiry = datetime.strptime(f"20{expiry_str}", "%Y%m%d")
                        except Exception:
                            option_type = 'CALL'
                            strike = 0
                            expiry = self._get_this_weeks_friday()

                        self.session.position = MARAPaperPosition(
                            symbol=symbol,
                            underlying="MARA",
                            qty=qty,
                            side='long',
                            entry_price=estimated_entry,
                            entry_time=datetime.now(EST),
                            option_type=option_type,
                            strike=strike,
                            expiration=expiry,
                            signal='BULLISH' if option_type == 'CALL' else 'BEARISH',
                            stop_loss_price=estimated_entry * (1 - self.config.stop_loss_pct / 100),
                            take_profit_price=tp_price,
                            highest_price_since_entry=estimated_entry,
                        )
                        self.session.position.tp_order_id = order.get('id', '')
                        self.session.has_traded_today = True
                        self.session.trades_today = 1
                        self._log(f"Position RECOVERED from TP order", "SUCCESS")
                        return
            except Exception as e:
                self._log(f"Error checking open orders: {e}", "WARN")
            self._log("No existing MARA positions found", "INFO")
            return

        # Process positions found
        for pos in alpaca_positions:
            symbol = pos.get('symbol', '')
            if 'MARA' in symbol:
                self._log(f"FOUND EXISTING MARA POSITION: {symbol}", "WARN")
                entry_price = pos.get('avg_entry_price', 0)
                qty = int(pos.get('qty', 0))

                try:
                    option_type = 'CALL' if 'C' in symbol[6:13] else 'PUT'
                    strike_str = symbol[-8:]
                    strike = float(strike_str) / 1000
                    expiry_str = symbol[4:10]
                    expiry = datetime.strptime(f"20{expiry_str}", "%Y%m%d")
                except Exception:
                    option_type = 'CALL'
                    strike = 0
                    expiry = self._get_this_weeks_friday()

                self.session.position = MARAPaperPosition(
                    symbol=symbol,
                    underlying="MARA",
                    qty=qty,
                    side='long',
                    entry_price=entry_price,
                    entry_time=datetime.now(EST),
                    option_type=option_type,
                    strike=strike,
                    expiration=expiry,
                    signal='BULLISH' if option_type == 'CALL' else 'BEARISH',
                    stop_loss_price=entry_price * (1 - self.config.stop_loss_pct / 100),
                    take_profit_price=entry_price * (1 + self.config.target_profit_pct / 100),
                    highest_price_since_entry=pos.get('current_price', entry_price),
                )
                self.session.has_traded_today = True
                self.session.trades_today = 1
                self._log(f"Position RECOVERED", "SUCCESS")
                break

    def _calculate_signal(self) -> str:
        """Calculate trading signal using dual confirmation."""
        from datetime import timedelta

        # Get 1-MINUTE bars for Technical Scoring
        start_time = datetime.now(EST) - timedelta(hours=3)
        bars_1min = self.client.get_stock_bars(
            "MARA",
            timeframe='1Min',
            start=start_time,
            limit=180
        )

        if not bars_1min or len(bars_1min) < 30:
            self._log(f"Not enough 1-min bar data ({len(bars_1min) if bars_1min else 0} bars)", "WARN")
            return 'NEUTRAL'

        bars_1min = bars_1min[-60:]
        if bars_1min:
            self.latest_underlying_bar = bars_1min[-1]

        # Get 5-MINUTE bars for Price Action
        start_time_5min = datetime.now(EST) - timedelta(hours=6)
        bars_5min = self.client.get_stock_bars(
            "MARA",
            timeframe='5Min',
            start=start_time_5min,
            limit=100
        )

        if not bars_5min or len(bars_5min) < 12:
            self._log(f"Not enough 5-min bar data ({len(bars_5min) if bars_5min else 0} bars)", "WARN")
            return 'NEUTRAL'

        bars_5min = bars_5min[-30:]

        # DUAL SIGNAL VALIDATION
        print()
        self._log("=" * 65, "INFO")
        self._log("  MARA DUAL SIGNAL VALIDATION", "INFO")
        self._log("=" * 65, "INFO")

        # METHOD 1: Technical Scoring
        tech_result = MARADaily0DTEMomentum.calculate_signal_from_bars(bars_1min)
        tech_signal = tech_result['signal']
        tech_confidence = tech_result['confidence']
        indicators = tech_result.get('indicators', {})

        self._log("  METHOD 1 - Technical Scoring (1-MIN):", "INFO")
        if indicators:
            self._log(f"    Price: ${indicators.get('price', 0):.2f} | VWAP: ${indicators.get('vwap', 0):.2f}", "INFO")
            self._log(f"    EMA9: ${indicators.get('ema_9', 0):.2f} | EMA20: ${indicators.get('ema_20', 0):.2f}", "INFO")
            self._log(f"    RSI: {indicators.get('rsi', 0):.1f} | MACD: {indicators.get('macd_line', 0):.4f}", "INFO")

        self._log(f"    Signal: {tech_signal} | Score: {tech_result['bullish_score']}/17 vs {tech_result['bearish_score']}/17 | Confidence: {tech_confidence}", "INFO")

        for sig in tech_result.get('bullish_signals', []):
            self._log(f"      + {sig}", "SUCCESS")
        for sig in tech_result.get('bearish_signals', []):
            self._log(f"      - {sig}", "WARN")

        self._log("-" * 65, "INFO")

        # METHOD 2: Price Action
        pa_result = MARADaily0DTEMomentum.calculate_price_action_signal(bars_5min)
        pa_signal = pa_result['signal']
        pa_strength = pa_result['strength']

        self._log("  METHOD 2 - Price Action (5-MIN):", "INFO")
        self._log(f"    Signal: {pa_signal} | Strength: {pa_strength} | Points: {pa_result['bullish_points']} bull vs {pa_result['bearish_points']} bear", "INFO")

        for reason in pa_result.get('reasons', []):
            if any(x in reason.lower() for x in ['bullish', 'green', 'higher', 'above', 'uptrend']) or '+' in reason:
                self._log(f"      + {reason}", "SUCCESS")
            elif any(x in reason.lower() for x in ['bearish', 'red', 'lower', 'below', 'downtrend']):
                self._log(f"      - {reason}", "WARN")
            else:
                self._log(f"      * {reason}", "INFO")

        self._log("-" * 65, "INFO")

        # FINAL DECISION
        self._log("  FINAL DECISION:", "INFO")

        if tech_signal == 'BULLISH' and pa_signal == 'BULLISH':
            final_signal = 'BULLISH'
            self._log(f"    CONFIRMED BULLISH - BOTH METHODS AGREE", "SUCCESS")
            self._log(f"    >>> EXECUTING: BUY CALLS", "SUCCESS")

        elif tech_signal == 'BEARISH' and pa_signal == 'BEARISH':
            final_signal = 'BEARISH'
            self._log(f"    CONFIRMED BEARISH - BOTH METHODS AGREE", "WARN")
            self._log(f"    >>> EXECUTING: BUY PUTS", "WARN")

        elif tech_signal == 'NEUTRAL' or pa_signal == 'NEUTRAL':
            final_signal = 'NEUTRAL'
            self._log(f"    NO TRADE - One or both methods neutral", "INFO")
            self._log(f"    >>> SKIPPING TRADE", "INFO")

        else:
            final_signal = 'NEUTRAL'
            self._log(f"    CONFLICTING SIGNALS - NO TRADE", "WARN")
            self._log(f"    Technical: {tech_signal} | Price Action: {pa_signal}", "WARN")
            self._log(f"    >>> SKIPPING TRADE (signals disagree)", "WARN")

        self._log("=" * 65, "INFO")

        return final_signal

    def _find_atm_option(self, option_type: str) -> Optional[str]:
        """Find ATM option contract for MARA."""
        if not self.latest_underlying_quote:
            return None

        underlying_price = self.latest_underlying_quote.mid
        expiry = self._get_this_weeks_friday()

        # MARA has $1 strikes, so round to nearest dollar
        strike = round(underlying_price)

        occ_symbol = self.client.format_occ_symbol(
            underlying="MARA",
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

    def _enter_position(self, signal: str):
        """Enter a new MARA position."""
        if self.session.position is not None:
            self._log("Already in position, skipping entry", "WARN")
            return

        if self.session.trades_today >= self.config.max_trades_per_day:
            self._log(f"Max trades per day reached ({self.config.max_trades_per_day})", "INFO")
            return

        # Safety check: verify no existing position in Alpaca
        try:
            alpaca_positions = self.client.get_options_positions()
            for pos in alpaca_positions:
                if 'MARA' in pos.get('symbol', ''):
                    self._log(f"BLOCKING ENTRY: Found existing MARA position", "WARN")
                    self._recover_existing_positions()
                    return
        except Exception as e:
            self._log(f"Warning: Could not verify Alpaca state: {e}", "WARN")

        option_type = 'C' if signal == 'BULLISH' else 'P'
        option_type_name = 'CALL' if signal == 'BULLISH' else 'PUT'
        occ_symbol = self._find_atm_option(option_type)

        if not occ_symbol:
            self._log("Could not find ATM option", "ERROR")
            return

        option_quote = self.client.get_latest_option_quote(occ_symbol)
        if not option_quote:
            self._log(f"Could not get quote for {occ_symbol}", "ERROR")
            return

        qty = self._calculate_position_size(option_quote.ask)
        if qty == 0:
            self._log("Position size would be 0, skipping", "WARN")
            return

        option_details = self.client.parse_occ_symbol(occ_symbol)

        print()
        self._log("=" * 70, "TRADE")
        self._log(f">>> MARA ENTRY SIGNAL: {signal} - {option_type_name} OPTION", "TRADE")
        self._log("=" * 70, "TRADE")
        self._log(f"    Underlying: MARA @ ${self.latest_underlying_quote.mid:.2f}", "TRADE")
        self._log(f"    Option: {occ_symbol}", "TRADE")
        self._log(f"    Strike: ${option_details['strike']:.0f} | Expiry: {option_details['expiration']}", "TRADE")
        self._log(f"    Option Bid: ${option_quote.bid:.2f} | Ask: ${option_quote.ask:.2f} | Mid: ${option_quote.mid:.2f}", "TRADE")
        self._log(f"    Qty: {qty} contracts | Value: ${self.config.fixed_position_value:.2f}", "TRADE")
        self._log(f"    Submitting BUY order to Alpaca...", "TRADE")

        try:
            order = self.client.submit_market_order(
                symbol=occ_symbol,
                qty=qty,
                side='buy'
            )

            order_id = order.get('id', 'N/A')
            order_status = order.get('status', 'unknown')

            self._log(f"    ORDER SUBMITTED:", "SUCCESS")
            self._log(f"      Order ID: {order_id}", "SUCCESS")
            self._log(f"      Status: {order_status}", "SUCCESS")

            # Wait for fill
            actual_fill_price = option_quote.ask
            self._log(f"    Waiting for fill confirmation...", "TRADE")

            import time as time_module
            for wait_attempt in range(10):
                time_module.sleep(1)
                filled_order = self.client.get_order(order_id)
                if filled_order:
                    fill_status = filled_order.get('status', '')
                    fill_price = filled_order.get('filled_avg_price')
                    if fill_status == 'filled' and fill_price:
                        actual_fill_price = float(fill_price)
                        self._log(f"    ORDER FILLED @ ${actual_fill_price:.2f}", "SUCCESS")
                        break

            # Create position tracking
            self.session.position = MARAPaperPosition(
                symbol=occ_symbol,
                underlying="MARA",
                qty=qty,
                side='long',
                entry_price=actual_fill_price,
                entry_time=datetime.now(EST),
                option_type=option_type_name,
                strike=option_details['strike'],
                expiration=option_details['expiration'],
                signal=signal,
                entry_order_id=order_id,
                highest_price_since_entry=actual_fill_price,
            )

            # Calculate SL/TP
            sl_price = actual_fill_price * (1 - self.config.stop_loss_pct / 100)
            tp_price = actual_fill_price * (1 + self.config.target_profit_pct / 100)
            self.session.position.stop_loss_price = sl_price
            self.session.position.take_profit_price = tp_price

            self._log(f"    POSITION OPENED:", "SUCCESS")
            self._log(f"      Entry Price: ${actual_fill_price:.2f}", "SUCCESS")
            self._log(f"      Take Profit: ${tp_price:.2f} (+{self.config.target_profit_pct}%)", "SUCCESS")
            self._log(f"      Stop Loss: ${sl_price:.2f} (-{self.config.stop_loss_pct}%)", "SUCCESS")

            # Place TP LIMIT order on exchange
            self._log(f"    Submitting TAKE PROFIT limit order to Alpaca...", "TRADE")
            tp_placed = False
            max_retries = 3

            for attempt in range(1, max_retries + 1):
                try:
                    tp_order = self.client.submit_option_limit_order(
                        symbol=occ_symbol,
                        qty=qty,
                        side='sell',
                        limit_price=round(tp_price, 2),
                    )
                    tp_order_id = tp_order.get('id', 'N/A')
                    self.session.position.tp_order_id = tp_order_id
                    tp_placed = True
                    self._log(f"    TAKE PROFIT ORDER PLACED:", "SUCCESS")
                    self._log(f"      TP Order ID: {tp_order_id}", "SUCCESS")
                    self._log(f"      TP Trigger: ${tp_price:.2f}", "SUCCESS")
                    break
                except Exception as tp_err:
                    if attempt < max_retries:
                        self._log(f"    Attempt {attempt}/{max_retries} failed: {tp_err}", "WARN")
                        import time
                        time.sleep(2)
                    else:
                        self._log(f"    FAILED after {max_retries} attempts: {tp_err}", "ERROR")

            print()
            if tp_placed:
                self._log("*" * 60, "SUCCESS")
                self._log("***  TP LIMIT ORDER ON EXCHANGE | SL MONITORED INTERNALLY ***", "SUCCESS")
                self._log("*" * 60, "SUCCESS")
            else:
                self._log("*" * 60, "WARN")
                self._log("***  TP ORDER FAILED - MONITORING BOTH TP/SL INTERNALLY  ***", "WARN")
                self._log("*" * 60, "WARN")

            self._log("=" * 70, "TRADE")
            print()

            self.session.has_traded_today = True
            self.session.trades_today += 1

            # Log trade entry with Greeks
            try:
                trade_id = f"PAPER_MARA_{occ_symbol}_{datetime.now(EST).strftime('%Y%m%d_%H%M%S')}"
                expiry_str = self.session.position.expiration.strftime("%Y-%m-%d") if isinstance(self.session.position.expiration, datetime) else str(self.session.position.expiration)[:10]

                # Fetch Greeks from Alpaca snapshot
                greeks_dict = {}
                try:
                    option_greeks = self.client.get_option_greeks(occ_symbol)
                    if option_greeks:
                        greeks_dict = {
                            'delta': option_greeks.delta,
                            'gamma': option_greeks.gamma,
                            'theta': option_greeks.theta,
                            'vega': option_greeks.vega,
                            'iv': option_greeks.implied_volatility
                        }
                        self._log(f"    Greeks: Δ={option_greeks.delta:.3f} Γ={option_greeks.gamma:.4f} Θ={option_greeks.theta:.3f} V={option_greeks.vega:.3f} IV={option_greeks.implied_volatility:.1%}", "INFO")
                except Exception as greek_err:
                    self._log(f"    Could not fetch Greeks: {greek_err}", "WARN")

                self.trade_logger.log_entry(
                    trade_id=trade_id,
                    underlying_symbol="MARA",
                    option_symbol=occ_symbol,
                    option_type='call' if option_type == 'C' else 'put',
                    strike_price=self.session.position.strike,
                    expiration_date=expiry_str,
                    entry_time=datetime.now(EST),
                    entry_price=actual_fill_price,
                    entry_qty=qty,
                    entry_order_id=order_id,
                    entry_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0.0,
                    greeks=greeks_dict,
                    target_profit_pct=self.config.target_profit_pct,
                    stop_loss_pct=self.config.stop_loss_pct,
                    notes="MARA PAPER TRADING"
                )
                self.session.position.entry_order_id = trade_id
            except Exception as log_err:
                self._log(f"Error logging trade entry: {log_err}", "WARN")

        except Exception as e:
            self._log(f"Error entering position: {e}", "ERROR")
            import traceback
            traceback.print_exc()

    def _check_exit_conditions(self):
        """Check if position should be exited."""
        if self.session.position is None:
            return

        pos = self.session.position

        # Check if TP order filled
        if pos.tp_order_id:
            try:
                tp_order = self.client.get_order(pos.tp_order_id)
                if tp_order and tp_order.get('status') == 'filled':
                    fill_price = float(tp_order.get('filled_avg_price', pos.take_profit_price))
                    pnl_dollars = (fill_price - pos.entry_price) * pos.qty * 100
                    self._log(f"TP LIMIT ORDER FILLED @ ${fill_price:.2f}!", "SUCCESS")
                    self._handle_tp_filled(fill_price, pnl_dollars, tp_order.get('id', ''))
                    return
            except Exception as e:
                self._log(f"Error checking TP order: {e}", "WARN")

        # Get position data from Alpaca
        alpaca_pos = self.client.get_position_by_symbol(pos.symbol)

        if not alpaca_pos and pos.tp_order_id:
            try:
                tp_order = self.client.get_order(pos.tp_order_id)
                tp_status = tp_order.get('status', 'unknown') if tp_order else 'unknown'
                if tp_status == 'filled':
                    fill_price = float(tp_order.get('filled_avg_price', pos.take_profit_price))
                    pnl_dollars = (fill_price - pos.entry_price) * pos.qty * 100
                    self._handle_tp_filled(fill_price, pnl_dollars, tp_order.get('id', ''))
                    return
            except Exception as e:
                self._log(f"Position check error: {e}", "WARN")

        # Get current price
        if alpaca_pos:
            current_price = alpaca_pos['current_price']
            if abs(alpaca_pos['avg_entry_price'] - pos.entry_price) > 0.01:
                pos.entry_price = alpaca_pos['avg_entry_price']
                pos.stop_loss_price = pos.entry_price * (1 - self.config.stop_loss_pct / 100)
                pos.take_profit_price = pos.entry_price * (1 + self.config.target_profit_pct / 100)
            pnl_pct = alpaca_pos['unrealized_plpc']
            pnl_dollars = alpaca_pos['unrealized_pl']
            self.latest_option_quote = Quote(
                symbol=pos.symbol,
                bid=current_price,
                ask=current_price,
                mid=current_price,
                timestamp=datetime.now(EST)
            )
        else:
            option_quote = self.client.get_latest_option_quote(pos.symbol)
            if not option_quote:
                return
            current_price = option_quote.mid
            self.latest_option_quote = option_quote
            pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
            pnl_dollars = (current_price - pos.entry_price) * pos.qty * 100

        if current_price > pos.highest_price_since_entry:
            pos.highest_price_since_entry = current_price

        exit_reason = None

        if current_price >= pos.take_profit_price:
            exit_reason = "TAKE_PROFIT"
        elif current_price <= pos.stop_loss_price:
            exit_reason = "STOP_LOSS"
        elif self._should_force_exit():
            exit_reason = "FORCE_EXIT"

        if exit_reason:
            self._exit_position(exit_reason, current_price, pnl_dollars)

    def _handle_tp_filled(self, fill_price: float, pnl: float, order_id: str):
        """Handle TP limit order fill."""
        if self.session.position is None:
            return

        pos = self.session.position
        hold_time = (datetime.now(EST) - pos.entry_time).total_seconds() / 60
        pnl_pct = ((fill_price - pos.entry_price) / pos.entry_price) * 100

        print()
        self._log("=" * 70, "TRADE")
        self._log("<<< TAKE PROFIT FILLED (Limit Order on Exchange)", "SUCCESS")
        self._log("=" * 70, "TRADE")
        self._log(f"    Option: {pos.symbol}", "TRADE")
        self._log(f"    Entry Price: ${pos.entry_price:.2f}", "TRADE")
        self._log(f"    Exit Price: ${fill_price:.2f}", "SUCCESS")
        self._log(f"    Hold Time: {hold_time:.1f} minutes", "TRADE")

        self.session.pnl_today += pnl
        if pnl > 0:
            self.session.wins += 1
        else:
            self.session.losses += 1

        log_level = "SUCCESS" if pnl >= 0 else "WARN"
        self._log(f"    RESULT: P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)", log_level)

        win_rate = (self.session.wins / (self.session.wins + self.session.losses) * 100) if (self.session.wins + self.session.losses) > 0 else 0
        self._log(f"    SESSION: W/L {self.session.wins}/{self.session.losses} ({win_rate:.0f}%) | Total P&L: ${self.session.pnl_today:+.2f}", "INFO")
        self._log("=" * 70, "TRADE")
        print()

        try:
            self.trade_logger.log_exit(
                trade_id=pos.entry_order_id,
                exit_time=datetime.now(EST),
                exit_price=fill_price,
                exit_qty=pos.qty,
                exit_order_id=order_id,
                exit_reason="TAKE_PROFIT",
                exit_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0.0,
                exit_greeks={},
                notes=f"MARA PAPER TRADING - TP Limit Order Filled - Hold time: {hold_time:.1f}m"
            )
        except Exception as log_err:
            self._log(f"Error logging trade exit: {log_err}", "WARN")

        self.session.position = None

    def _exit_position(self, reason: str, exit_price: float, pnl: float):
        """Exit the current position."""
        if self.session.position is None:
            return

        pos = self.session.position
        hold_time = (datetime.now(EST) - pos.entry_time).total_seconds() / 60
        pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100

        print()
        self._log("=" * 70, "TRADE")
        self._log(f"<<< MARA EXIT SIGNAL: {reason}", "TRADE")
        self._log("=" * 70, "TRADE")
        self._log(f"    Option: {pos.symbol}", "TRADE")
        self._log(f"    Entry Price: ${pos.entry_price:.2f}", "TRADE")
        self._log(f"    Exit Price: ${exit_price:.2f}", "TRADE")
        self._log(f"    Hold Time: {hold_time:.1f} minutes", "TRADE")

        # Cancel TP order if exiting for other reason
        if pos.tp_order_id and reason != "TAKE_PROFIT":
            self._log(f"    Cancelling take profit order...", "TRADE")
            try:
                self.client.cancel_order(pos.tp_order_id)
                self._log(f"    Take profit order cancelled", "SUCCESS")
            except Exception as cancel_err:
                self._log(f"    Warning: Error cancelling TP order: {cancel_err}", "WARN")

        self._log(f"    Submitting SELL order to Alpaca...", "TRADE")

        try:
            order = self.client.submit_market_order(
                symbol=pos.symbol,
                qty=pos.qty,
                side='sell'
            )

            order_id = order.get('id', 'N/A')
            self._log(f"    ORDER SUBMITTED: {order_id}", "SUCCESS")

            self.session.pnl_today += pnl
            if pnl > 0:
                self.session.wins += 1
            else:
                self.session.losses += 1

            log_level = "SUCCESS" if pnl >= 0 else "WARN"
            self._log(f"    RESULT: P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Reason: {reason}", log_level)

            win_rate = (self.session.wins / (self.session.wins + self.session.losses) * 100) if (self.session.wins + self.session.losses) > 0 else 0
            self._log(f"    SESSION: W/L {self.session.wins}/{self.session.losses} ({win_rate:.0f}%) | Total P&L: ${self.session.pnl_today:+.2f}", "INFO")
            self._log("=" * 70, "TRADE")
            print()

            try:
                self.trade_logger.log_exit(
                    trade_id=pos.entry_order_id,
                    exit_time=datetime.now(EST),
                    exit_price=exit_price,
                    exit_qty=pos.qty,
                    exit_order_id=order_id,
                    exit_reason=reason,
                    exit_underlying_price=self.latest_underlying_quote.mid if self.latest_underlying_quote else 0.0,
                    exit_greeks={},
                    notes=f"MARA PAPER TRADING - Hold time: {hold_time:.1f}m"
                )
            except Exception as log_err:
                self._log(f"Error logging trade exit: {log_err}", "WARN")

            self.session.position = None

        except Exception as e:
            self._log(f"Error exiting position: {e}", "ERROR")
            import traceback
            traceback.print_exc()

    def _print_status(self):
        """Print current status."""
        now = datetime.now(EST)
        is_entry = self._is_entry_window()
        is_open = self._is_market_open()

        parts = [
            f"{now.strftime('%H:%M:%S')} EST",
            f"Market: {'OPEN' if is_open else 'CLOSED'}",
        ]

        if is_entry:
            parts.append("ENTRY WINDOW")
        elif now.time() < self.entry_start:
            minutes_to_entry = (datetime.combine(now.date(), self.entry_start) - now).total_seconds() / 60
            parts.append(f"Entry in {minutes_to_entry:.0f}m")
        else:
            parts.append("Entry closed")

        if self.latest_underlying_quote:
            parts.append(f"MARA: ${self.latest_underlying_quote.mid:.2f}")

        if self.session.position:
            pos = self.session.position
            if self.latest_option_quote:
                current = self.latest_option_quote.mid
                pnl_pct = ((current - pos.entry_price) / pos.entry_price) * 100
                pnl_dollars = (current - pos.entry_price) * pos.qty * 100
                parts.append(f"{pos.option_type}: ${current:.2f} ({pnl_pct:+.1f}%)")
                parts.append(f"P&L: ${pnl_dollars:+.2f}")
        else:
            parts.append("No Position")
            parts.append(f"P&L: ${self.session.pnl_today:+.2f}")

        status = " | ".join(parts)
        print(f"\r{status}    ", end='', flush=True)

        self._status_counter += 1
        if self._status_counter % 6 == 0:
            self._print_monitoring_table()

    def _print_monitoring_table(self):
        """Print detailed monitoring table."""
        now = datetime.now(EST)
        print()
        print()
        print("-" * 70)
        print(f"MARA MONITORING @ {now.strftime('%H:%M:%S')} EST")
        print("-" * 70)

        print(f"  Market: {'OPEN' if self._is_market_open() else 'CLOSED'}")
        print(f"  Entry Window: {self.entry_start} - {self.entry_end} EST")
        print(f"  Force Exit: {self.force_exit} EST")

        if self.latest_underlying_quote:
            q = self.latest_underlying_quote
            spread = q.ask - q.bid
            spread_pct = (spread / q.mid * 100) if q.mid > 0 else 0
            print(f"\n  MARA:")
            print(f"    Bid: ${q.bid:.2f} | Ask: ${q.ask:.2f} | Mid: ${q.mid:.2f}")
            print(f"    Spread: ${spread:.2f} ({spread_pct:.2f}%)")

        if self.session.position:
            pos = self.session.position
            alpaca_pos = self.client.get_position_by_symbol(pos.symbol)

            print(f"\n  OPEN POSITION:")
            print(f"    {pos.option_type} @ Strike ${pos.strike:.0f}")

            if alpaca_pos:
                current = alpaca_pos['current_price']
                pnl_pct = alpaca_pos['unrealized_plpc']
                pnl_dollars = alpaca_pos['unrealized_pl']
                print(f"    Entry: ${alpaca_pos['avg_entry_price']:.2f} | Qty: {alpaca_pos['qty']}")
            else:
                current = pos.entry_price
                pnl_pct = 0
                pnl_dollars = 0
                print(f"    Entry: ${pos.entry_price:.2f} | Qty: {pos.qty}")

            if pos.tp_order_id:
                print(f"    TP Order: ON EXCHANGE | SL: Monitored internally")

            hold_minutes = (now - pos.entry_time).total_seconds() / 60
            dist_to_tp = (pos.take_profit_price - current) / current * 100 if current > 0 else 0
            dist_to_sl = (current - pos.stop_loss_price) / current * 100 if current > 0 else 0

            print(f"    Current: ${current:.2f} | P&L: ${pnl_dollars:+.2f} ({pnl_pct:+.2f}%)")
            print(f"    Hold Time: {hold_minutes:.1f}m")
            print(f"    SL ${pos.stop_loss_price:.2f} | TP ${pos.take_profit_price:.2f}")
            print(f"    Distance: SL {dist_to_sl:.1f}% away | TP {dist_to_tp:.1f}% away")
        else:
            print(f"\n  No open position")
            print(f"  Trades today: {self.session.trades_today}/{self.config.max_trades_per_day}")

        print(f"\n  SESSION STATS:")
        print(f"    Wins: {self.session.wins} | Losses: {self.session.losses}")
        win_rate = (self.session.wins / (self.session.wins + self.session.losses) * 100) if (self.session.wins + self.session.losses) > 0 else 0
        print(f"    Win Rate: {win_rate:.0f}%")
        print(f"    P&L: ${self.session.pnl_today:+.2f}")

        print("-" * 70)
        print()

    def run(self):
        """Main trading loop."""
        print()
        self._log("=" * 70)
        self._log("THE VOLUME AI - MARA 0DTE Options Paper Trading")
        self._log("=" * 70)
        self._log(f"Symbol: MARA (Marathon Digital)")
        self._log(f"Position Size: ${self.config.fixed_position_value:,.2f}")
        self._log(f"Max Trades/Day: {self.config.max_trades_per_day}")
        self._log(f"Take Profit: {self.config.target_profit_pct}%")
        self._log(f"Stop Loss: {self.config.stop_loss_pct}%")
        self._log(f"Entry Window: {self.config.entry_time_start} - {self.config.entry_time_end} EST")
        self._log(f"Force Exit: {self.config.force_exit_time} EST")
        self._log("=" * 70)

        try:
            account = self.client.get_account()
            self._log(f"Connected to Alpaca Paper Trading", "SUCCESS")
            self._log(f"Buying Power: ${account['buying_power']:,.2f}")
            self._log(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
        except Exception as e:
            self._log(f"Failed to connect to Alpaca: {e}", "ERROR")
            return

        self._recover_existing_positions()

        self.running = True
        self._log("Starting MARA trading loop... (Ctrl+C to stop)")
        print()

        try:
            while self.running and not self._stop_event.is_set():
                today = datetime.now(EST).date()
                if today != self.session.date:
                    self._log(f"New trading day: {today}")
                    self.session = MARATradingSession(date=today)

                self.latest_underlying_quote = self.client.get_latest_stock_quote("MARA")
                self.latest_underlying_bar = self.client.get_latest_stock_bar("MARA")

                if not self._is_market_open():
                    self._print_status()
                    self._stop_event.wait(timeout=60)
                    continue

                if self.session.position:
                    self._check_exit_conditions()

                elif self._is_entry_window() and self.session.trades_today < self.config.max_trades_per_day:
                    print()
                    self._log("=" * 50)
                    self._log("MARA IN ENTRY WINDOW - Analyzing signal...")
                    self._log("=" * 50)

                    signal = self._calculate_signal()

                    if signal in ['BULLISH', 'BEARISH']:
                        self._log(f"  SIGNAL DETECTED: {signal} - Attempting entry...", "SUCCESS")
                        self._enter_position(signal)
                    else:
                        self._log(f"  Signal is {signal} - waiting...", "WARN")
                    print()

                if self.session.position and self._should_force_exit():
                    self._check_exit_conditions()

                self._print_status()
                self._stop_event.wait(timeout=self.config.poll_interval_seconds)

        except KeyboardInterrupt:
            self._log("\nShutdown requested...")
        finally:
            self.running = False
            self._handle_shutdown()

    def _handle_shutdown(self):
        """Handle shutdown."""
        print()
        self._log("=" * 70)
        self._log("MARA PAPER TRADING SHUTDOWN")
        self._log("=" * 70)

        if self.session.position:
            pos = self.session.position
            option_quote = self.client.get_latest_option_quote(pos.symbol)

            if option_quote:
                current_price = option_quote.mid
                pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                pnl_dollars = (current_price - pos.entry_price) * pos.qty * 100

                print()
                self._log(f"OPEN POSITION: {pos.symbol}", "WARN")
                self._log(f"  Entry: ${pos.entry_price:.2f} | Current: ${current_price:.2f}", "INFO")
                self._log(f"  P&L: ${pnl_dollars:+.2f} ({pnl_pct:+.2f}%)", "INFO")
                print()

                while True:
                    try:
                        response = input("Close position before stopping? (y/n): ").strip().lower()
                        if response in ['y', 'yes']:
                            self._exit_position("USER_SHUTDOWN", current_price, pnl_dollars)
                            break
                        elif response in ['n', 'no']:
                            self._log("Keeping position open (TP order remains active)", "WARN")
                            break
                    except (EOFError, KeyboardInterrupt):
                        self._log("Keeping position open", "WARN")
                        break
        else:
            self._log("No open positions", "INFO")

        print()
        self._log(f"Session P&L: ${self.session.pnl_today:+.2f}", "INFO")
        self._log(f"Wins: {self.session.wins} | Losses: {self.session.losses}", "INFO")
        self._log("MARA paper trading engine stopped.")

    def stop(self):
        """Signal the engine to stop."""
        self.running = False
        self._stop_event.set()
