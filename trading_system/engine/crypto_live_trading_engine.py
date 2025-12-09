"""
Crypto LIVE Trading Engine

!!! WARNING: THIS TRADES WITH REAL MONEY !!!

Real-time LIVE trading execution engine for crypto scalping that:
- Connects to Alpaca LIVE API for real crypto trading
- Executes trades using REAL MONEY
- Uses the CryptoScalping strategy for signals
- Manages multiple concurrent positions
- Tracks P&L in real-time
- Includes daily loss protection
"""

import asyncio
import signal
import sys
import os
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
import pytz
import threading

from ..config.crypto_trading_config import CryptoLiveTradingConfig
from .alpaca_client import AlpacaClient, Quote, Bar, ALPACA_AVAILABLE
from ..strategies.crypto_scalping import (
    CryptoScalping, CryptoScalpingConfig, ALPACA_CRYPTO_SYMBOLS
)
from ..core.models import Bar as CoreBar
from ..analytics.crypto_trade_logger import get_crypto_trade_logger, CryptoTradeLogger
from ..data.crypto_market_collector import get_crypto_market_collector, CryptoMarketCollector


UTC = pytz.UTC


@dataclass
class CryptoPosition:
    """Tracks an open crypto position."""
    symbol: str
    qty: float
    side: str
    entry_price: float
    entry_cost: float
    entry_time: datetime
    signal_score: int

    # SL/TP tracking
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    trailing_stop_price: float = 0.0
    highest_price_since_entry: float = 0.0

    # Order tracking
    entry_order_id: str = ""
    entry_fill_price: float = 0.0
    entry_order_status: str = ""
    exit_order_id: str = ""
    stop_loss_order_id: str = ""
    stop_loss_order_status: str = ""


@dataclass
class CryptoTradingSession:
    """Tracks current trading session state."""
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    trades_total: int = 0
    trades_this_hour: int = 0
    wins: int = 0
    losses: int = 0
    pnl_total: float = 0.0
    pnl_today: float = 0.0  # Daily P&L for loss limit
    positions: Dict[str, CryptoPosition] = field(default_factory=dict)
    last_hour: int = -1
    daily_loss_limit_hit: bool = False


class CryptoLiveTradingEngine:
    """
    Real-time LIVE crypto trading execution engine.

    !!! WARNING: THIS TRADES WITH REAL MONEY !!!

    Uses the CryptoScalping strategy for signal generation
    and executes trades via Alpaca's LIVE trading API.
    """

    def __init__(self, config: CryptoLiveTradingConfig):
        """
        Initialize LIVE crypto trading engine.

        Args:
            config: Live trading configuration
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py package required. Run: pip install alpaca-py")

        self.config = config
        self.client = AlpacaClient(
            api_key=config.api_key,
            api_secret=config.api_secret,
            paper=False  # LIVE TRADING
        )

        # Session state
        self.session = CryptoTradingSession()
        self.running = False
        self._stop_event = threading.Event()

        # Market data cache
        self.latest_quotes: Dict[str, Quote] = {}
        self.latest_bars: Dict[str, Bar] = {}

        # Create strategy config
        self.strategy_config = CryptoScalpingConfig(
            symbols=config.symbols,
            fixed_position_value=config.fixed_position_value,
            target_profit_pct=config.target_profit_pct,
            stop_loss_pct=config.stop_loss_pct,
            trailing_stop_pct=config.trailing_stop_pct,
            use_trailing_stop=config.use_trailing_stop,
            use_time_filter=config.use_time_filter,
            allowed_trading_hours=config.allowed_trading_hours,
            min_entry_score=config.min_entry_score,
            max_concurrent_positions=config.max_concurrent_positions,
        )

        # Initialize strategy
        self.strategy = CryptoScalping(self.strategy_config)

        # Half spread for fee calculation
        self.half_spread = config.taker_fee_pct / 200

        # Callbacks
        self.on_trade: Optional[Callable] = None
        self.on_position_update: Optional[Callable] = None
        self.on_quote: Optional[Callable] = None

        # Trade logger and market data collector
        self.trade_logger: CryptoTradeLogger = get_crypto_trade_logger()
        self.market_collector: CryptoMarketCollector = get_crypto_market_collector()

    def _log(self, msg: str, level: str = "INFO"):
        """Log message with timestamp."""
        now = datetime.now(UTC)
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
            elif level == "DANGER":
                color = "\033[91m\033[1m"  # Bold red for live trading
            reset = "\033[0m"

        # Add LIVE prefix for important messages
        if level in ["TRADE", "SUCCESS", "ERROR"]:
            msg = f"[LIVE] {msg}"

        print(f"{color}[{now.strftime('%H:%M:%S UTC')}] [{level}] {msg}{reset}")

    def _is_trading_hour(self) -> bool:
        """Check if current hour is in allowed trading hours."""
        if not self.config.use_time_filter:
            return True
        current_hour = datetime.now(UTC).hour
        return current_hour in self.config.allowed_trading_hours

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been hit."""
        if self.session.pnl_today <= -self.config.max_daily_loss:
            if not self.session.daily_loss_limit_hit:
                self._log("=" * 70, "DANGER")
                self._log("!!! DAILY LOSS LIMIT HIT !!!", "DANGER")
                self._log(f"Daily P&L: ${self.session.pnl_today:+.2f}", "DANGER")
                self._log(f"Limit: ${self.config.max_daily_loss}", "DANGER")
                self._log("NO NEW POSITIONS WILL BE OPENED TODAY", "DANGER")
                self._log("=" * 70, "DANGER")
                self.session.daily_loss_limit_hit = True
            return True
        return False

    def _can_open_position(self, symbol: str) -> tuple[bool, str]:
        """Check if we can open a new position."""
        # Check daily loss limit first
        if self._check_daily_loss_limit():
            return False, "Daily loss limit hit"

        # Check if already in position
        if symbol in self.session.positions:
            return False, "Already in position"

        # Check max concurrent positions
        if len(self.session.positions) >= self.config.max_concurrent_positions:
            return False, f"Max positions reached ({self.config.max_concurrent_positions})"

        # Check hourly trade limit
        current_hour = datetime.now(UTC).hour
        if current_hour != self.session.last_hour:
            self.session.trades_this_hour = 0
            self.session.last_hour = current_hour

        if self.session.trades_this_hour >= self.config.max_trades_per_day:
            return False, f"Max trades/day reached ({self.config.max_trades_per_day})"

        # Check trading hours
        if not self._is_trading_hour():
            return False, "Outside trading hours"

        return True, "OK"

    def _warm_up_indicators(self):
        """Fetch historical bars to warm up indicators before trading."""
        self._log("[LIVE] Warming up indicators with historical data...")
        warmup_bars = 50  # Need at least 20 for BB, 14 for RSI, etc.

        for symbol in self.config.symbols:
            try:
                self._log(f"  Fetching {warmup_bars} bars for {symbol}...")
                bars = self.client.get_crypto_bars(symbol, '1Min', limit=warmup_bars)

                if bars:
                    # Feed all historical bars to strategy
                    for bar in bars:
                        core_bar = CoreBar(
                            symbol=symbol,
                            timestamp=bar.timestamp,
                            open=bar.open,
                            high=bar.high,
                            low=bar.low,
                            close=bar.close,
                            volume=bar.volume,
                        )
                        self.strategy.on_bar(core_bar)

                    self._log(f"    {symbol}: Fed {len(bars)} bars to strategy", "SUCCESS")

                    # Check if indicators initialized
                    if symbol in self.strategy.symbol_states:
                        state = self.strategy.symbol_states[symbol]
                        indicators = []
                        if state.rsi.initialized:
                            indicators.append(f"RSI={state.rsi.value:.1f}")
                        if state.bb.initialized:
                            indicators.append("BB")
                        if state.stoch.initialized:
                            indicators.append(f"Stoch={state.stoch.k:.1f}")
                        if state.adx.initialized:
                            indicators.append(f"ADX={state.adx.value:.1f}")
                        if state.macd.initialized:
                            indicators.append("MACD")
                        if indicators:
                            self._log(f"    Indicators: {', '.join(indicators)}", "SUCCESS")
                else:
                    self._log(f"    {symbol}: No bars returned", "WARN")

            except Exception as e:
                self._log(f"  Error warming up {symbol}: {e}", "WARN")

        self._log("[LIVE] Indicator warm-up complete!")

    def _resume_existing_positions(self):
        """
        Resume management of existing crypto positions from Alpaca.

        On restart:
        1. Fetches all existing crypto positions from Alpaca
        2. Creates internal position tracking for each
        3. Checks for existing SL orders
        4. Places SL orders if missing
        """
        self._log("[LIVE] Checking for existing positions to resume...")

        try:
            # Get all positions from Alpaca
            alpaca_positions = self.client.get_positions()

            crypto_positions = []
            for pos in alpaca_positions:
                symbol = pos.get('symbol', '')
                # Crypto symbols on Alpaca are like BTCUSD, ETHUSD (no slash)
                # Check if it's one of our trading symbols
                for config_symbol in self.config.symbols:
                    if symbol == config_symbol.replace('/', ''):
                        crypto_positions.append(pos)
                        break

            if not crypto_positions:
                self._log("  No existing crypto positions found", "SUCCESS")
                return

            self._log(f"  Found {len(crypto_positions)} existing crypto position(s)")

            # Get open orders to check for existing SL orders
            open_orders = self.client.get_orders(status='open')

            for pos in crypto_positions:
                alpaca_symbol = pos.get('symbol', '')  # e.g., "BTCUSD"
                # Convert to our format: "BTC/USD"
                our_symbol = alpaca_symbol[:-3] + '/' + alpaca_symbol[-3:]

                qty = float(pos.get('qty', 0))
                entry_price = float(pos.get('avg_entry_price', 0))
                market_value = float(pos.get('market_value', 0))
                unrealized_pl = float(pos.get('unrealized_pl', 0))

                self._log(f"\n  Resuming {our_symbol}:", "TRADE")
                self._log(f"    Qty: {qty}")
                self._log(f"    Entry Price: ${entry_price:.4f}")
                self._log(f"    Market Value: ${market_value:.2f}")
                self._log(f"    Unrealized P&L: ${unrealized_pl:+.2f}")

                # Calculate SL/TP based on entry price
                sl_price = entry_price * (1 - self.config.stop_loss_pct / 100)
                tp_price = entry_price * (1 + self.config.target_profit_pct / 100)
                entry_cost = qty * entry_price

                # Create position object
                position = CryptoPosition(
                    symbol=our_symbol,
                    qty=qty,
                    side='long',
                    entry_price=entry_price,
                    entry_cost=entry_cost,
                    entry_time=datetime.now(UTC),  # Unknown actual entry time
                    signal_score=0,  # Unknown
                    stop_loss_price=sl_price,
                    take_profit_price=tp_price,
                    highest_price_since_entry=entry_price,
                    entry_order_id="RESUMED",
                    entry_fill_price=entry_price,
                    entry_order_status="filled",
                )

                self.session.positions[our_symbol] = position

                self._log(f"    SL: ${sl_price:.4f} (-{self.config.stop_loss_pct}%)")
                self._log(f"    TP: ${tp_price:.4f} (+{self.config.target_profit_pct}%)")

                # Check if SL order already exists
                has_sl_order = False
                for order in open_orders:
                    order_symbol = order.get('symbol', '')
                    order_side = order.get('side', '')
                    order_type = order.get('type', '')

                    if order_symbol == alpaca_symbol and order_side == 'sell':
                        if order_type in ['stop', 'stop_limit']:
                            has_sl_order = True
                            position.stop_loss_order_id = order.get('id', '')
                            position.stop_loss_order_status = order.get('status', '')
                            self._log(f"    SL Order: EXISTS ({order.get('id', '')[:12]}...)", "SUCCESS")
                            break

                # Place SL order if missing
                if not has_sl_order:
                    self._log(f"    SL Order: MISSING - Placing now...", "WARN")
                    try:
                        sl_limit_price = sl_price * 0.985  # 1.5% below stop for volatile crypto
                        sl_order = self.client.submit_crypto_stop_limit_order(
                            symbol=our_symbol,
                            qty=qty,
                            side='sell',
                            stop_price=sl_price,
                            limit_price=sl_limit_price
                        )
                        position.stop_loss_order_id = sl_order.get('id', '')
                        position.stop_loss_order_status = sl_order.get('status', '')
                        self._log(f"    SL Order PLACED: {sl_order.get('id', '')[:12]}...", "SUCCESS")
                    except Exception as sl_e:
                        self._log(f"    ERROR placing SL order: {sl_e}", "ERROR")
                        self._log(f"    Position will be monitored manually", "WARN")

            self._log(f"\n[LIVE] Resumed {len(crypto_positions)} position(s)")

        except Exception as e:
            self._log(f"  Error resuming positions: {e}", "ERROR")

    def _update_market_data(self):
        """Fetch latest quotes for all symbols."""
        for symbol in self.config.symbols:
            try:
                quote = self.client.get_latest_crypto_quote(symbol)
                if quote:
                    self.latest_quotes[symbol] = quote

                    bars = self.client.get_crypto_bars(symbol, '1Min', limit=1)
                    if bars:
                        self.latest_bars[symbol] = bars[-1]

            except Exception as e:
                self._log(f"Error fetching data for {symbol}: {e}", "WARN")

    def _feed_strategy(self):
        """Feed latest bars to strategy for indicator updates."""
        for symbol, bar in self.latest_bars.items():
            core_bar = CoreBar(
                symbol=symbol,
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
            self.strategy.on_bar(core_bar)

    def _check_entry_signals(self):
        """Check for entry signals from strategy."""
        for symbol in self.config.symbols:
            can_open, reason = self._can_open_position(symbol)
            if not can_open:
                continue

            if symbol in self.strategy.symbol_states:
                state = self.strategy.symbol_states[symbol]

                if state.pending_entry_order_id:
                    self._enter_position(symbol, state.entry_score)
                    state.pending_entry_order_id = None

    def _enter_position(self, symbol: str, entry_score: int):
        """Enter a new LIVE position."""
        quote = self.latest_quotes.get(symbol)
        if not quote:
            self._log(f"No quote available for {symbol}", "WARN")
            return

        fill_price = quote.ask * (1 + self.half_spread)
        qty = self.config.fixed_position_value / fill_price

        print()
        self._log("=" * 70)
        self._log(f">>> LIVE ENTRY SIGNAL: {symbol} (Score: {entry_score}/{self.config.min_entry_score})", "TRADE")
        self._log(f"    Bid: ${quote.bid:.4f} | Ask: ${quote.ask:.4f} | Mid: ${quote.mid:.4f}", "TRADE")
        self._log(f"    Qty: {qty:.6f} | Value: ${self.config.fixed_position_value:.2f}", "TRADE")
        self._log(f"    Submitting LIVE BUY order to Alpaca...", "TRADE")

        try:
            order = self.client.submit_crypto_market_order(
                symbol=symbol,
                qty=qty,
                side='buy'
            )

            order_id = order.get('id', 'N/A')
            order_status = order.get('status', 'unknown')
            filled_price = order.get('filled_avg_price')
            filled_qty = order.get('filled_qty', 0)

            self._log(f"    LIVE ORDER SUBMITTED:", "SUCCESS")
            self._log(f"      Order ID: {order_id}", "SUCCESS")
            self._log(f"      Status: {order_status}", "SUCCESS")

            if filled_price:
                self._log(f"      Fill Price: ${filled_price:.4f}", "SUCCESS")
                fill_price = filled_price
            if filled_qty:
                self._log(f"      Filled Qty: {filled_qty}", "SUCCESS")
                # Use actual filled qty instead of requested qty
                qty = float(filled_qty)

            sl_price = fill_price * (1 - self.config.stop_loss_pct / 100)
            tp_price = fill_price * (1 + self.config.target_profit_pct / 100)

            entry_cost = qty * fill_price
            position = CryptoPosition(
                symbol=symbol,
                qty=qty,  # Now uses actual filled qty
                side='long',
                entry_price=fill_price,
                entry_cost=entry_cost,
                entry_time=datetime.now(UTC),
                signal_score=entry_score,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                highest_price_since_entry=fill_price,
                entry_order_id=order_id,
                entry_fill_price=filled_price if filled_price else fill_price,
                entry_order_status=order_status,
            )

            self.session.positions[symbol] = position
            self.session.trades_total += 1
            self.session.trades_this_hour += 1

            self._log(f"    LIVE POSITION OPENED:", "SUCCESS")
            self._log(f"      Take Profit: ${tp_price:.4f} (+{self.config.target_profit_pct}%) - MONITORED", "SUCCESS")
            self._log(f"      Stop Loss: ${sl_price:.4f} (-{self.config.stop_loss_pct}%)", "SUCCESS")

            # Submit SL order
            self._log(f"    Submitting LIVE STOP-LIMIT order...", "TRADE")
            sl_limit_price = sl_price * 0.985  # 1.5% below stop for volatile crypto
            try:
                sl_order = self.client.submit_crypto_stop_limit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    stop_price=sl_price,
                    limit_price=sl_limit_price
                )
                sl_order_id = sl_order.get('id', 'N/A')
                sl_order_status = sl_order.get('status', 'unknown')
                position.stop_loss_order_id = sl_order_id
                position.stop_loss_order_status = sl_order_status
                self._log(f"      SL Order ID: {sl_order_id}", "SUCCESS")
                self._log(f"      SL Status: {sl_order_status}", "SUCCESS")
            except Exception as sl_e:
                self._log(f"      SL ORDER FAILED: {sl_e}", "ERROR")
                self._log(f"      WARNING: SL will be managed manually", "WARN")

            self._log("=" * 70)
            print()

            if self.on_trade:
                self.on_trade('ENTRY', position)

            # Log trade entry for analytics
            try:
                # Get indicator values from strategy state
                indicators = {}
                if symbol in self.strategy.symbol_states:
                    state = self.strategy.symbol_states[symbol]
                    if state.rsi.initialized:
                        indicators['rsi'] = state.rsi.value
                    if state.macd.initialized:
                        indicators['macd'] = state.macd.value
                        indicators['macd_signal'] = state.macd.signal
                        indicators['macd_hist'] = state.macd.histogram
                    if state.bb.initialized:
                        indicators['bb_upper'] = state.bb.upper
                        indicators['bb_middle'] = state.bb.middle
                        indicators['bb_lower'] = state.bb.lower
                    if state.stoch.initialized:
                        indicators['stoch_k'] = state.stoch.k
                        indicators['stoch_d'] = state.stoch.d
                    if state.adx.initialized:
                        indicators['adx'] = state.adx.value
                    if state.atr.initialized:
                        indicators['atr'] = state.atr.value

                self.trade_logger.log_entry(
                    trade_id=order_id,
                    symbol=symbol,
                    entry_time=datetime.now(UTC),
                    entry_price=fill_price,
                    entry_qty=qty,
                    entry_order_id=order_id,
                    indicators=indicators,
                    signal_score=entry_score,
                    target_price=tp_price,
                    stop_loss_price=sl_price,
                    fee_pct=self.config.taker_fee_pct,
                    notes=f"LIVE - Score: {entry_score}/{self.config.min_entry_score}"
                )
            except Exception as log_e:
                self._log(f"    Trade logging error: {log_e}", "WARN")

        except Exception as e:
            self._log(f"    LIVE ORDER FAILED: {e}", "ERROR")
            self._log("=" * 70)
            print()

    def _check_sl_orders_filled(self):
        """Check if any SL orders have been filled."""
        positions_closed_by_sl = []

        for symbol, pos in list(self.session.positions.items()):
            if not pos.stop_loss_order_id:
                continue

            try:
                order = self.client.get_order(pos.stop_loss_order_id)
                if order and order.get('status') == 'filled':
                    fill_price = order.get('filled_avg_price', pos.stop_loss_price)
                    positions_closed_by_sl.append((symbol, fill_price))
            except Exception:
                pass

        for symbol, fill_price in positions_closed_by_sl:
            self._handle_sl_filled(symbol, fill_price)

    def _handle_sl_filled(self, symbol: str, fill_price: float):
        """Handle a position closed by SL on Alpaca."""
        pos = self.session.positions.get(symbol)
        if not pos:
            return

        hold_time = (datetime.now(UTC) - pos.entry_time).total_seconds() / 60

        proceeds = pos.qty * fill_price
        pnl = proceeds - pos.entry_cost
        pnl_pct = pnl / pos.entry_cost * 100

        print()
        self._log("=" * 70)
        self._log(f"<<< LIVE STOP LOSS FILLED: {symbol}", "TRADE")
        self._log(f"    Entry Price: ${pos.entry_price:.4f}", "TRADE")
        self._log(f"    SL Fill Price: ${fill_price:.4f}", "SUCCESS")
        self._log(f"    Hold Time: {hold_time:.1f} minutes", "TRADE")

        log_level = "SUCCESS" if pnl >= 0 else "WARN"
        self._log(f"    RESULT:", log_level)
        self._log(f"      Entry: ${pos.entry_price:.4f} -> Exit: ${fill_price:.4f}", log_level)
        self._log(f"      P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)", log_level)
        self._log(f"      Reason: STOP_LOSS", log_level)

        self.session.pnl_total += pnl
        self.session.pnl_today += pnl
        if pnl > 0:
            self.session.wins += 1
        else:
            self.session.losses += 1

        win_rate = (self.session.wins / (self.session.wins + self.session.losses) * 100) if (self.session.wins + self.session.losses) > 0 else 0
        self._log(f"    SESSION: W/L {self.session.wins}/{self.session.losses} ({win_rate:.0f}%) | Total P&L: ${self.session.pnl_total:+.2f}", "INFO")
        self._log("=" * 70)
        print()

        if self.on_trade:
            self.on_trade('EXIT', pos, 'STOP_LOSS', fill_price, pnl)

        # Log trade exit for analytics
        try:
            exit_indicators = {}
            if symbol in self.strategy.symbol_states:
                state = self.strategy.symbol_states[symbol]
                if state.rsi.initialized:
                    exit_indicators['rsi'] = state.rsi.value
                if state.macd.initialized:
                    exit_indicators['macd'] = state.macd.value

            self.trade_logger.log_exit(
                trade_id=pos.entry_order_id,
                exit_time=datetime.now(UTC),
                exit_price=fill_price,
                exit_qty=pos.qty,
                exit_order_id=pos.stop_loss_order_id,
                exit_reason='STOP_LOSS',
                exit_indicators=exit_indicators,
                fee_pct=self.config.taker_fee_pct,
                notes="LIVE - Alpaca SL order filled"
            )
        except Exception as log_e:
            self._log(f"    Trade logging error: {log_e}", "WARN")

        del self.session.positions[symbol]

        if symbol in self.strategy.symbol_states:
            self.strategy.symbol_states[symbol].reset_position_state()

    def _check_exit_conditions(self):
        """Check exit conditions for all open positions."""
        self._check_sl_orders_filled()

        positions_to_close = []

        for symbol, pos in self.session.positions.items():
            quote = self.latest_quotes.get(symbol)
            if not quote:
                continue

            current_price = quote.mid
            exit_reason = None

            if current_price > pos.highest_price_since_entry:
                pos.highest_price_since_entry = current_price

                if self.config.use_trailing_stop:
                    profit_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                    if profit_pct >= self.config.trailing_stop_pct:
                        new_trailing = current_price * (1 - self.config.trailing_stop_pct / 100)
                        if new_trailing > pos.trailing_stop_price:
                            pos.trailing_stop_price = new_trailing

            exit_price = current_price * (1 - self.half_spread)
            proceeds = pos.qty * exit_price
            pnl = proceeds - pos.entry_cost
            pnl_pct = pnl / pos.entry_cost * 100

            # Check take profit
            if current_price >= pos.take_profit_price:
                exit_reason = "TAKE_PROFIT"

            # Manual SL check
            elif not pos.stop_loss_order_id and current_price <= pos.stop_loss_price:
                exit_reason = "STOP_LOSS_MANUAL"

            # Trailing stop
            elif pos.trailing_stop_price > 0 and current_price <= pos.trailing_stop_price:
                exit_reason = "TRAILING_STOP"

            # Max hold time
            hold_minutes = (datetime.now(UTC) - pos.entry_time).total_seconds() / 60
            if hold_minutes >= self.config.max_hold_minutes:
                exit_reason = "TIME_EXIT"

            # TECHNICAL EXITS DISABLED - Let trades develop to SL/TP only
            # RSI Overbought exit removed per user request

            if exit_reason:
                positions_to_close.append((symbol, exit_reason, exit_price, pnl, pnl_pct))

        for symbol, reason, exit_price, pnl, pnl_pct in positions_to_close:
            self._exit_position(symbol, reason, exit_price, pnl, pnl_pct)

    def _exit_position(self, symbol: str, reason: str, exit_price: float, pnl: float, pnl_pct: float):
        """Exit a LIVE position."""
        pos = self.session.positions.get(symbol)
        if not pos:
            return

        hold_time = (datetime.now(UTC) - pos.entry_time).total_seconds() / 60
        quote = self.latest_quotes.get(symbol)

        print()
        self._log("=" * 70)
        self._log(f"<<< LIVE EXIT SIGNAL: {symbol} - {reason}", "TRADE")
        self._log(f"    Entry Price: ${pos.entry_price:.4f}", "TRADE")
        if quote:
            self._log(f"    Current: Bid=${quote.bid:.4f} | Ask=${quote.ask:.4f} | Mid=${quote.mid:.4f}", "TRADE")
        self._log(f"    Hold Time: {hold_time:.1f} minutes", "TRADE")

        # Cancel SL order first
        if pos.stop_loss_order_id:
            self._log(f"    Cancelling SL order {pos.stop_loss_order_id[:12]}...", "TRADE")
            try:
                cancelled = self.client.cancel_order(pos.stop_loss_order_id)
                if cancelled:
                    self._log(f"      SL order cancelled", "SUCCESS")
                else:
                    self._log(f"      SL already filled/cancelled", "WARN")
            except Exception as cancel_e:
                self._log(f"      Failed to cancel SL: {cancel_e}", "WARN")

        self._log(f"    Submitting LIVE SELL order...", "TRADE")

        try:
            # Get actual position from Alpaca to ensure we sell correct qty
            actual_qty = pos.qty
            try:
                alpaca_positions = self.client.get_positions()
                for ap in alpaca_positions:
                    # Crypto symbols in Alpaca: BTCUSD, ETHUSD etc (no slash)
                    alpaca_symbol = ap.get('symbol', '')
                    our_symbol = symbol.replace('/', '')
                    if alpaca_symbol == our_symbol:
                        actual_qty = float(ap.get('qty', pos.qty))
                        if abs(actual_qty - pos.qty) > 0.000001:
                            self._log(f"    Using Alpaca balance: {actual_qty} (vs tracked: {pos.qty})", "WARN")
                        break
            except Exception as pos_e:
                self._log(f"    Could not verify position: {pos_e}", "WARN")

            order = self.client.submit_crypto_market_order(
                symbol=symbol,
                qty=actual_qty,
                side='sell'
            )

            order_id = order.get('id', 'N/A')
            order_status = order.get('status', 'unknown')
            filled_price = order.get('filled_avg_price')
            filled_qty = order.get('filled_qty', 0)

            self._log(f"    LIVE ORDER SUBMITTED:", "SUCCESS")
            self._log(f"      Order ID: {order_id}", "SUCCESS")
            self._log(f"      Status: {order_status}", "SUCCESS")

            if filled_price:
                self._log(f"      Fill Price: ${filled_price:.4f}", "SUCCESS")
                exit_price = filled_price
                proceeds = pos.qty * filled_price
                pnl = proceeds - pos.entry_cost
                pnl_pct = pnl / pos.entry_cost * 100

            if filled_qty:
                self._log(f"      Filled Qty: {filled_qty}", "SUCCESS")

            pos.exit_order_id = order_id

            log_level = "SUCCESS" if pnl >= 0 else "WARN"
            self._log(f"    RESULT:", log_level)
            self._log(f"      Entry: ${pos.entry_price:.4f} -> Exit: ${exit_price:.4f}", log_level)
            self._log(f"      P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)", log_level)
            self._log(f"      Reason: {reason}", log_level)

            self.session.pnl_total += pnl
            self.session.pnl_today += pnl
            if pnl > 0:
                self.session.wins += 1
            else:
                self.session.losses += 1

            win_rate = (self.session.wins / (self.session.wins + self.session.losses) * 100) if (self.session.wins + self.session.losses) > 0 else 0
            self._log(f"    SESSION: W/L {self.session.wins}/{self.session.losses} ({win_rate:.0f}%) | Total P&L: ${self.session.pnl_total:+.2f} | Today: ${self.session.pnl_today:+.2f}", "INFO")
            self._log("=" * 70)
            print()

            if self.on_trade:
                self.on_trade('EXIT', pos, reason, exit_price, pnl)

            # Log trade exit for analytics
            try:
                exit_indicators = {}
                if symbol in self.strategy.symbol_states:
                    state = self.strategy.symbol_states[symbol]
                    if state.rsi.initialized:
                        exit_indicators['rsi'] = state.rsi.value
                    if state.macd.initialized:
                        exit_indicators['macd'] = state.macd.value

                self.trade_logger.log_exit(
                    trade_id=pos.entry_order_id,
                    exit_time=datetime.now(UTC),
                    exit_price=exit_price,
                    exit_qty=float(filled_qty) if filled_qty else pos.qty,
                    exit_order_id=order_id,
                    exit_reason=reason,
                    exit_indicators=exit_indicators,
                    fee_pct=self.config.taker_fee_pct,
                    notes=f"LIVE - Hold: {hold_time:.1f}m"
                )
            except Exception as log_e:
                self._log(f"    Trade logging error: {log_e}", "WARN")

            del self.session.positions[symbol]

            if symbol in self.strategy.symbol_states:
                self.strategy.symbol_states[symbol].reset_position_state()

        except Exception as e:
            self._log(f"    LIVE ORDER FAILED: {e}", "ERROR")
            self._log("=" * 70)
            print()

    def _get_symbol_scores(self) -> Dict[str, int]:
        """Get current entry score for each symbol."""
        scores = {}
        for symbol in self.config.symbols:
            if symbol in self.strategy.symbol_states:
                state = self.strategy.symbol_states[symbol]
                score = 0
                if state.rsi.initialized and state.rsi.value <= 30:
                    score += 1
                if state.bb.initialized and state.bb.lower > 0:
                    quote = self.latest_quotes.get(symbol)
                    if quote and quote.mid <= state.bb.lower * 1.01:
                        score += 1
                if state.stoch.initialized and state.stoch.k <= 20:
                    score += 1
                if state.adx.initialized and state.adx.value >= 20:
                    score += 1
                if state.macd.initialized and state.macd.histogram > 0:
                    score += 1
                if hasattr(state, 'volume_ma') and state.volume_ma.initialized:
                    score += 1
                scores[symbol] = score
            else:
                scores[symbol] = 0
        return scores

    def _print_status(self):
        """Print current status."""
        now = datetime.now(UTC)
        is_trading_hour = self._is_trading_hour()

        if is_trading_hour:
            hours_status = "TRADING"
        else:
            hours_status = "OUTSIDE HOURS"
            current_hour = now.hour
            next_hours = [h for h in self.config.allowed_trading_hours if h > current_hour]
            if next_hours:
                next_start = next_hours[0]
            else:
                next_start = self.config.allowed_trading_hours[0]
            hours_until = (next_start - current_hour) % 24
            hours_status += f" (next in {hours_until}h)"

        # Daily loss status
        loss_status = ""
        if self.session.daily_loss_limit_hit:
            loss_status = " | LOSS LIMIT HIT"
        elif self.session.pnl_today < 0:
            remaining = self.config.max_daily_loss + self.session.pnl_today
            loss_status = f" | Loss limit in ${remaining:.0f}"

        parts = [
            "[LIVE]",
            f"{now.strftime('%H:%M:%S')} UTC",
            hours_status,
            f"Pos: {len(self.session.positions)}/{self.config.max_concurrent_positions}",
            f"Trades: {self.session.trades_total}",
            f"W/L: {self.session.wins}/{self.session.losses}",
            f"P&L: ${self.session.pnl_total:+.2f}",
            f"Today: ${self.session.pnl_today:+.2f}{loss_status}",
        ]

        if self.session.positions:
            pos_details = []
            # Get actual P&L from Alpaca for accuracy
            try:
                alpaca_positions = self.client.get_positions()
                alpaca_pnl = {}
                for ap in alpaca_positions:
                    ap_symbol = ap.get('symbol', '')  # e.g., "AVAXUSD"
                    # Convert to our format: "AVAX/USD"
                    if len(ap_symbol) > 3:
                        our_symbol = ap_symbol[:-3] + '/' + ap_symbol[-3:]
                        # alpaca_client already converts to % (multiplied by 100)
                        unrealized_plpc = float(ap.get('unrealized_plpc', 0))
                        alpaca_pnl[our_symbol] = unrealized_plpc
            except Exception:
                alpaca_pnl = {}

            for symbol, pos in self.session.positions.items():
                # Prefer Alpaca's P&L percentage (more accurate)
                if symbol in alpaca_pnl:
                    pnl_pct = alpaca_pnl[symbol]
                else:
                    # Fallback to calculated P&L from quote
                    quote = self.latest_quotes.get(symbol)
                    if quote and pos.entry_price > 0:
                        pnl_pct = (quote.mid - pos.entry_price) / pos.entry_price * 100
                    else:
                        pnl_pct = 0.0
                pos_details.append(f"{symbol.split('/')[0]}:{pnl_pct:+.1f}%")
            if pos_details:
                parts.append(f"[{' | '.join(pos_details)}]")

        status = " | ".join(parts)
        print(f"\r{status}    ", end='', flush=True)

        if hasattr(self, '_status_counter'):
            self._status_counter += 1
        else:
            self._status_counter = 0

        if self._status_counter % 6 == 0:
            self._print_scores_table()
            self._print_positions_table()

    def _print_scores_table(self):
        """Print confirmation scores table."""
        from trading_system.strategies.crypto_scalping import SYMBOL_RISK_PARAMS

        scores = self._get_symbol_scores()
        default_min_score = self.config.min_entry_score

        print()
        print(f"\n--- [LIVE] Signal Scores (per-symbol thresholds) ---")

        ready = []
        close = []
        waiting = []

        for symbol, score in scores.items():
            short_sym = symbol.split('/')[0]
            quote = self.latest_quotes.get(symbol)
            price_str = f"${quote.mid:,.2f}" if quote else "N/A"

            # Get per-symbol min_entry_score
            symbol_params = SYMBOL_RISK_PARAMS.get(symbol, {})
            min_score = symbol_params.get('min_entry_score', default_min_score)

            if score >= min_score:
                ready.append(f"  {short_sym}: {score}/{min_score} READY @ {price_str}")
            elif score >= min_score - 2:
                close.append(f"  {short_sym}: {score}/{min_score} @ {price_str}")
            else:
                waiting.append(f"  {short_sym}: {score}/{min_score} @ {price_str}")

        if ready:
            print("SIGNALS READY:")
            for r in ready:
                print(r)
        if close:
            print("CLOSE (2 away):")
            for c in close:
                print(c)
        if waiting:
            print("WAITING:")
            for w in waiting:
                print(w)

        print("-" * 40)
        print()

    def _print_positions_table(self):
        """Print positions table."""
        if not self.session.positions:
            return

        print()
        print("=" * 70)
        print("[LIVE] OPEN POSITIONS - SL/TP MONITORING")
        print("=" * 70)

        for symbol, pos in self.session.positions.items():
            quote = self.latest_quotes.get(symbol)
            if not quote:
                continue

            current_price = quote.mid
            hold_minutes = (datetime.now(UTC) - pos.entry_time).total_seconds() / 60

            dist_to_tp = (pos.take_profit_price - current_price) / current_price * 100
            dist_to_sl = (current_price - pos.stop_loss_price) / current_price * 100

            exit_price = current_price * (1 - self.half_spread)
            proceeds = pos.qty * exit_price
            pnl = proceeds - pos.entry_cost
            pnl_pct = pnl / pos.entry_cost * 100

            price_range = pos.take_profit_price - pos.stop_loss_price
            if price_range > 0:
                progress = (current_price - pos.stop_loss_price) / price_range
                progress = max(0, min(1, progress))
                bar_width = 20
                filled = int(progress * bar_width)
                bar = "#" * filled + "-" * (bar_width - filled)
            else:
                bar = "-" * 20

            short_sym = symbol.split('/')[0]
            print(f"\n  {short_sym} | Entry: ${pos.entry_price:.2f} | Now: ${current_price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
            print(f"  Entry Order: {pos.entry_order_id[:12]}...")
            print(f"  SL ${pos.stop_loss_price:.2f} [{bar}] TP ${pos.take_profit_price:.2f}")
            print(f"  Distance: SL {dist_to_sl:.2f}% away | TP {dist_to_tp:.2f}% away")

            if pos.stop_loss_order_id:
                print(f"  SL Order: {pos.stop_loss_order_id[:12]}... (ON ALPACA)")
            else:
                print(f"  SL Order: MANUAL MONITORING")

            if self.config.use_trailing_stop:
                if pos.trailing_stop_price > 0:
                    trail_dist = (current_price - pos.trailing_stop_price) / current_price * 100
                    print(f"  Trailing Stop: ACTIVE at ${pos.trailing_stop_price:.2f} ({trail_dist:.2f}% away)")
                else:
                    print(f"  Trailing Stop: PENDING (activates at +{self.config.trailing_stop_pct}%)")

            remaining = self.config.max_hold_minutes - hold_minutes
            if remaining < 10:
                print(f"  Hold Time: {hold_minutes:.0f}m / {self.config.max_hold_minutes}m - TIME EXIT in {remaining:.0f}m!")
            else:
                print(f"  Hold Time: {hold_minutes:.0f}m / {self.config.max_hold_minutes}m")

        print()
        print("=" * 70)
        print()

    def run(self):
        """Main LIVE trading loop."""
        self._log("=" * 70, "DANGER")
        self._log("THE VOLUME AI - Crypto LIVE Trading Engine", "DANGER")
        self._log("!!! TRADING WITH REAL MONEY !!!", "DANGER")
        self._log("=" * 70, "DANGER")
        self._log(f"Symbols: {', '.join(self.config.symbols)}")
        self._log(f"Position Size: ${self.config.fixed_position_value:,.2f}")
        self._log(f"Max Positions: {self.config.max_concurrent_positions}")
        self._log(f"TP/SL: Per-symbol (V6.1)")
        self._log(f"Exit Modes: TP, SL, Trailing Stop only (V6)")
        self._log(f"Daily Loss Limit: ${self.config.max_daily_loss}")
        self._log("=" * 70)

        # Verify LIVE connection
        try:
            account = self.client.get_account()
            self._log("Connected to Alpaca LIVE Trading", "SUCCESS")
            self._log(f"Account ID: {account['id']}")
            self._log(f"Cash: ${account['cash']:,.2f}")
            self._log(f"Buying Power: ${account['buying_power']:,.2f}")
            self._log(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
        except Exception as e:
            self._log(f"Failed to connect to Alpaca LIVE: {e}", "ERROR")
            return

        self.strategy.on_start()

        # Warm up indicators with historical data
        self._warm_up_indicators()

        # Resume any existing positions from Alpaca and ensure SL orders exist
        self._resume_existing_positions()

        self.running = True
        self._log("Starting LIVE trading loop... (Ctrl+C to stop)")
        print()

        try:
            iteration = 0
            while self.running and not self._stop_event.is_set():
                iteration += 1

                self._update_market_data()
                self._feed_strategy()
                self._check_exit_conditions()
                self._check_entry_signals()
                self._print_status()

                self._stop_event.wait(timeout=10)

        except KeyboardInterrupt:
            print()
            self._log("Shutdown requested...")
        finally:
            self.running = False

            if self.session.positions:
                print()
                self._log(f"Closing {len(self.session.positions)} open LIVE positions...")
                for symbol in list(self.session.positions.keys()):
                    quote = self.latest_quotes.get(symbol)
                    if quote:
                        pos = self.session.positions[symbol]
                        exit_price = quote.mid * (1 - self.half_spread)
                        proceeds = pos.qty * exit_price
                        pnl = proceeds - pos.entry_cost
                        pnl_pct = pnl / pos.entry_cost * 100
                        self._exit_position(symbol, "SHUTDOWN", exit_price, pnl, pnl_pct)

            self.strategy.on_stop()
            self._log("LIVE trading engine stopped.")
            self._print_session_summary()

    def stop(self):
        """Signal the engine to stop."""
        self.running = False
        self._stop_event.set()

    def _print_session_summary(self):
        """Print session summary."""
        print()
        self._log("=" * 70)
        self._log("[LIVE] SESSION SUMMARY")
        self._log("=" * 70)

        duration = datetime.now(UTC) - self.session.start_time
        hours = duration.total_seconds() / 3600

        self._log(f"Duration: {hours:.1f} hours")
        self._log(f"Total Trades: {self.session.trades_total}")
        self._log(f"Wins: {self.session.wins}")
        self._log(f"Losses: {self.session.losses}")

        if self.session.trades_total > 0:
            win_rate = self.session.wins / self.session.trades_total * 100
            self._log(f"Win Rate: {win_rate:.1f}%")

        self._log(f"Total P&L: ${self.session.pnl_total:+.2f}")
        self._log(f"Today's P&L: ${self.session.pnl_today:+.2f}")
        if self.session.daily_loss_limit_hit:
            self._log("Daily loss limit was hit during this session", "WARN")
        self._log("=" * 70)


def run_crypto_live_trading(config: CryptoLiveTradingConfig):
    """Run the LIVE crypto trading engine with given config."""
    engine = CryptoLiveTradingEngine(config)

    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        engine.stop()

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    engine.run()
