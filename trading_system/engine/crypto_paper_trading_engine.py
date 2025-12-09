"""
Crypto Paper Trading Engine

Real-time paper trading execution engine for crypto scalping that:
- Connects to Alpaca for live crypto market data
- Executes trades using the paper trading API
- Uses the CryptoScalping strategy for signals
- Manages multiple concurrent positions
- Tracks P&L in real-time
"""

import asyncio
import signal
import sys
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
import pytz
import threading

from ..config.crypto_paper_trading_config import CryptoPaperTradingConfig
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
    symbol: str  # e.g., 'BTC/USD'
    qty: float  # Fractional quantity allowed
    side: str  # 'long' only for now (buy low, sell high)
    entry_price: float
    entry_cost: float  # Total cost including spread
    entry_time: datetime
    signal_score: int  # Entry score that triggered the trade

    # SL/TP tracking
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    trailing_stop_price: float = 0.0
    highest_price_since_entry: float = 0.0

    # Order tracking
    entry_order_id: str = ""
    entry_fill_price: float = 0.0  # Actual fill price from Alpaca
    entry_order_status: str = ""  # Order status
    exit_order_id: str = ""  # For tracking exit orders
    stop_loss_order_id: str = ""  # SL order placed on Alpaca
    stop_loss_order_status: str = ""  # Status of SL order


@dataclass
class CryptoTradingSession:
    """Tracks current trading session state."""
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    trades_total: int = 0
    trades_this_hour: int = 0
    wins: int = 0
    losses: int = 0
    pnl_total: float = 0.0
    positions: Dict[str, CryptoPosition] = field(default_factory=dict)
    last_hour: int = -1


class CryptoPaperTradingEngine:
    """
    Real-time crypto paper trading execution engine.

    Uses the CryptoScalping strategy for signal generation
    and executes trades via Alpaca's paper trading API.
    """

    def __init__(self, config: CryptoPaperTradingConfig):
        """
        Initialize crypto paper trading engine.

        Args:
            config: Crypto paper trading configuration
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
        self.session = CryptoTradingSession()
        self.running = False
        self._stop_event = threading.Event()

        # Market data cache
        self.latest_quotes: Dict[str, Quote] = {}
        self.latest_bars: Dict[str, Bar] = {}

        # V9.1: M0 (15-min trend) cache - prevents flip-flopping
        # Only update when a NEW 15-min bar completes
        self._m0_cache: Dict[str, dict] = {}  # symbol -> trend result
        self._m0_last_bar_time: Dict[str, datetime] = {}  # symbol -> last bar timestamp

        # V10: 1H Macro context cache - update every hour
        self._macro_cache: Dict[str, dict] = {}  # symbol -> macro context
        self._macro_last_bar_time: Dict[str, datetime] = {}  # symbol -> last 1H bar timestamp

        # Create strategy config from paper trading config
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

        # Half spread for fee calculation (0.25% total = 0.125% each side)
        self.half_spread = config.taker_fee_pct / 200  # Convert to decimal and halve

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
            reset = "\033[0m"

        print(f"{color}[{now.strftime('%H:%M:%S UTC')}] [{level}] {msg}{reset}")

    def _is_trading_hour(self) -> bool:
        """Check if current hour is in allowed trading hours."""
        if not self.config.use_time_filter:
            return True
        current_hour = datetime.now(UTC).hour
        return current_hour in self.config.allowed_trading_hours

    def _can_open_position(self, symbol: str) -> tuple[bool, str]:
        """Check if we can open a new position."""
        # Check if already in position for this symbol
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

        if self.session.trades_this_hour >= self.config.max_trades_per_hour:
            return False, f"Max trades/hour reached ({self.config.max_trades_per_hour})"

        # Check if in trading hours
        if not self._is_trading_hour():
            return False, "Outside trading hours"

        return True, "OK"

    def _warm_up_indicators(self):
        """Fetch historical bars to warm up indicators before trading."""
        self._log("Warming up indicators with historical data...")
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

        self._log("Indicator warm-up complete!")

    def _resume_existing_positions(self):
        """
        Resume management of existing crypto positions from Alpaca.

        On restart:
        1. Fetches all existing crypto positions from Alpaca
        2. Creates internal position tracking for each
        3. Checks for existing SL orders
        4. Places SL orders if missing
        """
        self._log("Checking for existing positions to resume...")

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

            self._log(f"\nResumed {len(crypto_positions)} position(s)")

        except Exception as e:
            self._log(f"  Error resuming positions: {e}", "ERROR")

    def _update_market_data(self):
        """Fetch latest quotes for all symbols."""
        for symbol in self.config.symbols:
            try:
                quote = self.client.get_latest_crypto_quote(symbol)
                if quote:
                    self.latest_quotes[symbol] = quote

                    # Also get recent bar for strategy
                    bars = self.client.get_crypto_bars(
                        symbol, '1Min', limit=1
                    )
                    if bars:
                        self.latest_bars[symbol] = bars[-1]

            except Exception as e:
                self._log(f"Error fetching data for {symbol}: {e}", "WARN")

    def _feed_strategy(self):
        """
        V10.5: Feed latest bars to strategy for indicator updates.

        Also fetches and provides multi-timeframe bars to strategy
        so it can perform V10 layer validation internally.
        """
        for symbol, bar in self.latest_bars.items():
            # V10.5: Provide multi-timeframe bars to strategy BEFORE on_bar
            # This allows the strategy to perform V10 layer checking
            if symbol in self.strategy.symbol_states:
                state = self.strategy.symbol_states[symbol]

                # Fetch 5-min bars for M2 (Price Action)
                try:
                    bars_5min = self.client.get_crypto_bars(symbol, '5Min', limit=15)
                    state.bars_5min = bars_5min
                except Exception:
                    state.bars_5min = None

                # Fetch 15-min bars for M0 (Master Trend) - use cached if available
                try:
                    bars_15min = self.client.get_crypto_bars(symbol, '15Min', limit=30)
                    state.bars_15min = bars_15min
                except Exception:
                    state.bars_15min = None

                # Fetch 1H bars for MACRO context - use cached if available
                try:
                    bars_1h = self.client.get_crypto_bars(symbol, '1Hour', limit=220)
                    state.bars_1h = bars_1h
                except Exception:
                    state.bars_1h = None

            # Convert to CoreBar format
            core_bar = CoreBar(
                symbol=symbol,
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
            # Update strategy state
            self.strategy.on_bar(core_bar)

    def _check_entry_signals(self):
        """
        V10.5: Check for entry signals from strategy's pending_entry_order_id.

        The STRATEGY is responsible for all V10 signal layer checking:
          - MACRO (1H): Market bias/context
          - M0: Master Trend (15-min)
          - M1: Technical Score (1-min)
          - M2: Price Action (5-min)

        The ENGINE simply checks if strategy has set pending_entry_order_id,
        which means all V10 layers have been validated by the strategy.

        This keeps the engine strategy-agnostic and reusable.
        """
        for symbol in self.config.symbols:
            can_open, reason = self._can_open_position(symbol)
            if not can_open:
                continue

            # V10.5: Check if strategy has signaled an entry
            if symbol in self.strategy.symbol_states:
                state = self.strategy.symbol_states[symbol]

                # Strategy sets pending_entry_order_id when ALL V10 layers align
                if state.pending_entry_order_id is not None:
                    entry_score = state.entry_score

                    # Get V10 context for logging (strategy already validated these)
                    master_trend_result = self._check_master_trend_signal(symbol)
                    price_action_result = self._check_price_action_signal(symbol)
                    macro_context = self._check_macro_context(symbol)

                    self._log(f">>> V10.5 STRATEGY ENTRY SIGNAL for {symbol}:", "TRADE")
                    self._log(f"    Entry Score: {entry_score}", "SUCCESS")

                    # Execute entry (strategy has already validated all layers)
                    self._enter_position(symbol, entry_score, price_action_result, master_trend_result, macro_context)

                    # Clear the pending entry flag
                    state.pending_entry_order_id = None

    def _check_master_trend_signal(self, symbol: str) -> dict:
        """
        Fetch 15-minute bars and calculate MASTER TREND signal (METHOD 0).

        V9.1: CACHING - Only recalculate when a NEW 15-min bar completes.
        This prevents flip-flopping caused by the incomplete current bar.

        V8: The master trend represents the REAL market direction.
        Only trade when M0 = UP (crypto spot = long only).

        Returns dict with trend, strength, score, and reasons.
        """
        try:
            # Fetch 15-minute bars for master trend analysis
            bars_15min = self.client.get_crypto_bars(symbol, '15Min', limit=25)

            if not bars_15min or len(bars_15min) < 20:
                self._log(f"    Not enough 15-min bars for {symbol} ({len(bars_15min) if bars_15min else 0})", "WARN")
                return {
                    'trend': 'NEUTRAL',
                    'strength': 'WEAK',
                    'score': 0,
                    'bullish_score': 0,
                    'bearish_score': 0,
                    'ema20_slope': 0.0,
                    'price_vs_ema': 'N/A',
                    'reasons': ['Insufficient 15-min bars']
                }

            # V9.1: Check if we should use cached result
            # Get the SECOND-TO-LAST bar (the last COMPLETED one) - skip the current incomplete bar
            completed_bar = bars_15min[-2] if len(bars_15min) >= 2 else bars_15min[-1]
            completed_bar_time = getattr(completed_bar, 'timestamp', None) or getattr(completed_bar, 't', None)

            # Check if we have a cached result and the bar hasn't changed
            if symbol in self._m0_cache and symbol in self._m0_last_bar_time:
                last_bar_time = self._m0_last_bar_time[symbol]
                if completed_bar_time and last_bar_time == completed_bar_time:
                    # Same 15-min bar - use cached result
                    return self._m0_cache[symbol]

            # New 15-min bar completed OR no cache - recalculate
            from ..strategies.crypto_scalping import CryptoScalping
            result = CryptoScalping.calculate_master_trend_signal(bars_15min)

            # Cache the result
            self._m0_cache[symbol] = result
            if completed_bar_time:
                self._m0_last_bar_time[symbol] = completed_bar_time

            return result

        except Exception as e:
            self._log(f"    Error fetching 15-min bars for {symbol}: {e}", "ERROR")
            return {
                'trend': 'NEUTRAL',
                'strength': 'WEAK',
                'score': 0,
                'bullish_score': 0,
                'bearish_score': 0,
                'ema20_slope': 0.0,
                'price_vs_ema': 'N/A',
                'reasons': [f'Error: {str(e)}']
            }

    def _check_price_action_signal(self, symbol: str) -> dict:
        """
        Fetch 5-minute bars and calculate price action signal (METHOD 2).

        Returns dict with signal, strength, bullish/bearish points, and reasons.
        """
        try:
            # Fetch 5-minute bars for price action analysis
            bars_5min = self.client.get_crypto_bars(symbol, '5Min', limit=15)

            if not bars_5min or len(bars_5min) < 10:
                self._log(f"    Not enough 5-min bars for {symbol} ({len(bars_5min) if bars_5min else 0})", "WARN")
                return {
                    'signal': 'NEUTRAL',
                    'strength': 'WEAK',
                    'bullish_points': 0,
                    'bearish_points': 0,
                    'reasons': ['Insufficient 5-min bars']
                }

            # Use static method from CryptoScalping strategy
            from ..strategies.crypto_scalping import CryptoScalping
            result = CryptoScalping.calculate_price_action_signal(bars_5min)

            return result

        except Exception as e:
            self._log(f"    Error fetching 5-min bars for {symbol}: {e}", "ERROR")
            return {
                'signal': 'NEUTRAL',
                'strength': 'WEAK',
                'bullish_points': 0,
                'bearish_points': 0,
                'reasons': [f'Error: {str(e)}']
            }

    def _check_macro_context(self, symbol: str) -> dict:
        """
        V10: Fetch 1-hour bars and calculate MACRO CONTEXT for market bias.

        This provides CONTEXT not a hard filter:
        - Trend direction from 50/200 EMA
        - Market regime (trending vs ranging)
        - S/R levels for awareness
        - Score adjustment (-2 to +2) for M1 entries

        Returns dict with bias, regime, score_adjustment, etc.
        """
        try:
            # Fetch 1-hour bars for macro context
            bars_1h = self.client.get_crypto_bars(symbol, '1Hour', limit=220)

            if not bars_1h or len(bars_1h) < 20:
                return {
                    'bias': 'NEUTRAL',
                    'regime': 'UNKNOWN',
                    'trend_strength': 0.0,
                    'ema50': 0.0,
                    'ema200': 0.0,
                    'price_vs_ema50': 'N/A',
                    'price_vs_ema200': 'N/A',
                    'adx': 0.0,
                    'support': 0.0,
                    'resistance': 0.0,
                    'score_adjustment': 0,
                    'reasons': ['Insufficient 1H bars']
                }

            # V10: Check if we should use cached result (update every hour)
            completed_bar = bars_1h[-2] if len(bars_1h) >= 2 else bars_1h[-1]
            completed_bar_time = getattr(completed_bar, 'timestamp', None) or getattr(completed_bar, 't', None)

            # Check if we have a cached result and the bar hasn't changed
            if symbol in self._macro_cache and symbol in self._macro_last_bar_time:
                last_bar_time = self._macro_last_bar_time[symbol]
                if completed_bar_time and last_bar_time == completed_bar_time:
                    # Same 1H bar - use cached result
                    return self._macro_cache[symbol]

            # New 1H bar completed OR no cache - recalculate
            from ..strategies.crypto_scalping import CryptoScalping
            result = CryptoScalping.calculate_macro_context(bars_1h)

            # Cache the result
            self._macro_cache[symbol] = result
            if completed_bar_time:
                self._macro_last_bar_time[symbol] = completed_bar_time

            return result

        except Exception as e:
            self._log(f"    Error fetching 1H bars for {symbol}: {e}", "ERROR")
            return {
                'bias': 'NEUTRAL',
                'regime': 'UNKNOWN',
                'trend_strength': 0.0,
                'ema50': 0.0,
                'ema200': 0.0,
                'price_vs_ema50': 'N/A',
                'price_vs_ema200': 'N/A',
                'adx': 0.0,
                'support': 0.0,
                'resistance': 0.0,
                'score_adjustment': 0,
                'reasons': [f'Error: {str(e)}']
            }

    def _enter_position(self, symbol: str, entry_score: int, price_action_result: dict = None, master_trend_result: dict = None, macro_context: dict = None):
        """Enter a new position with V8 triple signal confirmation."""
        quote = self.latest_quotes.get(symbol)
        if not quote:
            self._log(f"No quote available for {symbol}", "WARN")
            return

        # Calculate quantity (buy at ask + half spread)
        fill_price = quote.ask * (1 + self.half_spread)
        qty = self.config.fixed_position_value / fill_price

        print()  # New line for better visibility
        self._log("=" * 60)
        self._log(f">>> V10 TRIPLE SIGNAL ENTRY: {symbol}", "TRADE")
        # V10: Show 1H Macro Context first (bias layer)
        if macro_context:
            bias = macro_context.get('bias', 'N/A')
            regime = macro_context.get('regime', 'N/A')
            adj = macro_context.get('score_adjustment', 0)
            adj_str = f"+{adj}" if adj > 0 else str(adj)
            self._log(f"    MACRO (1H): {bias} | {regime} | Score Adj: {adj_str}", "INFO")
        if master_trend_result:
            self._log(f"    M0 (15-min Trend): {master_trend_result['trend']} ({master_trend_result['strength']}) EMA slope: {master_trend_result['ema20_slope']:+.2f}%", "SUCCESS")
        self._log(f"    M1 (Technical): Score {entry_score}/{self.config.min_entry_score}", "SUCCESS")
        if price_action_result:
            self._log(f"    M2 (Price Action): {price_action_result['signal']} ({price_action_result['strength']}) {price_action_result['bullish_points']}B/{price_action_result['bearish_points']}R", "SUCCESS")
        self._log(f"    Bid: ${quote.bid:.4f} | Ask: ${quote.ask:.4f} | Mid: ${quote.mid:.4f}", "TRADE")
        self._log(f"    Qty: {qty:.6f} | Value: ${self.config.fixed_position_value:.2f}", "TRADE")
        self._log(f"    Submitting BUY order to Alpaca...", "TRADE")

        try:
            # Submit market order
            order = self.client.submit_crypto_market_order(
                symbol=symbol,
                qty=qty,
                side='buy'
            )

            order_id = order.get('id', 'N/A')
            order_status = order.get('status', 'unknown')
            filled_price = order.get('filled_avg_price')
            filled_qty = order.get('filled_qty', 0)

            self._log(f"    ORDER SUBMITTED:", "SUCCESS")
            self._log(f"      Order ID: {order_id}", "SUCCESS")
            self._log(f"      Status: {order_status}", "SUCCESS")

            # Wait for order to be filled before placing SL order
            # This prevents "wash trade" and "insufficient balance" errors
            if order_status in ['pending_new', 'new', 'accepted', 'partially_filled']:
                self._log(f"      Waiting for order to fill...", "INFO")
                import time as time_module
                max_wait = 10  # seconds
                wait_interval = 0.5
                elapsed = 0
                while elapsed < max_wait:
                    time_module.sleep(wait_interval)
                    elapsed += wait_interval
                    try:
                        updated_order = self.client.get_order(order_id)
                        order_status = updated_order.get('status', 'unknown')
                        filled_price = updated_order.get('filled_avg_price')
                        filled_qty = updated_order.get('filled_qty', 0)
                        if order_status == 'filled':
                            self._log(f"      Order FILLED after {elapsed:.1f}s", "SUCCESS")
                            break
                        elif order_status in ['cancelled', 'expired', 'rejected']:
                            self._log(f"      Order {order_status} - aborting entry", "ERROR")
                            return
                    except Exception as poll_e:
                        self._log(f"      Poll error: {poll_e}", "WARN")

                if order_status != 'filled':
                    self._log(f"      Order still {order_status} after {max_wait}s - will retry SL later", "WARN")

            if filled_price:
                self._log(f"      Fill Price: ${float(filled_price):.4f}", "SUCCESS")
                fill_price = float(filled_price)  # Use actual fill price
            if filled_qty:
                self._log(f"      Filled Qty: {filled_qty}", "SUCCESS")
                # Use actual filled qty instead of requested qty
                qty = float(filled_qty)

            # Calculate SL/TP prices based on fill price
            sl_price = fill_price * (1 - self.config.stop_loss_pct / 100)
            tp_price = fill_price * (1 + self.config.target_profit_pct / 100)

            # Create position tracking (now uses actual filled qty)
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

            self._log(f"    POSITION OPENED:", "SUCCESS")
            self._log(f"      Take Profit: ${tp_price:.4f} (+{self.config.target_profit_pct}%) - MONITORED", "SUCCESS")
            self._log(f"      Stop Loss: ${sl_price:.4f} (-{self.config.stop_loss_pct}%)", "SUCCESS")

            # Submit SL order to Alpaca (Stop-Limit order - only type supported for crypto)
            self._log(f"    Submitting STOP-LIMIT order to Alpaca...", "TRADE")
            # Set limit price slightly below stop to ensure fill
            sl_limit_price = sl_price * 0.985  # 1.5% below stop for volatile crypto

            # Get actual position quantity from Alpaca (may be less due to fees)
            actual_qty = self._get_actual_position_qty(symbol)
            sl_qty = actual_qty if actual_qty and actual_qty > 0 else qty
            if actual_qty and actual_qty != qty:
                self._log(f"      Using actual position qty: {sl_qty:.6f} (vs order: {qty:.6f})", "INFO")
                position.qty = sl_qty  # Update position qty to actual

            try:
                sl_order = self.client.submit_crypto_stop_limit_order(
                    symbol=symbol,
                    qty=sl_qty,
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
                self._log(f"      Stop: ${sl_price:.2f} | Limit: ${sl_limit_price:.2f}", "SUCCESS")
            except Exception as sl_e:
                self._log(f"      SL ORDER FAILED: {sl_e}", "ERROR")
                self._log(f"      WARNING: Will retry on next loop", "WARN")

            if self.config.use_trailing_stop:
                self._log(f"      Trailing Stop: {self.config.trailing_stop_pct}% (activates when in profit)", "SUCCESS")
            self._log(f"    NOTE: TP monitored manually | SL order placed on Alpaca", "INFO")
            self._log("=" * 60)
            print()  # Extra line after entry

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
                    notes=f"Score: {entry_score}/{self.config.min_entry_score}"
                )
            except Exception as log_e:
                self._log(f"    Trade logging error: {log_e}", "WARN")

        except Exception as e:
            self._log(f"    ORDER FAILED: {e}", "ERROR")
            self._log("=" * 60)
            print()

    def _get_actual_position_qty(self, symbol: str) -> Optional[float]:
        """Get the actual position quantity from Alpaca."""
        try:
            positions = self.client.get_positions()
            # Symbol format: SOL/USD -> SOLUSD
            alpaca_symbol = symbol.replace('/', '')
            for pos in positions:
                if pos.get('symbol') == alpaca_symbol:
                    return float(pos.get('qty', 0))
            return None
        except Exception as e:
            self._log(f"    Error getting position qty for {symbol}: {e}", "WARN")
            return None

    def _retry_missing_sl_orders(self):
        """Retry placing SL orders for positions that don't have one."""
        for symbol, pos in list(self.session.positions.items()):
            if pos.stop_loss_order_id:
                continue  # Already has SL order

            # Try to place SL order now that entry should be settled
            sl_price = pos.stop_loss_price
            sl_limit_price = sl_price * 0.985  # 1.5% below stop for volatile crypto

            # Get actual available quantity from Alpaca (may be less due to fees)
            actual_qty = self._get_actual_position_qty(symbol)
            if actual_qty is None or actual_qty <= 0:
                continue  # Position doesn't exist on Alpaca yet

            # Use actual available qty instead of order qty
            sl_qty = actual_qty

            try:
                self._log(f"    Retrying SL order for {symbol} (qty: {sl_qty:.6f})...", "INFO")
                sl_order = self.client.submit_crypto_stop_limit_order(
                    symbol=symbol,
                    qty=sl_qty,
                    side='sell',
                    stop_price=sl_price,
                    limit_price=sl_limit_price
                )
                sl_order_id = sl_order.get('id', 'N/A')
                sl_order_status = sl_order.get('status', 'unknown')
                pos.stop_loss_order_id = sl_order_id
                pos.stop_loss_order_status = sl_order_status
                # Update position qty to actual
                pos.qty = sl_qty
                self._log(f"    SL Order placed: {sl_order_id} ({sl_order_status})", "SUCCESS")
            except Exception as sl_e:
                # Don't spam - only log once per minute
                hold_time = (datetime.now(UTC) - pos.entry_time).total_seconds() / 60
                if int(hold_time) % 5 == 0:  # Log every 5 minutes
                    self._log(f"    SL retry failed for {symbol}: {sl_e}", "WARN")

    def _check_sl_orders_filled(self):
        """Check if any SL orders have been filled on Alpaca."""
        positions_closed_by_sl = []

        for symbol, pos in list(self.session.positions.items()):
            if not pos.stop_loss_order_id:
                continue

            try:
                order = self.client.get_order(pos.stop_loss_order_id)
                if order and order.get('status') == 'filled':
                    # SL order filled on Alpaca
                    fill_price = order.get('filled_avg_price', pos.stop_loss_price)
                    positions_closed_by_sl.append((symbol, fill_price))
            except Exception as e:
                # Order might not exist anymore, check position
                pass

        # Handle positions closed by SL
        for symbol, fill_price in positions_closed_by_sl:
            self._handle_sl_filled(symbol, fill_price)

    def _handle_sl_filled(self, symbol: str, fill_price: float):
        """Handle a position that was closed by the SL order on Alpaca."""
        pos = self.session.positions.get(symbol)
        if not pos:
            return

        hold_time = (datetime.now(UTC) - pos.entry_time).total_seconds() / 60

        # Calculate P&L
        proceeds = pos.qty * fill_price
        pnl = proceeds - pos.entry_cost
        pnl_pct = pnl / pos.entry_cost * 100

        print()
        self._log("=" * 60)
        self._log(f"<<< STOP LOSS FILLED ON ALPACA: {symbol}", "TRADE")
        self._log(f"    Entry Price: ${pos.entry_price:.4f}", "TRADE")
        self._log(f"    SL Fill Price: ${fill_price:.4f}", "SUCCESS")
        self._log(f"    Hold Time: {hold_time:.1f} minutes", "TRADE")

        log_level = "SUCCESS" if pnl >= 0 else "WARN"
        self._log(f"    RESULT:", log_level)
        self._log(f"      Entry: ${pos.entry_price:.4f} -> Exit: ${fill_price:.4f}", log_level)
        self._log(f"      P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)", log_level)
        self._log(f"      Reason: STOP_LOSS (Alpaca Order)", log_level)

        # Update session stats
        self.session.pnl_total += pnl
        if pnl > 0:
            self.session.wins += 1
        else:
            self.session.losses += 1

        win_rate = (self.session.wins / (self.session.wins + self.session.losses) * 100) if (self.session.wins + self.session.losses) > 0 else 0
        self._log(f"    SESSION: W/L {self.session.wins}/{self.session.losses} ({win_rate:.0f}%) | Total P&L: ${self.session.pnl_total:+.2f}", "INFO")
        self._log("=" * 60)
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
                notes="Alpaca SL order filled"
            )
        except Exception as log_e:
            self._log(f"    Trade logging error: {log_e}", "WARN")

        # Remove position
        del self.session.positions[symbol]

        # Reset strategy state for this symbol
        if symbol in self.strategy.symbol_states:
            self.strategy.symbol_states[symbol].reset_position_state()

    def _check_exit_conditions(self):
        """Check exit conditions for all open positions."""
        # First retry placing SL orders for positions that don't have one
        self._retry_missing_sl_orders()

        # Then check if any SL orders were filled on Alpaca
        self._check_sl_orders_filled()

        positions_to_close = []

        for symbol, pos in self.session.positions.items():
            quote = self.latest_quotes.get(symbol)
            if not quote:
                continue

            current_price = quote.mid
            exit_reason = None

            # Update highest price for trailing stop
            if current_price > pos.highest_price_since_entry:
                pos.highest_price_since_entry = current_price

                # Update trailing stop if in profit
                if self.config.use_trailing_stop:
                    profit_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                    if profit_pct >= self.config.trailing_stop_pct:
                        new_trailing = current_price * (1 - self.config.trailing_stop_pct / 100)
                        if new_trailing > pos.trailing_stop_price:
                            pos.trailing_stop_price = new_trailing

            # Calculate current P&L
            exit_price = current_price * (1 - self.half_spread)  # Sell at bid - half spread
            proceeds = pos.qty * exit_price
            pnl = proceeds - pos.entry_cost
            pnl_pct = pnl / pos.entry_cost * 100

            # ===================================================================
            # V7: ONLY TP AND SL EXITS - No minimum hold time, no technical exits
            # ===================================================================

            # Check take profit (monitored manually - TP order not on Alpaca)
            if current_price >= pos.take_profit_price:
                exit_reason = "TAKE_PROFIT"

            # Note: Stop loss is handled by Alpaca's stop order
            # Only check manually if SL order wasn't successfully placed
            elif not pos.stop_loss_order_id and current_price <= pos.stop_loss_price:
                exit_reason = "STOP_LOSS_MANUAL"

            # Check trailing stop (monitored manually - this is a dynamic SL)
            elif pos.trailing_stop_price > 0 and current_price <= pos.trailing_stop_price:
                exit_reason = "TRAILING_STOP"

            if exit_reason:
                positions_to_close.append((symbol, exit_reason, exit_price, pnl, pnl_pct))

        # Close positions
        for symbol, reason, exit_price, pnl, pnl_pct in positions_to_close:
            self._exit_position(symbol, reason, exit_price, pnl, pnl_pct)

    def _exit_position(self, symbol: str, reason: str, exit_price: float, pnl: float, pnl_pct: float):
        """Exit a position."""
        pos = self.session.positions.get(symbol)
        if not pos:
            return

        hold_time = (datetime.now(UTC) - pos.entry_time).total_seconds() / 60
        quote = self.latest_quotes.get(symbol)

        print()  # New line for better visibility
        self._log("=" * 60)
        self._log(f"<<< EXIT SIGNAL: {symbol} - {reason}", "TRADE")
        self._log(f"    Entry Price: ${pos.entry_price:.4f}", "TRADE")
        if quote:
            self._log(f"    Current: Bid=${quote.bid:.4f} | Ask=${quote.ask:.4f} | Mid=${quote.mid:.4f}", "TRADE")
        self._log(f"    Hold Time: {hold_time:.1f} minutes", "TRADE")

        # Cancel the SL order on Alpaca before submitting sell (to avoid double sell)
        if pos.stop_loss_order_id:
            self._log(f"    Cancelling SL order {pos.stop_loss_order_id[:12]}...", "TRADE")
            try:
                cancelled = self.client.cancel_order(pos.stop_loss_order_id)
                if cancelled:
                    self._log(f"      SL order cancelled successfully", "SUCCESS")
                else:
                    self._log(f"      SL order already filled or cancelled", "WARN")
            except Exception as cancel_e:
                self._log(f"      Failed to cancel SL order: {cancel_e}", "WARN")

        self._log(f"    Submitting SELL order to Alpaca...", "TRADE")

        try:
            # Get actual position from Alpaca to ensure we sell correct qty
            actual_qty = pos.qty
            try:
                alpaca_positions = self.client.get_positions()
                for ap in alpaca_positions:
                    alpaca_symbol = ap.get('symbol', '')
                    our_symbol = symbol.replace('/', '')
                    if alpaca_symbol == our_symbol:
                        actual_qty = float(ap.get('qty', pos.qty))
                        if abs(actual_qty - pos.qty) > 0.000001:
                            self._log(f"    Using Alpaca balance: {actual_qty} (vs tracked: {pos.qty})", "WARN")
                        break
            except Exception as pos_e:
                self._log(f"    Could not verify position: {pos_e}", "WARN")

            # Submit sell order with actual qty from Alpaca
            order = self.client.submit_crypto_market_order(
                symbol=symbol,
                qty=actual_qty,
                side='sell'
            )

            order_id = order.get('id', 'N/A')
            order_status = order.get('status', 'unknown')
            filled_price = order.get('filled_avg_price')
            filled_qty = order.get('filled_qty', 0)

            self._log(f"    ORDER SUBMITTED:", "SUCCESS")
            self._log(f"      Order ID: {order_id}", "SUCCESS")
            self._log(f"      Status: {order_status}", "SUCCESS")

            if filled_price:
                self._log(f"      Fill Price: ${filled_price:.4f}", "SUCCESS")
                exit_price = filled_price  # Use actual fill price
                # Recalculate P&L with actual fill
                proceeds = pos.qty * filled_price
                pnl = proceeds - pos.entry_cost
                pnl_pct = pnl / pos.entry_cost * 100

            if filled_qty:
                self._log(f"      Filled Qty: {filled_qty}", "SUCCESS")

            # Update position's exit order ID
            pos.exit_order_id = order_id

            # Display P&L
            log_level = "SUCCESS" if pnl >= 0 else "WARN"
            self._log(f"    RESULT:", log_level)
            self._log(f"      Entry: ${pos.entry_price:.4f} -> Exit: ${exit_price:.4f}", log_level)
            self._log(f"      P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)", log_level)
            self._log(f"      Reason: {reason}", log_level)

            # Update session stats
            self.session.pnl_total += pnl
            if pnl > 0:
                self.session.wins += 1
            else:
                self.session.losses += 1

            win_rate = (self.session.wins / (self.session.wins + self.session.losses) * 100) if (self.session.wins + self.session.losses) > 0 else 0
            self._log(f"    SESSION: W/L {self.session.wins}/{self.session.losses} ({win_rate:.0f}%) | Total P&L: ${self.session.pnl_total:+.2f}", "INFO")
            self._log("=" * 60)
            print()  # Extra line after exit

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
                    notes=f"Hold: {hold_time:.1f}m"
                )
            except Exception as log_e:
                self._log(f"    Trade logging error: {log_e}", "WARN")

            # Remove position
            del self.session.positions[symbol]

            # Reset strategy state for this symbol
            if symbol in self.strategy.symbol_states:
                self.strategy.symbol_states[symbol].reset_position_state()

        except Exception as e:
            self._log(f"    ORDER FAILED: {e}", "ERROR")
            self._log("=" * 60)
            print()

    def _get_symbol_scores(self) -> Dict[str, int]:
        """Get current entry score for each symbol from strategy."""
        scores = {}
        for symbol in self.config.symbols:
            if symbol in self.strategy.symbol_states:
                state = self.strategy.symbol_states[symbol]
                # Calculate current score based on indicators
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
                # Add volume confirmation if available
                if hasattr(state, 'volume_ma') and state.volume_ma.initialized:
                    score += 1
                scores[symbol] = score
            else:
                scores[symbol] = 0
        return scores

    def _print_status(self):
        """Print current status with trading hours and confirmation scores."""
        now = datetime.now(UTC)
        is_trading_hour = self._is_trading_hour()

        # Trading hours status
        if is_trading_hour:
            hours_status = "TRADING"
        else:
            hours_status = "OUTSIDE HOURS"
            # Calculate next trading window
            current_hour = now.hour
            next_hours = [h for h in self.config.allowed_trading_hours if h > current_hour]
            if next_hours:
                next_start = next_hours[0]
            else:
                next_start = self.config.allowed_trading_hours[0]  # Tomorrow
            hours_until = (next_start - current_hour) % 24
            hours_status += f" (next in {hours_until}h)"

        # Build main status line
        parts = [
            f"{now.strftime('%H:%M:%S')} UTC",
            hours_status,
            f"Pos: {len(self.session.positions)}/{self.config.max_concurrent_positions}",
            f"Trades: {self.session.trades_total}",
            f"W/L: {self.session.wins}/{self.session.losses}",
            f"P&L: ${self.session.pnl_total:+.2f}",
        ]

        # Add position details if any
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

        # Every 6 iterations (1 minute), print detailed scores
        if hasattr(self, '_status_counter'):
            self._status_counter += 1
        else:
            self._status_counter = 0

        if self._status_counter % 6 == 0:  # Every ~60 seconds
            self._print_scores_table()
            self._print_positions_table()  # Show SL/TP monitoring for open positions

    def _print_scores_table(self):
        """Print a table of current confirmation scores for all symbols (V10 Signal Hierarchy)."""
        from trading_system.strategies.crypto_scalping import SYMBOL_RISK_PARAMS, CryptoScalping

        scores = self._get_symbol_scores()
        default_min_score = self.config.min_entry_score

        print()  # New line after status
        print(f"\n{'='*100}")
        print(f"  V10 SIGNAL HIERARCHY - Macro(1H) + Trend(15m) + Technical(1m) + Price Action(5m)")
        print(f"{'='*100}")

        # Print header
        print(f"\n  {'SYM':<5} {'MACRO(1H)':<12} {'M0 (15m)':<10} {'M1 (1m)':<10} {'M2 (5m)':<14} {'PRICE':<12} {'STATUS'}")
        print(f"  {'-'*5} {'-'*12} {'-'*10} {'-'*10} {'-'*14} {'-'*12} {'-'*12}")

        for symbol in self.config.symbols:
            short_sym = symbol.split('/')[0]
            quote = self.latest_quotes.get(symbol)
            price_str = f"${quote.mid:,.2f}" if quote else "N/A"

            # MACRO: 1H Context (bias layer) - V10 NEW
            try:
                macro_result = self._check_macro_context(symbol)
                macro_bias = macro_result.get('bias', 'N/A')[:4]
                macro_regime = macro_result.get('regime', 'N/A')[:4]
                macro_adj = macro_result.get('score_adjustment', 0)
                adj_str = f"+{macro_adj}" if macro_adj > 0 else str(macro_adj)
                macro_str = f"{macro_bias} {adj_str}"
            except Exception:
                macro_str = "ERROR"
                macro_adj = 0

            # M0: Master Trend (15-min) - THE MOST IMPORTANT
            # V9.1: Use cached method to prevent flip-flopping
            try:
                trend_result = self._check_master_trend_signal(symbol)
                m0_trend = trend_result['trend']
                m0_strength = trend_result['strength']
                m0_slope = trend_result['ema20_slope']
                # Show trend with slope indicator (ASCII for Windows compatibility)
                slope_ind = "^" if m0_slope > 0.05 else ("v" if m0_slope < -0.05 else "-")
                m0_str = f"{m0_trend[:4]} {slope_ind}"
                m0_ready = m0_trend == 'UP'
                if 'Insufficient' in str(trend_result.get('reasons', [])):
                    m0_str = "NO DATA"
                    m0_ready = False
            except Exception:
                m0_str = "ERROR"
                m0_ready = False

            # M1: Technical score (1-min) - V10: show with macro adjustment
            m1_score = scores.get(symbol, 0)
            symbol_params = SYMBOL_RISK_PARAMS.get(symbol, {})
            min_score = symbol_params.get('min_entry_score', default_min_score)
            adjusted_score = m1_score + macro_adj
            m1_str = f"{adjusted_score}/{min_score}"
            m1_ready = adjusted_score >= min_score

            # M2: Price Action (5-min)
            try:
                bars_5min = self.client.get_crypto_bars(symbol, '5Min', limit=15)
                if bars_5min and len(bars_5min) >= 10:
                    pa_result = CryptoScalping.calculate_price_action_signal(bars_5min)
                    m2_signal = pa_result['signal']
                    m2_bull = pa_result['bullish_points']
                    m2_bear = pa_result['bearish_points']
                    m2_str = f"{m2_signal[:4]} {m2_bull}B/{m2_bear}R"
                    m2_ready = m2_signal == 'BULLISH'
                else:
                    m2_str = "NO DATA"
                    m2_ready = False
            except Exception:
                m2_str = "ERROR"
                m2_ready = False

            # Determine overall status - V10 still requires M0+M1+M2
            if m0_ready and m1_ready and m2_ready:
                status = ">>> ENTRY!"
            elif not m0_ready:
                status = "NO TREND"  # Master trend blocks everything
            elif m0_ready and m1_ready and not m2_ready:
                status = "M0+M1 ok"
            elif m0_ready and not m1_ready and m2_ready:
                status = "M0+M2 ok"
            elif m0_ready and adjusted_score >= min_score - 2:
                status = "M1 close"
            else:
                status = "Waiting"

            print(f"  {short_sym:<5} {macro_str:<12} {m0_str:<10} {m1_str:<10} {m2_str:<14} {price_str:<12} {status}")

        print(f"\n  MACRO: 1H bias (BULL/BEAR/NEUT) + score adj (-2 to +2)")
        print(f"  M0=MasterTrend(15m), M1=Technical+MacroAdj(1m), M2=PriceAction(5m)")
        print(f"  Entry: M0=UP AND M1(adjusted)>=threshold AND M2=BULLISH")
        print(f"{'='*100}")
        print()  # Extra line before resuming status updates

    def _print_positions_table(self):
        """Print detailed SL/TP monitoring table for all open positions."""
        if not self.session.positions:
            return

        print()
        print("=" * 70)
        print("OPEN POSITIONS - SL/TP MONITORING (Manual - Alpaca no bracket orders)")
        print("=" * 70)

        for symbol, pos in self.session.positions.items():
            quote = self.latest_quotes.get(symbol)
            if not quote:
                continue

            current_price = quote.mid
            hold_minutes = (datetime.now(UTC) - pos.entry_time).total_seconds() / 60

            # Calculate distances to SL/TP
            dist_to_tp = (pos.take_profit_price - current_price) / current_price * 100
            dist_to_sl = (current_price - pos.stop_loss_price) / current_price * 100

            # Calculate current P&L
            exit_price = current_price * (1 - self.half_spread)
            proceeds = pos.qty * exit_price
            pnl = proceeds - pos.entry_cost
            pnl_pct = pnl / pos.entry_cost * 100

            # Progress bar for SL/TP
            # Full range from SL to TP
            price_range = pos.take_profit_price - pos.stop_loss_price
            if price_range > 0:
                progress = (current_price - pos.stop_loss_price) / price_range
                progress = max(0, min(1, progress))  # Clamp to 0-1
                bar_width = 20
                filled = int(progress * bar_width)
                # Use ASCII characters for Windows compatibility
                bar = "#" * filled + "." * (bar_width - filled)
            else:
                bar = "." * 20

            short_sym = symbol.split('/')[0]
            print(f"\n  {short_sym} | Entry: ${pos.entry_price:.2f} | Now: ${current_price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
            print(f"  Entry Order: {pos.entry_order_id[:12]}...")
            print(f"  SL ${pos.stop_loss_price:.2f} [{bar}] TP ${pos.take_profit_price:.2f}")
            print(f"  Distance: SL {dist_to_sl:.2f}% away | TP {dist_to_tp:.2f}% away")

            # Show SL order status
            if pos.stop_loss_order_id:
                print(f"  SL Order: {pos.stop_loss_order_id[:12]}... (ON ALPACA)")
            else:
                print(f"  SL Order: MANUAL MONITORING (order not placed)")

            # Trailing stop status
            if self.config.use_trailing_stop:
                if pos.trailing_stop_price > 0:
                    trail_dist = (current_price - pos.trailing_stop_price) / current_price * 100
                    print(f"  Trailing Stop: ACTIVE at ${pos.trailing_stop_price:.2f} ({trail_dist:.2f}% away)")
                else:
                    print(f"  Trailing Stop: PENDING (activates at +{self.config.trailing_stop_pct}%)")

            # Hold time warning
            remaining = self.config.max_hold_minutes - hold_minutes
            if remaining < 10:
                print(f"  Hold Time: {hold_minutes:.0f}m / {self.config.max_hold_minutes}m - TIME EXIT in {remaining:.0f}m!")
            else:
                print(f"  Hold Time: {hold_minutes:.0f}m / {self.config.max_hold_minutes}m")

            # High water mark for trailing stop reference
            if pos.highest_price_since_entry > pos.entry_price:
                hwm_gain = (pos.highest_price_since_entry - pos.entry_price) / pos.entry_price * 100
                print(f"  High Water Mark: ${pos.highest_price_since_entry:.2f} (+{hwm_gain:.2f}%)")

        print()
        print("=" * 70)
        print()

    def run(self):
        """Main trading loop."""
        self._log("=" * 70)
        self._log("THE VOLUME AI - Crypto Paper Trading Engine V10.5")
        self._log("=" * 70)
        self._log(f"Symbols: {', '.join(self.config.symbols)}")
        self._log(f"Position Size: ${self.config.fixed_position_value:,.2f}")
        self._log(f"Max Positions: {self.config.max_concurrent_positions}")
        self._log(f"TP/SL: Per-symbol (V6.1)")
        self._log(f"Exit Modes: TP, SL, Trailing Stop only")
        self._log(f"Time Filter: {'ON' if self.config.use_time_filter else 'OFF'}")
        self._log("--- V10 SIGNAL HIERARCHY ---")
        self._log("  MACRO: Market Bias (1-hour bars) - Context/score adjustment")
        self._log("  M0: Master Trend (15-min bars) - REAL market direction")
        self._log("  M1: Technical Indicators (1-min bars) - Entry timing (adjusted by MACRO)")
        self._log("  M2: Price Action Analysis (5-min bars) - Confirmation")
        self._log("  Entry: M0=UP AND M1(+MACRO adj)>=threshold AND M2=BULLISH")
        self._log("=" * 70)

        # Test connection
        try:
            account = self.client.get_account()
            self._log("Connected to Alpaca Paper Trading", "SUCCESS")
            self._log(f"Account ID: {account['id']}")
            self._log(f"Cash: ${account['cash']:,.2f}")
            self._log(f"Buying Power: ${account['buying_power']:,.2f}")
        except Exception as e:
            self._log(f"Failed to connect to Alpaca: {e}", "ERROR")
            return

        # Initialize strategy
        self.strategy.on_start()

        # Warm up indicators with historical data
        self._warm_up_indicators()

        # Resume any existing positions from Alpaca and ensure SL orders exist
        self._resume_existing_positions()

        self.running = True
        self._log("Starting trading loop... (Ctrl+C to stop)")
        print()  # New line for status updates

        try:
            iteration = 0
            while self.running and not self._stop_event.is_set():
                iteration += 1

                # Update market data
                self._update_market_data()

                # Feed strategy with latest bars
                self._feed_strategy()

                # Check exit conditions for open positions
                self._check_exit_conditions()

                # Check for new entry signals
                self._check_entry_signals()

                # Print status
                self._print_status()

                # Sleep between iterations (10 seconds for crypto)
                self._stop_event.wait(timeout=10)

        except KeyboardInterrupt:
            print()  # New line after status
            self._log("Shutdown requested...")
        finally:
            self.running = False

            # Ask user about open positions
            if self.session.positions:
                print()
                print("=" * 60)
                print("  SHUTDOWN - OPEN POSITIONS DETECTED")
                print("=" * 60)

                # Show current positions with P&L
                total_pnl = 0.0
                for symbol, pos in self.session.positions.items():
                    quote = self.latest_quotes.get(symbol)
                    if quote:
                        exit_price = quote.mid * (1 - self.half_spread)
                        proceeds = pos.qty * exit_price
                        pnl = proceeds - pos.entry_cost
                        pnl_pct = pnl / pos.entry_cost * 100
                        total_pnl += pnl
                        short_sym = symbol.split('/')[0]
                        print(f"  {short_sym}: ${pnl:+.2f} ({pnl_pct:+.2f}%)")

                print(f"\n  Total unrealized P&L: ${total_pnl:+.2f}")
                print("=" * 60)

                # Ask user what to do
                print("\n  What would you like to do with open positions?")
                print("  [K] KEEP positions open on Alpaca (SL orders remain active)")
                print("  [C] CLOSE all positions now")
                print()

                try:
                    response = input("  Enter choice (K/C): ").strip().upper()
                except (EOFError, KeyboardInterrupt):
                    response = 'K'  # Default to keep on second Ctrl+C

                if response == 'C':
                    self._log(f"Closing {len(self.session.positions)} open positions...")
                    for symbol in list(self.session.positions.keys()):
                        quote = self.latest_quotes.get(symbol)
                        if quote:
                            pos = self.session.positions[symbol]
                            exit_price = quote.mid * (1 - self.half_spread)
                            proceeds = pos.qty * exit_price
                            pnl = proceeds - pos.entry_cost
                            pnl_pct = pnl / pos.entry_cost * 100
                            self._exit_position(symbol, "SHUTDOWN", exit_price, pnl, pnl_pct)
                else:
                    self._log("Keeping positions open on Alpaca.")
                    self._log("SL orders (if placed) will remain active on Alpaca.")
                    self._log("Use Alpaca dashboard to manage positions manually.")
                    # Cancel SL orders if keeping positions? No, keep them for protection
                    print()
                    print("  Open positions kept on Alpaca:")
                    for symbol, pos in self.session.positions.items():
                        short_sym = symbol.split('/')[0]
                        if pos.stop_loss_order_id:
                            print(f"    {short_sym}: SL order active ({pos.stop_loss_order_id[:12]}...)")
                        else:
                            print(f"    {short_sym}: NO SL order - manage manually!")
                    print()

            self.strategy.on_stop()
            self._log("Paper trading engine stopped.")
            self._print_session_summary()

    def stop(self):
        """Signal the engine to stop."""
        self.running = False
        self._stop_event.set()

    def _print_session_summary(self):
        """Print session summary."""
        print()
        self._log("=" * 70)
        self._log("SESSION SUMMARY")
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
        self._log("=" * 70)


# Import os for color detection
import os


def run_crypto_paper_trading(config: CryptoPaperTradingConfig):
    """Run the crypto paper trading engine with given config."""
    engine = CryptoPaperTradingEngine(config)

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        engine.stop()

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    engine.run()
