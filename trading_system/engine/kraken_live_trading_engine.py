"""
Kraken Futures LIVE Trading Engine

ENGINE FOR REAL MONEY MARGIN TRADING!

Engine for ETH margin trading on Kraken Futures PRODUCTION environment.
Uses futures.kraken.com for live trading with real funds.

WARNING: This engine trades with REAL MONEY.
Leverage amplifies both gains AND losses.
Risk of LIQUIDATION exists.

Features:
- Real-time price polling from Kraken Futures API
- Multi-timeframe bar storage (1m, 5m, 15m, 1h)
- V10 signal hierarchy validation
- LIVE order execution
- Position management with leverage
- Trade logging
- Risk controls
"""

import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from threading import Event
import pytz

from ..clients.kraken_futures_client import KrakenFuturesClient, Bar, Quote
from ..strategies.eth_margin_scalping import (
    ETHMarginStrategy,
    ETHMarginConfig,
    ETHMarginState,
    BarData,
)
from ..config.kraken_margin_config import KrakenLiveConfig


UTC = pytz.UTC


class KrakenLiveTradingEngine:
    """
    LIVE trading engine for ETH margin on Kraken Futures.

    WARNING: This trades with REAL MONEY!
    """

    def __init__(
        self,
        config: KrakenLiveConfig,
        strategy_config: Optional[ETHMarginConfig] = None
    ):
        """
        Initialize the LIVE trading engine.

        Args:
            config: Kraken live configuration
            strategy_config: Optional strategy configuration
        """
        self.config = config
        self.strategy_config = strategy_config or ETHMarginConfig(
            position_value_usd=config.position_value_usd,
            leverage=config.leverage,
            target_profit_pct=config.target_profit_pct,
            stop_loss_pct=config.stop_loss_pct,
            trailing_stop_pct=config.trailing_stop_pct,
            use_trailing_stop=config.use_trailing_stop,
            min_entry_score=config.min_entry_score,
            use_time_filter=config.use_time_filter,
            allowed_trading_hours=config.allowed_trading_hours,
        )

        # Initialize client (LIVE mode)
        self.client = KrakenFuturesClient(
            api_key=config.api_key,
            api_secret=config.api_secret,
            demo=False  # LIVE MODE!
        )

        # Initialize strategy
        self.strategy = ETHMarginStrategy(self.strategy_config)
        self.symbol = config.symbol  # PI_ETHUSD

        # Multi-timeframe bar storage
        self.bars_1min: List[BarData] = []
        self.bars_5min: List[BarData] = []
        self.bars_15min: List[BarData] = []
        self.bars_1h: List[BarData] = []

        # Bar aggregation state
        self.last_5min_bar_time: Optional[datetime] = None
        self.last_15min_bar_time: Optional[datetime] = None
        self.last_1h_bar_time: Optional[datetime] = None
        self.current_5min_bars: List[BarData] = []
        self.current_15min_bars: List[BarData] = []
        self.current_1h_bars: List[BarData] = []

        # Trading state
        self.running = False
        self.stop_event = Event()
        self.last_bar_time: Optional[datetime] = None

        # Position tracking (from Kraken)
        self.current_position: Optional[Dict] = None

        # Risk controls
        self.daily_loss = 0.0
        self.daily_loss_limit = config.position_value_usd * config.max_daily_loss_pct / 100
        self.trading_halted = False

        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.start_time: Optional[datetime] = None

        # Trade log
        self.trade_log: List[Dict] = []

        print("\n" + "!" * 60)
        print("!" * 60)
        print("         KRAKEN FUTURES LIVE TRADING ENGINE")
        print("           *** REAL MONEY - MARGIN TRADING ***")
        print("!" * 60)
        print("!" * 60)
        print(f"\nEnvironment: LIVE (futures.kraken.com)")
        print(f"Symbol: {self.symbol}")
        print(f"Position Size: ${config.position_value_usd}")
        print(f"Leverage: {config.leverage}x")
        print(f"Effective Exposure: ${config.position_value_usd * config.leverage}")
        print(f"Daily Loss Limit: ${self.daily_loss_limit:.2f}")
        print(f"Take Profit: {config.target_profit_pct}%")
        print(f"Stop Loss: {config.stop_loss_pct}%")
        print("\n" + "!" * 60)

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            print("\n\n!!! SHUTDOWN SIGNAL RECEIVED !!!")
            print("Closing positions and shutting down...")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def start(self):
        """Start the LIVE trading engine."""
        # Final confirmation
        print("\n" + "!" * 60)
        print("WARNING: You are about to start LIVE trading with REAL MONEY!")
        print("This uses LEVERAGE which can result in losses exceeding your position size.")
        print("!" * 60)

        confirm = input("\nType 'START LIVE TRADING' to confirm: ").strip()
        if confirm != 'START LIVE TRADING':
            print("\nLive trading cancelled.")
            return

        self._setup_signal_handlers()
        self.running = True
        self.start_time = datetime.now(UTC)

        print("\n" + "=" * 60)
        print("STARTING LIVE TRADING ENGINE")
        print("=" * 60)
        print("Press Ctrl+C to stop.\n")

        # Test connection
        print("Testing connection to Kraken Futures LIVE...")
        if not self.client.test_connection():
            print("ERROR: Failed to connect to Kraken Futures")
            return

        # Test authentication
        print("Testing authentication...")
        if not self.client.test_auth():
            print("ERROR: Authentication failed. Check API credentials.")
            return

        # Check account balance
        summary = self.client.get_account_summary()
        if summary:
            print(f"\nAccount Summary:")
            print(f"  Portfolio Value: ${summary.get('portfolio_value', 0):,.2f}")
            print(f"  Available Margin: ${summary.get('available_margin', 0):,.2f}")

            # Check if we have enough margin
            required_margin = self.config.position_value_usd
            available = summary.get('available_margin', 0)
            if available < required_margin:
                print(f"\nWARNING: Available margin (${available:.2f}) < position size (${required_margin:.2f})")
                print("Consider reducing position size or adding funds.")
                confirm2 = input("Continue anyway? [y/N]: ").strip().lower()
                if confirm2 != 'y':
                    print("Live trading cancelled.")
                    return

        # Check for existing positions
        positions = self.client.get_positions()
        if positions:
            print(f"\nWARNING: You have {len(positions)} existing positions!")
            for pos in positions:
                print(f"  {pos.symbol}: {pos.side} {pos.size} @ ${pos.entry_price:.2f}")
            print("The engine will manage these positions.")

        # Load historical bars for indicators
        self._load_historical_bars()

        print("\n" + "=" * 60)
        print("LIVE TRADING STARTED")
        print("=" * 60)

        # Start main loop
        self._run_main_loop()

    def stop(self):
        """Stop the LIVE trading engine."""
        print("\n" + "!" * 60)
        print("STOPPING LIVE TRADING ENGINE")
        print("!" * 60)

        self.running = False
        self.stop_event.set()

        # Close any open position
        if self.strategy.state.has_position:
            print("\nClosing open position...")
            self._close_position("USER_SHUTDOWN")

        # Print final stats
        self._print_session_stats()

        print("\nLive trading engine stopped.")

    def _check_risk_limits(self) -> bool:
        """Check if trading should continue based on risk limits."""
        # Daily loss limit
        if self.daily_loss >= self.daily_loss_limit:
            if not self.trading_halted:
                print("\n" + "!" * 60)
                print(f"DAILY LOSS LIMIT REACHED: ${self.daily_loss:.2f}")
                print("Trading halted for the day.")
                print("!" * 60)
                self.trading_halted = True
            return False

        return True

    def _load_historical_bars(self):
        """Load historical bars for indicator initialization."""
        print("\nLoading historical bars...")

        try:
            # Load 1-minute bars (last 100)
            bars_1m = self.client.get_bars(self.symbol, '1Min', limit=100)
            for bar in bars_1m:
                bar_data = BarData(
                    symbol=self.symbol,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    timestamp=bar.timestamp
                )
                self.bars_1min.append(bar_data)
                self.strategy.state.update_indicators(bar_data)

            print(f"  Loaded {len(self.bars_1min)} 1-minute bars")

            # Load 5-minute bars
            bars_5m = self.client.get_bars(self.symbol, '5Min', limit=50)
            for bar in bars_5m:
                self.bars_5min.append(BarData(
                    symbol=self.symbol,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    timestamp=bar.timestamp
                ))
            print(f"  Loaded {len(self.bars_5min)} 5-minute bars")

            # Load 15-minute bars
            bars_15m = self.client.get_bars(self.symbol, '15Min', limit=30)
            for bar in bars_15m:
                self.bars_15min.append(BarData(
                    symbol=self.symbol,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    timestamp=bar.timestamp
                ))
            print(f"  Loaded {len(self.bars_15min)} 15-minute bars")

            # Load 1-hour bars
            bars_1h = self.client.get_bars(self.symbol, '1Hour', limit=25)
            for bar in bars_1h:
                self.bars_1h.append(BarData(
                    symbol=self.symbol,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    timestamp=bar.timestamp
                ))
            print(f"  Loaded {len(self.bars_1h)} 1-hour bars")

            # Update strategy state with multi-timeframe bars
            self._update_strategy_bars()

            print("Historical bars loaded successfully.")

        except Exception as e:
            print(f"WARNING: Error loading historical bars: {e}")
            print("Starting with empty history.")

    def _update_strategy_bars(self):
        """Update strategy state with multi-timeframe bars."""
        self.strategy.state.bars_5min = self.bars_5min[-20:] if self.bars_5min else None
        self.strategy.state.bars_15min = self.bars_15min[-30:] if self.bars_15min else None
        self.strategy.state.bars_1h = self.bars_1h[-25:] if self.bars_1h else None

    def _run_main_loop(self):
        """Main trading loop - poll for new bars."""
        poll_interval = 60  # Poll every 60 seconds for 1-minute bars

        while self.running and not self.stop_event.is_set():
            try:
                # Check risk limits
                if not self._check_risk_limits():
                    self.stop_event.wait(300)  # Wait 5 minutes if halted
                    continue

                # Get latest quote
                quote = self.client.get_quote(self.symbol)
                if not quote:
                    print("WARNING: Failed to get quote")
                    time.sleep(10)
                    continue

                # Create bar from quote (simulated 1-min bar)
                now = datetime.now(UTC)
                bar = BarData(
                    symbol=self.symbol,
                    open=quote.mid,
                    high=quote.ask,
                    low=quote.bid,
                    close=quote.mid,
                    volume=0,
                    timestamp=now
                )

                # Process bar
                self._process_bar(bar)

                # Status update
                self._print_status(quote)

                # Sync position from Kraken periodically
                if now.minute % 5 == 0:
                    self._sync_position()

                # Wait for next poll
                self.stop_event.wait(poll_interval)

            except Exception as e:
                print(f"ERROR in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(10)

        print("\nMain loop ended.")

    def _sync_position(self):
        """Sync position state from Kraken."""
        try:
            position = self.client.get_position(self.symbol)
            if position:
                self.strategy.state.has_position = True
                self.strategy.state.position_side = position.side
                self.strategy.state.position_size = position.size
                if self.strategy.state.entry_price is None:
                    self.strategy.state.entry_price = position.entry_price
            else:
                if self.strategy.state.has_position:
                    # Position was closed externally
                    print("\nWARNING: Position closed externally!")
                self.strategy.state.reset_position_state()
        except Exception as e:
            print(f"Error syncing position: {e}")

    def _process_bar(self, bar: BarData):
        """Process a new bar through the strategy."""
        # Add to 1-min bars
        self.bars_1min.append(bar)
        if len(self.bars_1min) > 200:
            self.bars_1min = self.bars_1min[-200:]

        # Aggregate to higher timeframes
        self._aggregate_bars(bar)

        # Update strategy with multi-timeframe bars
        self._update_strategy_bars()

        # Process through strategy
        result = self.strategy.on_bar(bar)

        if result:
            action = result.get('action')

            if action == 'enter_long':
                self._enter_position(result)
            elif action == 'exit':
                self._close_position(result.get('reason', 'STRATEGY_EXIT'))

    def _aggregate_bars(self, bar: BarData):
        """Aggregate 1-min bars to higher timeframes."""
        now = bar.timestamp

        # 5-minute aggregation
        current_5min = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
        if self.last_5min_bar_time != current_5min:
            if self.current_5min_bars:
                aggregated = self._aggregate_bar_list(self.current_5min_bars)
                self.bars_5min.append(aggregated)
                if len(self.bars_5min) > 100:
                    self.bars_5min = self.bars_5min[-100:]
            self.current_5min_bars = []
            self.last_5min_bar_time = current_5min
        self.current_5min_bars.append(bar)

        # 15-minute aggregation
        current_15min = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        if self.last_15min_bar_time != current_15min:
            if self.current_15min_bars:
                aggregated = self._aggregate_bar_list(self.current_15min_bars)
                self.bars_15min.append(aggregated)
                if len(self.bars_15min) > 100:
                    self.bars_15min = self.bars_15min[-100:]
            self.current_15min_bars = []
            self.last_15min_bar_time = current_15min
        self.current_15min_bars.append(bar)

        # 1-hour aggregation
        current_1h = now.replace(minute=0, second=0, microsecond=0)
        if self.last_1h_bar_time != current_1h:
            if self.current_1h_bars:
                aggregated = self._aggregate_bar_list(self.current_1h_bars)
                self.bars_1h.append(aggregated)
                if len(self.bars_1h) > 50:
                    self.bars_1h = self.bars_1h[-50:]
            self.current_1h_bars = []
            self.last_1h_bar_time = current_1h
        self.current_1h_bars.append(bar)

    def _aggregate_bar_list(self, bars: List[BarData]) -> BarData:
        """Aggregate a list of bars into a single bar."""
        if not bars:
            return bars[0] if bars else None

        return BarData(
            symbol=bars[0].symbol,
            open=bars[0].open,
            high=max(b.high for b in bars),
            low=min(b.low for b in bars),
            close=bars[-1].close,
            volume=sum(b.volume for b in bars),
            timestamp=bars[-1].timestamp
        )

    def _enter_position(self, result: Dict):
        """Enter a long position - LIVE ORDER."""
        if self.strategy.state.has_position:
            return

        if self.trading_halted:
            return

        price = result.get('price', 0)
        score = result.get('score', 0)

        # Calculate position size
        position_size = self.config.position_value_usd / price

        print("\n" + "!" * 50)
        print("LIVE ORDER - ENTERING LONG POSITION")
        print(f"  Price: ${price:,.2f}")
        print(f"  Size: {position_size:.6f} ETH")
        print(f"  Value: ${self.config.position_value_usd:,.2f}")
        print(f"  Leverage: {self.config.leverage}x")
        print(f"  Score: {score}")
        print("!" * 50)

        # Submit LIVE order to Kraken
        try:
            order_result = self.client.submit_market_order(
                symbol=self.symbol,
                side='buy',
                size=position_size
            )

            if order_result.get('result') == 'success':
                # Update strategy state
                self.strategy.state.has_position = True
                self.strategy.state.position_side = 'long'
                self.strategy.state.position_size = position_size
                self.strategy.state.entry_price = price
                self.strategy.state.entry_time = datetime.now(UTC)
                self.strategy.state.entry_score = score
                self.strategy.state.last_trade_time = datetime.now(UTC)

                # Log trade
                self.trade_log.append({
                    'action': 'entry',
                    'side': 'long',
                    'price': price,
                    'size': position_size,
                    'score': score,
                    'time': datetime.now(UTC).isoformat(),
                    'order_id': order_result.get('sendStatus', {}).get('order_id', 'N/A'),
                    'live': True
                })

                print("LIVE ORDER SUBMITTED SUCCESSFULLY!")
            else:
                print(f"LIVE ORDER FAILED: {order_result}")

        except Exception as e:
            print(f"ERROR submitting LIVE order: {e}")

    def _close_position(self, reason: str):
        """Close the current position - LIVE ORDER."""
        if not self.strategy.state.has_position:
            return

        state = self.strategy.state
        entry_price = state.entry_price or 0
        position_size = state.position_size

        # Get current price
        quote = self.client.get_quote(self.symbol)
        if not quote:
            print("ERROR: Cannot get quote for exit")
            return

        exit_price = quote.mid

        # Calculate P&L
        if state.position_side == 'long':
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            pnl_usd = (exit_price - entry_price) * position_size * self.config.leverage
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            pnl_usd = (entry_price - exit_price) * position_size * self.config.leverage

        # Account for fees
        fee_pct = self.config.taker_fee_pct * 2
        net_pnl_pct = pnl_pct - fee_pct
        net_pnl_usd = pnl_usd - (self.config.position_value_usd * fee_pct / 100)

        is_win = net_pnl_usd > 0

        print("\n" + "!" * 50)
        print(f"LIVE ORDER - CLOSING POSITION: {reason}")
        print(f"  Entry: ${entry_price:,.2f}")
        print(f"  Exit: ${exit_price:,.2f}")
        print(f"  P&L: {net_pnl_pct:+.2f}% (${net_pnl_usd:+.2f})")
        print(f"  Result: {'WIN' if is_win else 'LOSS'}")
        print("!" * 50)

        # Submit LIVE close order
        try:
            order_result = self.client.submit_market_order(
                symbol=self.symbol,
                side='sell',
                size=position_size,
                reduce_only=True
            )

            # Update statistics
            self.total_trades += 1
            self.total_pnl += net_pnl_usd
            if is_win:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
                self.daily_loss += abs(net_pnl_usd)

            # Record trade
            self.strategy.record_trade(net_pnl_usd, is_win)

            # Log trade
            self.trade_log.append({
                'action': 'exit',
                'reason': reason,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': position_size,
                'pnl_pct': net_pnl_pct,
                'pnl_usd': net_pnl_usd,
                'time': datetime.now(UTC).isoformat(),
                'live': True
            })

            # Reset state
            self.strategy.state.reset_position_state()

            print("LIVE CLOSE ORDER EXECUTED!")

        except Exception as e:
            print(f"ERROR closing LIVE position: {e}")

    def _print_status(self, quote: Quote):
        """Print current status."""
        now = datetime.now(UTC)
        price = quote.mid
        state = self.strategy.state

        # Build status line
        status_parts = [
            "[LIVE]",
            f"[{now.strftime('%H:%M:%S')}]",
            f"ETH=${price:,.2f}",
        ]

        # Position info
        if state.has_position:
            pnl_pct = ((price - state.entry_price) / state.entry_price) * 100
            status_parts.append(f"POS: LONG @${state.entry_price:,.2f} ({pnl_pct:+.2f}%)")
        else:
            m0_trend = "?"
            if state.last_m0_result:
                m0_trend = state.last_m0_result.get('trend', '?')
            status_parts.append(f"M0={m0_trend}")

        # Stats
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            status_parts.append(f"Trades={self.total_trades} WR={win_rate:.0f}% P&L=${self.total_pnl:+.2f}")

        # Daily loss warning
        if self.daily_loss > self.daily_loss_limit * 0.5:
            status_parts.append(f"!DLOSS=${self.daily_loss:.2f}!")

        print(" | ".join(status_parts))

    def _print_session_stats(self):
        """Print session statistics."""
        print("\n" + "=" * 60)
        print("LIVE TRADING SESSION STATISTICS")
        print("=" * 60)

        if self.start_time:
            duration = datetime.now(UTC) - self.start_time
            print(f"Duration: {duration}")

        print(f"\nTotal Trades: {self.total_trades}")
        print(f"  Winning: {self.winning_trades}")
        print(f"  Losing: {self.losing_trades}")

        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            print(f"  Win Rate: {win_rate:.1f}%")

        print(f"\nTotal P&L: ${self.total_pnl:+.2f}")
        print(f"Daily Loss: ${self.daily_loss:.2f} / ${self.daily_loss_limit:.2f}")
        print("=" * 60)


def run_kraken_live_trading():
    """Run the Kraken LIVE trading engine."""
    print("\n" + "!" * 60)
    print("!" * 60)
    print("     KRAKEN FUTURES ETH MARGIN - LIVE TRADING")
    print("           *** REAL MONEY ***")
    print("!" * 60)
    print("!" * 60)

    # Load config
    if not KrakenLiveConfig.exists():
        print("\nNo LIVE configuration found. Running setup wizard...")
        from ..config.kraken_margin_config import run_kraken_live_setup_wizard
        config = run_kraken_live_setup_wizard()
    else:
        config = KrakenLiveConfig.load()

    if not config.is_configured():
        print("\nLIVE configuration incomplete. Please run setup wizard.")
        return

    # Create and start engine
    engine = KrakenLiveTradingEngine(config)
    engine.start()


if __name__ == "__main__":
    run_kraken_live_trading()
