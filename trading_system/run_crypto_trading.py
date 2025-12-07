"""
Crypto Scalping Strategy - Live/Paper Trading Runner

This script runs the crypto scalping strategy on Alpaca (paper or live).
Crypto markets trade 24/7 so this runs continuously.

Usage:
    # First time setup
    python -m trading_system.run_crypto_trading --setup

    # Start trading (after setup)
    python -m trading_system.run_crypto_trading

    # Test connection
    python -m trading_system.run_crypto_trading --test

    # Reconfigure settings
    python -m trading_system.run_crypto_trading --reconfigure
"""

import argparse
import asyncio
import signal
import sys
import io
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from trading_system.config.crypto_trading_config import (
    CryptoTradingConfig,
    run_crypto_setup_wizard,
    run_crypto_reconfigure,
    mask_api_key,
)


def test_connection(config: CryptoTradingConfig) -> bool:
    """Test Alpaca API connection."""
    print("\n" + "=" * 60)
    print("     TESTING ALPACA CONNECTION")
    print("=" * 60)

    try:
        from trading_system.engine.alpaca_client import AlpacaClient

        client = AlpacaClient(
            api_key=config.api_key,
            api_secret=config.api_secret,
            paper=config.use_paper
        )

        # Test account access
        account = client.get_account()
        print(f"\nConnection successful!")
        print(f"  Account ID: {account['id']}")
        print(f"  Status: {account['status']}")
        print(f"  Cash: ${account['cash']:,.2f}")
        print(f"  Buying Power: ${account['buying_power']:,.2f}")
        print(f"  Mode: {'Paper Trading' if config.use_paper else 'LIVE TRADING'}")

        # Test crypto quote
        print(f"\nTesting crypto data access...")
        quote = client.get_latest_crypto_quote('BTC/USD')
        if quote:
            print(f"  BTC/USD: ${quote.mid:,.2f}")
        else:
            print(f"  Note: Could not fetch BTC quote (may be outside market hours)")

        print("\nAll tests passed!")
        return True

    except ImportError:
        print("\nERROR: alpaca-py package not installed.")
        print("Run: pip install alpaca-py")
        return False
    except Exception as e:
        print(f"\nConnection failed: {e}")
        return False


class CryptoTradingRunner:
    """
    Runs the crypto scalping strategy in real-time.

    Features:
    - 24/7 operation (crypto never sleeps)
    - Real-time market data via Alpaca
    - Automatic position management
    - Risk controls and daily limits
    """

    def __init__(self, config: CryptoTradingConfig):
        self.config = config
        self.running = False
        self.client = None
        self.strategy = None

        # Statistics
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None

    async def start(self):
        """Start the trading loop."""
        print("\n" + "=" * 80)
        print("     CRYPTO SCALPING STRATEGY - STARTING")
        print("=" * 80)

        # Initialize Alpaca client
        from trading_system.engine.alpaca_client import AlpacaClient
        from trading_system.strategies.crypto_scalping import CryptoScalping, CryptoScalpingConfig

        self.client = AlpacaClient(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            paper=self.config.use_paper
        )

        # Get account info
        account = self.client.get_account()
        print(f"\nAccount Status: {account['status']}")
        print(f"Available Cash: ${account['cash']:,.2f}")
        print(f"Mode: {'PAPER TRADING' if self.config.use_paper else '*** LIVE TRADING ***'}")

        # Create strategy config
        strategy_config = CryptoScalpingConfig(
            symbols=self.config.symbols,
            fixed_position_value=self.config.fixed_position_value,
            max_position_value=self.config.max_position_value,
            target_profit_pct=self.config.target_profit_pct,
            stop_loss_pct=self.config.stop_loss_pct,
            trailing_stop_pct=self.config.trailing_stop_pct,
            use_trailing_stop=self.config.use_trailing_stop,
            max_daily_loss=self.config.max_daily_loss,
            max_trades_per_day=self.config.max_trades_per_day,
            max_concurrent_positions=self.config.max_concurrent_positions,
            min_time_between_trades=self.config.min_time_between_trades,
            rsi_period=self.config.rsi_period,
            rsi_oversold=self.config.rsi_oversold,
            rsi_overbought=self.config.rsi_overbought,
            bb_period=self.config.bb_period,
            bb_std_dev=self.config.bb_std_dev,
            vwap_period=self.config.vwap_period,
            volume_ma_period=self.config.volume_ma_period,
            volume_spike_multiplier=self.config.volume_spike_multiplier,
            adx_period=self.config.adx_period,
            adx_trend_threshold=self.config.adx_trend_threshold,
        )

        self.strategy = CryptoScalping(strategy_config)

        print(f"\nTrading Symbols: {', '.join(self.config.symbols)}")
        print(f"Position Size: ${self.config.fixed_position_value}")
        print(f"TP/SL: +{self.config.target_profit_pct}% / -{self.config.stop_loss_pct}%")
        print(f"Max Daily Loss: ${self.config.max_daily_loss}")
        print(f"Max Trades/Day: {self.config.max_trades_per_day}")

        # Start strategy
        self.strategy.on_start()
        self.running = True

        print("\n" + "-" * 60)
        print("Strategy is now running. Press Ctrl+C to stop.")
        print("-" * 60 + "\n")

        # Main trading loop
        await self._trading_loop()

    async def _trading_loop(self):
        """Main trading loop - polls for data and runs strategy."""
        from trading_system.core.models import Bar
        import pytz

        poll_interval = 60  # Poll every 60 seconds (1 minute bars)

        while self.running:
            try:
                current_time = datetime.now(pytz.UTC)

                # Check for new day
                current_date = current_time.date()
                if self.last_trade_date != current_date:
                    if self.last_trade_date is not None:
                        print(f"\n{'='*60}")
                        print(f"NEW TRADING DAY: {current_date}")
                        print(f"Previous Day P&L: ${self.daily_pnl:.2f}")
                        print(f"Trades: {self.trades_today}")
                        print(f"{'='*60}\n")

                    self.last_trade_date = current_date
                    self.trades_today = 0
                    self.daily_pnl = 0.0

                # Fetch latest data for each symbol
                for symbol in self.config.symbols:
                    try:
                        # Get latest bars (last 100 for indicator warmup)
                        bars = self.client.get_crypto_bars(
                            symbol=symbol,
                            timeframe='1Min',
                            limit=100
                        )

                        if bars:
                            # Process most recent bar
                            latest = bars[-1]
                            bar = Bar(
                                symbol=symbol,
                                timestamp=latest.timestamp,
                                open=latest.open,
                                high=latest.high,
                                low=latest.low,
                                close=latest.close,
                                volume=latest.volume,
                            )

                            # First warm up indicators with historical bars
                            if not self._is_warmed_up(symbol):
                                for b in bars[:-1]:
                                    warm_bar = Bar(
                                        symbol=symbol,
                                        timestamp=b.timestamp,
                                        open=b.open,
                                        high=b.high,
                                        low=b.low,
                                        close=b.close,
                                        volume=b.volume,
                                    )
                                    self.strategy.on_bar(warm_bar)

                            # Process latest bar
                            self.strategy.on_bar(bar)

                            # Check for trade signals and execute
                            await self._check_and_execute_trades(symbol)

                    except Exception as e:
                        print(f"Error processing {symbol}: {e}")

                # Show status periodically
                if current_time.minute % 5 == 0 and current_time.second < 60:
                    self._print_status()

                # Wait for next poll
                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in trading loop: {e}")
                await asyncio.sleep(10)  # Wait before retry

        # Cleanup
        self.strategy.on_stop()
        print("\nTrading stopped.")

    def _is_warmed_up(self, symbol: str) -> bool:
        """Check if indicators are warmed up for a symbol."""
        if symbol in self.strategy.symbol_states:
            state = self.strategy.symbol_states[symbol]
            return state.indicators_ready()
        return False

    async def _check_and_execute_trades(self, symbol: str):
        """Check for pending orders and execute them."""
        if symbol not in self.strategy.symbol_states:
            return

        state = self.strategy.symbol_states[symbol]

        # Check for entry signal
        if state.pending_entry_order_id:
            try:
                # Get current price
                quote = self.client.get_latest_crypto_quote(symbol)
                if quote:
                    qty = self.config.fixed_position_value / quote.mid

                    print(f"\n{'='*60}")
                    print(f"EXECUTING ENTRY: {symbol}")
                    print(f"  Price: ${quote.mid:.4f}")
                    print(f"  Quantity: {qty:.6f}")
                    print(f"  Value: ${self.config.fixed_position_value:.2f}")

                    # Submit order
                    order = self.client.submit_crypto_market_order(
                        symbol=symbol,
                        qty=qty,
                        side='buy'
                    )

                    if order:
                        print(f"  Order ID: {order['id']}")
                        print(f"  Status: {order['status']}")

                        # Update state
                        state.entry_price = quote.mid
                        state.entry_time = datetime.now()
                        state.highest_price_since_entry = quote.mid
                        state.position = type('Position', (), {'quantity': qty, 'is_flat': False})()

                        self.trades_today += 1

                    state.pending_entry_order_id = None
                    print(f"{'='*60}\n")

            except Exception as e:
                print(f"Error executing entry for {symbol}: {e}")
                state.pending_entry_order_id = None

    def _print_status(self):
        """Print current status."""
        print(f"\n--- Status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"Trades Today: {self.trades_today}")
        print(f"Daily P&L: ${self.daily_pnl:.2f}")

        # Show positions
        positions = self.client.get_positions()
        crypto_positions = [p for p in positions if '/' in p['symbol'] or 'USD' in p['symbol']]

        if crypto_positions:
            print(f"Open Positions:")
            for pos in crypto_positions:
                print(f"  {pos['symbol']}: {pos['qty']} @ ${pos['avg_entry_price']:.4f} "
                      f"(P&L: ${pos['unrealized_pl']:.2f})")
        else:
            print("No open positions")

        print("-" * 40)

    def stop(self):
        """Stop the trading loop."""
        self.running = False


def main():
    parser = argparse.ArgumentParser(
        description='TheVolumeAI Crypto Scalping Strategy - Trading Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time setup
  python -m trading_system.run_crypto_trading --setup

  # Start trading
  python -m trading_system.run_crypto_trading

  # Test connection
  python -m trading_system.run_crypto_trading --test

  # Reconfigure settings
  python -m trading_system.run_crypto_trading --reconfigure
        """
    )

    parser.add_argument('--setup', action='store_true',
                       help='Run setup wizard')
    parser.add_argument('--test', action='store_true',
                       help='Test API connection only')
    parser.add_argument('--reconfigure', action='store_true',
                       help='Quick reconfigure settings')
    parser.add_argument('--help-config', action='store_true',
                       help='Show configuration help')

    args = parser.parse_args()

    # Handle setup
    if args.setup:
        config = run_crypto_setup_wizard()
        return

    # Handle reconfigure
    if args.reconfigure:
        config = run_crypto_reconfigure()
        return

    # Load existing config
    if not CryptoTradingConfig.exists():
        print("No configuration found. Running setup wizard...")
        config = run_crypto_setup_wizard()
    else:
        config = CryptoTradingConfig.load()

    # Validate config
    if not config.is_configured():
        print("Configuration incomplete. Please run setup: --setup")
        return

    # Handle test
    if args.test:
        test_connection(config)
        return

    # Start trading
    print("\n" + "=" * 80)
    print("     THEVOLUMEAI CRYPTO SCALPING STRATEGY")
    print("=" * 80)
    print(f"\nAPI Key: {mask_api_key(config.api_key)}")
    print(f"Mode: {'Paper Trading' if config.use_paper else '*** LIVE TRADING ***'}")
    print(f"Symbols: {', '.join(config.symbols)}")

    # Confirm before starting
    if not config.use_paper:
        print("\n" + "!" * 60)
        print("WARNING: LIVE TRADING MODE - REAL MONEY AT RISK")
        print("!" * 60)
        confirm = input("\nType 'I UNDERSTAND' to continue: ")
        if confirm != 'I UNDERSTAND':
            print("Aborted.")
            return

    confirm = input("\nType 'START' to begin trading: ")
    if confirm.upper() != 'START':
        print("Aborted.")
        return

    # Create and run trader
    runner = CryptoTradingRunner(config)

    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        print("\n\nShutdown signal received...")
        runner.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    try:
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        print("\nShutdown complete.")


if __name__ == '__main__':
    main()
