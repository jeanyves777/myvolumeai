#!/usr/bin/env python3
"""
THE VOLUME AI - MARA Paper Trading Runner

Run MARA 0DTE Options paper trading with Alpaca.

Usage:
    python -m trading_system.run_mara_paper_trading [--test] [-y]

Options:
    --test    Test Alpaca connection only (don't start trading)
    -y        Skip confirmation prompt

Requirements:
    - Alpaca paper trading account
    - Alpaca API key and secret configured
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import PaperTradingConfig
from trading_system.strategies.mara_0dte_momentum import MARADaily0DTEMomentumConfig
from trading_system.engine.mara_paper_trading_engine import MARAPaperTradingEngine
from trading_system.engine.alpaca_client import test_connection, ALPACA_AVAILABLE


def print_banner():
    """Print application banner."""
    print("""
================================================================================
                           THE VOLUME AI
                 MARA Paper Trading - 0DTE Options Strategy
================================================================================
    """)


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    if not ALPACA_AVAILABLE:
        missing.append("alpaca-py")

    try:
        import pytz
    except ImportError:
        missing.append("pytz")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False

    return True


def display_config(config: MARADaily0DTEMomentumConfig, api_key: str):
    """Display current configuration."""
    from trading_system.config.paper_trading_config import mask_api_key

    print("\n--- MARA Trading Configuration ---")
    print(f"  API Key:          {mask_api_key(api_key)}")
    print(f"  Trading Mode:     PAPER")
    print(f"  Symbol:           MARA (Marathon Digital)")
    print(f"  Position Size:    ${config.fixed_position_value:,.2f}")
    print(f"  Take Profit:      {config.target_profit_pct}% (LIMIT ORDER on exchange)")
    print(f"  Stop Loss:        {config.stop_loss_pct}% (monitored internally)")
    print(f"  Entry Window:     {config.entry_time_start} - {config.entry_time_end} EST")
    print(f"  Force Exit:       {config.force_exit_time} EST (if no TP/SL hit)")
    print()
    print("  --- Order Management ---")
    print("  Alpaca only supports MARKET and LIMIT orders for options.")
    print("  TP: Placed as LIMIT SELL order on Alpaca (auto-fills at target)")
    print("  SL: Monitored internally (market sell when triggered)")
    print()
    print("  --- DUAL SIGNAL VALIDATION ---")
    print("  Both methods must AGREE before executing a trade:")
    print()
    print("  METHOD 1 - Technical Scoring (1-MIN bars):")
    print("    - EMA Stack (Price vs EMA9 vs EMA20)")
    print("    - VWAP position")
    print("    - RSI momentum")
    print("    - MACD crossover")
    print("    - Bollinger Bands position")
    print("    - Volume confirmation")
    print()
    print("  METHOD 2 - Price Action (5-MIN bars):")
    print("    - Candle color pattern")
    print("    - Higher highs / Lower lows trend")
    print("    - Price vs 5-bar average")
    print("    - 5-bar momentum")
    print("    - Last bar strength")
    print()
    print("  Signal Decision:")
    print("    - BOTH BULLISH -> Buy CALLs")
    print("    - BOTH BEARISH -> Buy PUTs")
    print("    - CONFLICT/NEUTRAL -> NO TRADE")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="THE VOLUME AI - MARA Paper Trading Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m trading_system.run_mara_paper_trading
    python -m trading_system.run_mara_paper_trading --test
    python -m trading_system.run_mara_paper_trading -y
        """
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Test Alpaca connection only'
    )

    parser.add_argument(
        '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )

    args = parser.parse_args()

    print_banner()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Load Alpaca credentials from existing config
    alpaca_config = PaperTradingConfig.load()
    if not alpaca_config.is_configured():
        print("ERROR: Alpaca API not configured.")
        print("Please run: python -m trading_system.run_paper_trading --setup")
        sys.exit(1)

    # Create MARA-specific config
    mara_config = MARADaily0DTEMomentumConfig(
        underlying_symbol="MARA",
        fixed_position_value=200.0,
        target_profit_pct=7.5,
        stop_loss_pct=25.0,
        entry_time_start="09:30:00",
        entry_time_end="15:45:00",
        force_exit_time="15:50:00",
        max_hold_minutes=30,
        max_trades_per_day=1,
        poll_interval_seconds=10,
    )

    # Display current configuration
    display_config(mara_config, alpaca_config.api_key)

    # Test connection mode
    if args.test:
        print("Testing Alpaca connection...")
        if test_connection(alpaca_config.api_key, alpaca_config.api_secret, paper=True):
            print("\nConnection successful!")

            # Test MARA data
            from trading_system.engine.alpaca_client import AlpacaClient
            client = AlpacaClient(
                api_key=alpaca_config.api_key,
                api_secret=alpaca_config.api_secret,
                paper=True
            )
            quote = client.get_latest_stock_quote("MARA")
            if quote:
                print(f"\nMARA Quote: ${quote.mid:.2f} (Bid: ${quote.bid:.2f} / Ask: ${quote.ask:.2f})")
            else:
                print("\nWARNING: Could not get MARA quote")
        else:
            print("\nConnection failed. Please check your API credentials.")
            sys.exit(1)
        sys.exit(0)

    # Confirm before starting
    if not args.y:
        print("Ready to start MARA paper trading.")
        print("\nWARNING: This will execute trades in your Alpaca PAPER account.")
        print("Press Enter to continue, or Ctrl+C to cancel...")

        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    # Run MARA paper trading
    import signal

    engine = MARAPaperTradingEngine(
        config=mara_config,
        api_key=alpaca_config.api_key,
        api_secret=alpaca_config.api_secret
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        engine.stop()

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        engine.run()
    except KeyboardInterrupt:
        print("\nShutdown requested.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
