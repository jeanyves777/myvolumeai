#!/usr/bin/env python3
"""
THE VOLUME AI - IB MARA Paper Trading Runner

Run MARA 0DTE Options paper trading with Interactive Brokers.

Usage:
    python -m trading_system.run_ib_mara_paper_trading [--test]

Options:
    --test    Test IB connection only (don't start trading)
    -y        Skip confirmation prompt

Requirements:
    - TWS or IB Gateway running (paper trading mode)
    - Port 7497 for paper trading
    - ib_insync package installed
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Default config path
CONFIG_PATH = os.path.expanduser("~/.thevolumeai/ib_mara_trading_config.json")


def print_banner():
    """Print application banner."""
    print("""
================================================================================
                           THE VOLUME AI
              IB Paper Trading - MARA 0DTE Options Strategy
================================================================================
    """)


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        from ib_insync import IB
    except ImportError:
        missing.append("ib_insync")

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


def display_config(config):
    """Display current configuration."""
    print("\n--- Current Configuration ---")
    print(f"  IB Host:          {config.ib_host}")
    print(f"  IB Port:          {config.ib_port} {'(PAPER)' if config.ib_port == 7497 else '(LIVE - CAREFUL!)'}")
    print(f"  Client ID:        {config.ib_client_id}")
    print(f"  Symbol:           {config.underlying_symbol}")
    print(f"  Position Size:    ${config.fixed_position_value:,.2f}")
    print(f"  Take Profit:      {config.target_profit_pct}% (LIMIT ORDER)")
    print(f"  Stop Loss:        {config.stop_loss_pct}% (monitored internally)")
    print(f"  Entry Window:     {config.entry_time_start} - {config.entry_time_end} EST")
    print(f"  Force Exit:       {config.force_exit_time} EST")
    print(f"  Max Hold:         {config.max_hold_minutes} minutes")
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


def test_ib_connection(config):
    """Test connection to IB."""
    from trading_system.engine.ib_client import IBClient

    print(f"Testing connection to IB at {config.ib_host}:{config.ib_port}...")

    client = IBClient(
        host=config.ib_host,
        port=config.ib_port,
        client_id=config.ib_client_id
    )

    if client.connect():
        print("\n[SUCCESS] Connected to Interactive Brokers!")

        # Get account info
        account_info = client.get_account_info()
        print(f"\nAccount Info:")
        print(f"  Net Liquidation: ${account_info.get('NetLiquidation', 0):,.2f}")
        print(f"  Total Cash:      ${account_info.get('TotalCashValue', 0):,.2f}")
        print(f"  Buying Power:    ${account_info.get('BuyingPower', 0):,.2f}")

        # Test market data
        print(f"\nTesting {config.underlying_symbol} market data...")
        quote = client.get_latest_stock_quote(config.underlying_symbol)
        if quote:
            print(f"  {config.underlying_symbol}: ${quote.mid:.2f} (Bid: ${quote.bid:.2f} / Ask: ${quote.ask:.2f})")
        else:
            print(f"  [WARN] Could not get quote for {config.underlying_symbol}")

        client.disconnect()
        return True
    else:
        print("\n[ERROR] Failed to connect to Interactive Brokers!")
        print("\nPlease check:")
        print("  1. TWS or IB Gateway is running")
        print("  2. API connections are enabled in TWS settings")
        print("  3. Port 7497 (paper) or 7496 (live) is correct")
        print("  4. Socket client ID is not already in use")
        return False


def create_default_config():
    """Create default configuration file."""
    import json

    config_dir = os.path.dirname(CONFIG_PATH)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    default_config = {
        "ib_host": "127.0.0.1",
        "ib_port": 7497,
        "ib_client_id": 2,
        "underlying_symbol": "MARA",
        "fixed_position_value": 2000.0,
        "target_profit_pct": 7.5,
        "stop_loss_pct": 25.0,
        "entry_time_start": "09:30:00",
        "entry_time_end": "15:45:00",
        "force_exit_time": "15:50:00",
        "max_hold_minutes": 30,
        "poll_interval_seconds": 5
    }

    with open(CONFIG_PATH, 'w') as f:
        json.dump(default_config, f, indent=2)

    print(f"Created default config at: {CONFIG_PATH}")
    return CONFIG_PATH


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="THE VOLUME AI - IB MARA Paper Trading Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m trading_system.run_ib_mara_paper_trading
    python -m trading_system.run_ib_mara_paper_trading --test
    python -m trading_system.run_ib_mara_paper_trading -y
        """
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Test IB connection only'
    )

    parser.add_argument(
        '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=CONFIG_PATH,
        help=f'Config file path (default: {CONFIG_PATH})'
    )

    args = parser.parse_args()

    print_banner()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Load configuration
    from trading_system.engine.ib_paper_trading_engine import load_ib_config, IBPaperTradingEngine

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Creating default configuration...")
        config_path = create_default_config()

    try:
        config = load_ib_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Display current configuration
    display_config(config)

    # Test connection mode
    if args.test:
        if test_ib_connection(config):
            print("\nConnection test successful!")
        else:
            print("\nConnection test failed.")
            sys.exit(1)
        sys.exit(0)

    # Confirm before starting
    if not args.y:
        print("Ready to start IB paper trading for MARA.")
        print("\nWARNING: This will execute trades in your IB PAPER account.")
        print("\nRequirements:")
        print("  - TWS or IB Gateway must be running")
        print("  - API connections enabled")
        print("  - Paper trading mode (port 7497)")
        print("\nPress Enter to continue, or Ctrl+C to cancel...")

        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    # Run paper trading
    try:
        engine = IBPaperTradingEngine(config)
        engine.start()
    except KeyboardInterrupt:
        print("\nShutdown requested.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
