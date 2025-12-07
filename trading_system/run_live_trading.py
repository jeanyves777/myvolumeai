#!/usr/bin/env python3
"""
THE VOLUME AI - LIVE TRADING Runner

!!! WARNING: THIS TRADES WITH REAL MONEY !!!

Run live trading with real-time market data and real order execution.

Usage:
    python -m trading_system.run_live_trading [--setup] [--reconfigure] [--test] [--log]

Options:
    --setup         Force run setup wizard (even if config exists)
    --reconfigure   Quick reconfigure specific settings
    --test          Test API connection only (no trading)
    --log           View trade log history
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config.live_trading_config import (
    LiveTradingConfig,
    get_trade_log,
    mask_api_key,
)
from trading_system.config.live_setup_wizard import (
    run_live_setup_wizard,
    quick_reconfigure_live,
)
from trading_system.engine.live_trading_engine import run_live_trading
from trading_system.engine.alpaca_client import AlpacaClient, ALPACA_AVAILABLE


def print_banner():
    """Print application banner."""
    print("""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                           THE VOLUME AI
                    !!! LIVE TRADING - REAL MONEY !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """)


def print_danger(msg: str):
    """Print danger message."""
    print("\n" + "!" * 60)
    print(f"  !!! {msg} !!!")
    print("!" * 60 + "\n")


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


def display_config(config: LiveTradingConfig):
    """Display current configuration."""
    print("\n--- LIVE Trading Configuration ---")
    print(f"  API Key:          {mask_api_key(config.api_key)}")
    print(f"  Trading Mode:     *** LIVE (REAL MONEY) ***")
    print(f"  Symbol:           {config.underlying_symbol}")
    print(f"  Position Size:    ${config.fixed_position_value:,.2f}")
    print(f"  Max Position:     ${config.max_position_value:,.2f}")
    print(f"  Max Daily Loss:   ${config.max_daily_loss:,.2f}")
    print(f"  Max Trades/Day:   {config.max_trades_per_day}")
    print(f"  Take Profit:      {config.target_profit_pct}%")
    print(f"  Stop Loss:        {config.stop_loss_pct}%")
    print(f"  Max Hold Time:    {config.max_hold_minutes} min")
    print(f"  Entry Window:     {config.entry_time_start} - {config.entry_time_end} EST")
    print(f"  Force Exit:       {config.force_exit_time} EST")
    print(f"  Strategy:         {config.strategy_file}")
    print(f"  Confirm Trades:   {'Yes' if config.require_confirmation else 'No'}")

    # Show today's stats if available
    if config.last_trade_date:
        print(f"\n  --- Today's Stats ({config.last_trade_date}) ---")
        print(f"  Trades Today:     {config.trades_today}")
        print(f"  P&L Today:        ${config.pnl_today:,.2f}")

    print()


def display_trade_log():
    """Display trade log history."""
    trades = get_trade_log()

    if not trades:
        print("\nNo trades logged yet.")
        return

    print("\n" + "=" * 80)
    print("  LIVE TRADE LOG")
    print("=" * 80)

    for i, trade in enumerate(trades[-20:], 1):  # Show last 20
        trade_type = trade.get('type', 'UNKNOWN')
        symbol = trade.get('symbol', 'N/A')
        timestamp = trade.get('timestamp', 'N/A')

        if trade_type == 'ENTRY':
            print(f"\n  [{i}] ENTRY - {timestamp}")
            print(f"      Symbol: {symbol}")
            print(f"      Signal: {trade.get('signal', 'N/A')}")
            print(f"      Qty: {trade.get('qty', 0)} @ ${trade.get('entry_price', 0):.2f}")
            print(f"      Value: ${trade.get('position_value', 0):.2f}")

        elif trade_type == 'EXIT':
            print(f"\n  [{i}] EXIT - {timestamp}")
            print(f"      Symbol: {symbol}")
            print(f"      Reason: {trade.get('reason', 'N/A')}")
            print(f"      Entry: ${trade.get('entry_price', 0):.2f} -> Exit: ${trade.get('exit_price', 0):.2f}")
            pnl = trade.get('pnl', 0)
            pnl_pct = trade.get('pnl_pct', 0)
            print(f"      P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)")
            print(f"      Hold: {trade.get('hold_time_minutes', 0):.1f} min")

    print("\n" + "=" * 80)
    print(f"  Total trades in log: {len(trades)}")
    print("=" * 80)


def test_live_connection(config: LiveTradingConfig):
    """Test live API connection."""
    print("\nTesting LIVE API connection...")
    print_danger("This will connect to your LIVE trading account")

    try:
        client = AlpacaClient(
            api_key=config.api_key,
            api_secret=config.api_secret,
            paper=False  # LIVE
        )

        account = client.get_account()

        print("\n" + "=" * 50)
        print("  LIVE CONNECTION SUCCESSFUL")
        print("=" * 50)
        print(f"  Account ID: {account['id']}")
        print(f"  Status: {account['status']}")
        print(f"  Cash: ${account['cash']:,.2f}")
        print(f"  Buying Power: ${account['buying_power']:,.2f}")
        print(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
        print(f"  PDT Status: {account['pattern_day_trader']}")
        print("=" * 50)

        # Get positions
        positions = client.get_positions()
        if positions:
            print("\n  Current Positions:")
            for pos in positions:
                print(f"    {pos['symbol']}: {pos['qty']} @ ${pos['avg_entry_price']:.2f}")
                print(f"      P&L: ${pos['unrealized_pl']:.2f} ({pos['unrealized_plpc']:+.1f}%)")

        # Get open orders
        orders = client.get_orders(status='open')
        if orders:
            print("\n  Open Orders:")
            for o in orders:
                print(f"    {o['symbol']}: {o['side']} {o['qty']} ({o['status']})")

        return True

    except Exception as e:
        print(f"\nConnection FAILED: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="THE VOLUME AI LIVE Trading Runner - REAL MONEY",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
!!! WARNING: THIS TRADES WITH REAL MONEY !!!

Examples:
    python -m trading_system.run_live_trading           # Start live trading
    python -m trading_system.run_live_trading --setup   # Configure live trading
    python -m trading_system.run_live_trading --test    # Test connection only
    python -m trading_system.run_live_trading --log     # View trade log
        """
    )

    parser.add_argument(
        '--setup',
        action='store_true',
        help='Force run setup wizard'
    )

    parser.add_argument(
        '--reconfigure',
        action='store_true',
        help='Quick reconfigure settings'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Test API connection only (no trading)'
    )

    parser.add_argument(
        '--log',
        action='store_true',
        help='View trade log history'
    )

    args = parser.parse_args()

    print_banner()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # View log mode
    if args.log:
        display_trade_log()
        sys.exit(0)

    # Load or create configuration
    config = LiveTradingConfig.load()

    # Handle setup/reconfigure modes
    if args.setup:
        config = run_live_setup_wizard(force_reconfigure=True)
    elif args.reconfigure:
        config = quick_reconfigure_live()
    elif not config.is_configured():
        print_danger("LIVE TRADING SETUP REQUIRED")
        print("You must configure live trading before use.")
        print("\nWe STRONGLY recommend paper trading first!")
        print("  python -m trading_system.run_paper_trading")

        response = input("\nContinue to live setup? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\nSetup cancelled.")
            sys.exit(0)

        config = run_live_setup_wizard()

    # Validate configuration
    if not config.is_configured():
        print("\nConfiguration incomplete. Please run setup wizard.")
        print("  python -m trading_system.run_live_trading --setup")
        sys.exit(1)

    # Display current configuration
    display_config(config)

    # Test connection mode
    if args.test:
        if test_live_connection(config):
            print("\nLive connection test successful!")
        else:
            print("\nLive connection test FAILED. Check your API credentials.")
            sys.exit(1)
        sys.exit(0)

    # Multiple confirmations before live trading
    print_danger("YOU ARE ABOUT TO START LIVE TRADING WITH REAL MONEY")

    print("""
  Before continuing, please confirm:
  1. You understand this will trade REAL MONEY
  2. You have tested thoroughly with paper trading
  3. You are using funds you can afford to lose
  4. You have reviewed and accept the risk limits
""")

    response = input("Type 'I UNDERSTAND' to continue: ").strip()
    if response.upper() != 'I UNDERSTAND':
        print("\nLive trading cancelled.")
        sys.exit(0)

    # Final warning
    print_danger("LAST CHANCE TO CANCEL")
    response = input("Press Enter to start LIVE trading, or Ctrl+C to cancel...")

    # Run live trading
    try:
        run_live_trading(config)
    except KeyboardInterrupt:
        print("\nShutdown requested.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
