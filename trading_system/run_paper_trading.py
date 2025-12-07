#!/usr/bin/env python3
"""
THE VOLUME AI - Paper Trading Runner

Run paper trading with real-time market data from Alpaca.

Usage:
    python -m trading_system.run_paper_trading [--setup] [--reconfigure]

Options:
    --setup         Force run setup wizard (even if config exists)
    --reconfigure   Quick reconfigure specific settings
    --test          Test API connection only
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import PaperTradingConfig, get_available_strategies
from trading_system.config.setup_wizard import run_setup_wizard, quick_reconfigure
from trading_system.engine.paper_trading_engine import run_paper_trading
from trading_system.engine.alpaca_client import test_connection, ALPACA_AVAILABLE


def print_banner():
    """Print application banner."""
    print("""
================================================================================
                           THE VOLUME AI
                    Paper Trading - 0DTE Options Strategy
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


def display_config(config: PaperTradingConfig):
    """Display current configuration."""
    from trading_system.config.paper_trading_config import mask_api_key

    print("\n--- Current Configuration ---")
    print(f"  API Key:          {mask_api_key(config.api_key)}")
    print(f"  Trading Mode:     PAPER")
    print(f"  Symbol:           {config.underlying_symbol}")
    print(f"  Position Size:    ${config.fixed_position_value:,.2f}")
    print(f"  Take Profit:      {config.target_profit_pct}%")
    print(f"  Stop Loss:        {config.stop_loss_pct}%")
    print(f"  Max Hold Time:    {config.max_hold_minutes} min")
    print(f"  Entry Window:     {config.entry_time_start} - {config.entry_time_end} EST")
    print(f"  Force Exit:       {config.force_exit_time} EST")
    print(f"  Strategy:         {config.strategy_file}")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="THE VOLUME AI Paper Trading Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m trading_system.run_paper_trading
    python -m trading_system.run_paper_trading --setup
    python -m trading_system.run_paper_trading --test
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
        help='Test API connection only'
    )

    args = parser.parse_args()

    print_banner()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Load or create configuration
    config = PaperTradingConfig.load()

    # Handle setup/reconfigure modes
    if args.setup:
        config = run_setup_wizard(force_reconfigure=True)
    elif args.reconfigure:
        config = quick_reconfigure()
    elif not config.is_configured():
        print("First-time setup required.\n")
        config = run_setup_wizard()

    # Validate configuration
    if not config.is_configured():
        print("\nConfiguration incomplete. Please run setup wizard.")
        print("  python -m trading_system.run_paper_trading --setup")
        sys.exit(1)

    # Display current configuration
    display_config(config)

    # Test connection mode
    if args.test:
        print("Testing API connection...")
        if test_connection(config.api_key, config.api_secret, paper=True):
            print("\nConnection successful!")
        else:
            print("\nConnection failed. Please check your API credentials.")
            sys.exit(1)
        sys.exit(0)

    # Confirm before starting
    print("Ready to start paper trading.")
    print("\nWARNING: This will execute trades in your Alpaca PAPER account.")
    print("Press Enter to continue, or Ctrl+C to cancel...")

    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)

    # Run paper trading
    try:
        run_paper_trading(config)
    except KeyboardInterrupt:
        print("\nShutdown requested.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
