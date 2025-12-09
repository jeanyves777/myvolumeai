#!/usr/bin/env python3
"""
THE VOLUME AI - MARA Continuous Trading Runner

Run MARA continuous options trading with Alpaca.
This strategy trades multiple times per day with proper entry validation.

Usage:
    python -m trading_system.run_mara_continuous_trading [--test] [-y]

Options:
    --test    Test Alpaca connection only (don't start trading)
    -y        Skip confirmation prompt

Key Features:
    - ATM contract selection at time of each entry
    - Weekly expiry options (not 0DTE)
    - Volume-based entry validation
    - Automatic re-entry after cooldown
    - Full trade logging with Greeks
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import PaperTradingConfig
from trading_system.strategies.mara_continuous_momentum import MARAContinuousMomentumConfig
from trading_system.engine.mara_continuous_trading_engine import MARAContinuousTradingEngine
from trading_system.engine.alpaca_client import test_connection, ALPACA_AVAILABLE


def print_banner():
    """Print application banner."""
    print("""
================================================================================
                           THE VOLUME AI
           MARA Continuous Trading - ATM Weekly Options Strategy
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


def display_config(config: MARAContinuousMomentumConfig, api_key: str):
    """Display current configuration."""
    from trading_system.config.paper_trading_config import mask_api_key

    print("\n--- MARA CONTINUOUS Trading Configuration ---")
    print(f"  API Key:          {mask_api_key(api_key)}")
    print(f"  Trading Mode:     PAPER (Continuous)")
    print(f"  Symbol:           MARA (Marathon Digital)")
    print()
    print("  --- Position Settings ---")
    print(f"  Position Size:    ${config.fixed_position_value:,.2f}")
    print(f"  Take Profit:      {config.target_profit_pct}% (LIMIT ORDER on exchange)")
    print(f"  Stop Loss:        {config.stop_loss_pct}% (monitored internally)")
    print()
    print("  --- Trading Window ---")
    print(f"  Entry Window:     {config.entry_time_start} - {config.entry_time_end} EST")
    print(f"  Force Exit:       {config.force_exit_time} EST")
    print(f"  Max Hold:         {config.max_hold_minutes} minutes per trade")
    print()
    print("  --- Continuous Trading ---")
    print(f"  Max Trades/Day:   {config.max_trades_per_day}")
    print(f"  Cooldown:         {config.cooldown_minutes} minutes between trades")
    print()
    print("  --- Contract Selection ---")
    print(f"  Expiry:           Weekly (Friday)")
    print(f"  Min DTE:          {config.min_days_to_expiry} days")
    print(f"  Max DTE:          {config.max_days_to_expiry} days")
    print(f"  Max Spread:       {config.max_bid_ask_spread_pct}%")
    print()
    print("  --- Entry Validation ---")
    print(f"  Volume Spike:     {config.volume_spike_multiplier}x average")
    print(f"  Min Volume:       {config.min_volume_threshold:,}")
    print(f"  Dual Confirm:     {'Yes' if config.require_dual_confirmation else 'No'}")
    print()
    print("  --- How It Works ---")
    print("  1. Monitor MARA price and volume continuously")
    print("  2. Wait for volume spike + momentum confirmation")
    print("  3. Select ATM contract (weekly expiry)")
    print("  4. Enter with TP limit order on exchange")
    print("  5. Monitor for TP/SL/max hold")
    print("  6. After exit, wait cooldown then repeat")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="THE VOLUME AI - MARA Continuous Trading Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m trading_system.run_mara_continuous_trading
    python -m trading_system.run_mara_continuous_trading --test
    python -m trading_system.run_mara_continuous_trading -y
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

    # Create MARA continuous config
    mara_config = MARAContinuousMomentumConfig(
        underlying_symbol="MARA",
        fixed_position_value=200.0,
        target_profit_pct=7.5,
        stop_loss_pct=25.0,
        entry_time_start="09:35:00",
        entry_time_end="15:30:00",
        force_exit_time="15:50:00",
        max_hold_minutes=30,
        cooldown_minutes=5,
        max_trades_per_day=10,
        volume_spike_multiplier=1.5,
        min_volume_threshold=10000,
        use_weekly_expiry=True,
        min_days_to_expiry=1,
        max_days_to_expiry=7,
        max_bid_ask_spread_pct=15.0,
        min_signal_score=3,
        require_dual_confirmation=True,
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
        print("Ready to start MARA CONTINUOUS trading.")
        print("\nWARNING: This will execute MULTIPLE trades in your Alpaca PAPER account.")
        print("Press Enter to continue, or Ctrl+C to cancel...")

        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    # Run MARA continuous trading
    import signal

    engine = MARAContinuousTradingEngine(
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
