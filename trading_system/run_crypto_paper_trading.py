#!/usr/bin/env python3
"""
THE VOLUME AI - Crypto Paper Trading Runner

Run paper trading with real-time crypto market data from Alpaca.

Crypto markets trade 24/7, so this can run anytime.
Uses the CryptoScalping strategy for signal generation.

Usage:
    python -m trading_system.run_crypto_paper_trading [--setup] [--test]

Options:
    --setup         Force run setup wizard (even if config exists)
    --test          Test API connection only
    --symbols       Override symbols to trade
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config.crypto_paper_trading_config import (
    CryptoPaperTradingConfig,
    mask_api_key,
    DEFAULT_CRYPTO_SYMBOLS,
)
from trading_system.engine.alpaca_client import test_connection, ALPACA_AVAILABLE
from trading_system.strategies.crypto_scalping import ALPACA_CRYPTO_SYMBOLS


def print_banner():
    """Print application banner."""
    print("""
================================================================================
                           THE VOLUME AI
                    Crypto Paper Trading - Scalping Strategy
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


def run_setup_wizard(config: CryptoPaperTradingConfig = None) -> CryptoPaperTradingConfig:
    """Run interactive setup wizard."""
    print("\n--- Crypto Paper Trading Setup ---\n")

    if config is None:
        config = CryptoPaperTradingConfig()

    # API Credentials
    print("Step 1: Alpaca API Credentials")
    print("  Get your API keys from: https://app.alpaca.markets/paper/dashboard/overview")
    print()

    api_key = input(f"  API Key [{mask_api_key(config.api_key) if config.api_key else 'not set'}]: ").strip()
    if api_key:
        config.api_key = api_key

    api_secret = input(f"  API Secret [{mask_api_key(config.api_secret) if config.api_secret else 'not set'}]: ").strip()
    if api_secret:
        config.api_secret = api_secret

    # Symbols
    print("\n\nStep 2: Symbols to Trade")
    print(f"  Available: {', '.join(ALPACA_CRYPTO_SYMBOLS)}")
    print(f"  Current: {', '.join(config.symbols)}")
    symbols_input = input("  Enter symbols (comma-separated, or press Enter to keep current): ").strip()
    if symbols_input:
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        valid_symbols = [s for s in symbols if s in ALPACA_CRYPTO_SYMBOLS]
        if valid_symbols:
            config.symbols = valid_symbols
        else:
            print("  Warning: No valid symbols entered, keeping current selection")

    # Position sizing (REQUIRED)
    print("\n\nStep 3: Position Sizing (REQUIRED)")
    print("  How much $ to invest per trade?")
    print("  Max risk with 9 symbols at 9 positions = 9 x position_size")
    while True:
        current = f" [${config.fixed_position_value:.0f}]" if config.fixed_position_value > 0 else ""
        pos_size = input(f"  Position value per trade{current}: ").strip()
        if pos_size:
            try:
                val = float(pos_size)
                if val > 0:
                    config.fixed_position_value = val
                    break
                else:
                    print("  Position size must be greater than $0")
            except ValueError:
                print("  Invalid value, please enter a number")
        elif config.fixed_position_value > 0:
            break  # Keep existing value
        else:
            print("  Position size is required")

    # Risk parameters
    print("\n\nStep 4: Risk Parameters")
    tp = input(f"  Take profit % [{config.target_profit_pct}]: ").strip()
    if tp:
        try:
            config.target_profit_pct = float(tp)
        except ValueError:
            pass

    sl = input(f"  Stop loss % [{config.stop_loss_pct}]: ").strip()
    if sl:
        try:
            config.stop_loss_pct = float(sl)
        except ValueError:
            pass

    trailing = input(f"  Use trailing stop? (y/n) [{'y' if config.use_trailing_stop else 'n'}]: ").strip().lower()
    if trailing in ['y', 'yes']:
        config.use_trailing_stop = True
    elif trailing in ['n', 'no']:
        config.use_trailing_stop = False

    # Max concurrent positions
    print("\n\nStep 5: Position Limits")
    max_pos = input(f"  Max concurrent positions [{config.max_concurrent_positions}]: ").strip()
    if max_pos:
        try:
            config.max_concurrent_positions = int(max_pos)
        except ValueError:
            pass

    max_hold = input(f"  Max hold time (minutes) [{config.max_hold_minutes}]: ").strip()
    if max_hold:
        try:
            config.max_hold_minutes = int(max_hold)
        except ValueError:
            pass

    # Time filter
    print("\n\nStep 6: Trading Hours")
    print("  Crypto markets are 24/7, but volume varies.")
    print("  Peak hours: 00-08 UTC (Asia) and 13-21 UTC (US)")
    time_filter = input(f"  Use time filter? (y/n) [{'y' if config.use_time_filter else 'n'}]: ").strip().lower()
    if time_filter in ['y', 'yes']:
        config.use_time_filter = True
    elif time_filter in ['n', 'no']:
        config.use_time_filter = False

    # Save configuration
    print("\n\n--- Configuration Summary ---")
    display_config(config)

    save = input("\nSave this configuration? (y/n): ").strip().lower()
    if save in ['y', 'yes']:
        config.save()
        print("\nConfiguration saved!")
    else:
        print("\nConfiguration NOT saved.")

    return config


def display_config(config: CryptoPaperTradingConfig):
    """Display current configuration."""
    from trading_system.strategies.crypto_scalping import SYMBOL_RISK_PARAMS

    print("\n--- Current Configuration ---")
    print(f"  API Key:          {mask_api_key(config.api_key) if config.api_key else 'NOT SET'}")
    print(f"  Trading Mode:     PAPER (Crypto)")
    print(f"  Position Size:    ${config.fixed_position_value:,.2f}")
    print(f"  Max Positions:    {config.max_concurrent_positions}")
    print(f"  Time Filter:      {'ON' if config.use_time_filter else 'OFF'}")
    print(f"  Exit Modes:       TP and SL only (V6)")
    print(f"  Taker Fee:        {config.taker_fee_pct}%")
    print()
    print("  Per-Symbol Settings (V6.1):")
    for symbol in config.symbols:
        params = SYMBOL_RISK_PARAMS.get(symbol, {})
        tp = params.get('target_profit_pct', config.target_profit_pct)
        sl = params.get('stop_loss_pct', config.stop_loss_pct)
        score = params.get('min_entry_score', 7)
        print(f"    {symbol}: TP={tp}% SL={sl}% MinScore={score}")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="THE VOLUME AI Crypto Paper Trading Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m trading_system.run_crypto_paper_trading
    python -m trading_system.run_crypto_paper_trading --setup
    python -m trading_system.run_crypto_paper_trading --test
    python -m trading_system.run_crypto_paper_trading --symbols BTC/USD ETH/USD
        """
    )

    parser.add_argument(
        '--setup',
        action='store_true',
        help='Force run setup wizard'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Test API connection only'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Override symbols to trade'
    )

    parser.add_argument(
        '--position-size',
        type=float,
        help='Override position size'
    )

    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip confirmation prompt and start immediately'
    )

    args = parser.parse_args()

    print_banner()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Load or create configuration
    config = CryptoPaperTradingConfig.load()

    # Handle setup mode
    if args.setup:
        config = run_setup_wizard(config)
    elif not config.is_configured():
        print("First-time setup required.\n")
        config = run_setup_wizard()

    # Override symbols if provided
    if args.symbols:
        valid_symbols = [s.upper() for s in args.symbols if s.upper() in ALPACA_CRYPTO_SYMBOLS]
        if valid_symbols:
            config.symbols = valid_symbols
            print(f"Using symbols: {', '.join(valid_symbols)}")

    # Override position size if provided
    if args.position_size:
        config.fixed_position_value = args.position_size
        print(f"Using position size: ${args.position_size:.2f}")

    # Validate configuration
    if not config.is_configured():
        print("\nConfiguration incomplete. Please run setup wizard.")
        print("  python -m trading_system.run_crypto_paper_trading --setup")
        sys.exit(1)

    # Display current configuration
    display_config(config)

    # Test connection mode
    if args.test:
        print("Testing API connection...")
        if test_connection(config.api_key, config.api_secret, paper=True):
            print("\nConnection successful!")

            # Also test crypto data
            from trading_system.engine.alpaca_client import AlpacaClient
            client = AlpacaClient(config.api_key, config.api_secret, paper=True)

            print("\nTesting crypto data access...")
            for symbol in config.symbols[:3]:  # Test first 3 symbols
                quote = client.get_latest_crypto_quote(symbol)
                if quote:
                    print(f"  {symbol}: ${quote.mid:.4f} (bid: ${quote.bid:.4f}, ask: ${quote.ask:.4f})")
                else:
                    print(f"  {symbol}: No data available")

            print("\nAll tests passed!")
        else:
            print("\nConnection failed. Please check your API credentials.")
            sys.exit(1)
        sys.exit(0)

    # Confirm before starting (unless --yes flag is set)
    if not args.yes:
        print("Ready to start crypto paper trading.")
        print("\nWARNING: This will execute trades in your Alpaca PAPER account.")
        print("  - Trades real crypto with paper money")
        print("  - Uses the CryptoScalping strategy")
        print(f"  - Will trade: {', '.join(config.symbols)}")
        print("\nPress Enter to continue, or Ctrl+C to cancel...")

        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
    else:
        print("Starting crypto paper trading (--yes flag set)...")

    # Run paper trading
    try:
        from trading_system.engine.crypto_paper_trading_engine import run_crypto_paper_trading
        run_crypto_paper_trading(config)
    except KeyboardInterrupt:
        print("\nShutdown requested.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
