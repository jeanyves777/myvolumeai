#!/usr/bin/env python3
"""
THE VOLUME AI - Crypto LIVE Trading Runner

!!! WARNING: THIS TRADES WITH REAL MONEY !!!

Run LIVE trading with real-time crypto market data from Alpaca.
Uses the CryptoScalping strategy for signal generation.

Usage:
    python -m trading_system.run_crypto_live_trading [--setup] [--test]

Options:
    --setup         Force run setup wizard (even if config exists)
    --test          Test API connection only (no trading)
    --symbols       Override symbols to trade
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config.crypto_trading_config import (
    CryptoLiveTradingConfig,
    CRYPTO_LIVE_CONFIG_FILE,
    mask_api_key,
    ALPACA_CRYPTO_SYMBOLS,
)
from trading_system.engine.alpaca_client import test_connection, ALPACA_AVAILABLE
from trading_system.strategies.crypto_scalping import ALPACA_CRYPTO_SYMBOLS as STRATEGY_SYMBOLS


def print_banner():
    """Print application banner with warning."""
    print("""
================================================================================
                           THE VOLUME AI
                    Crypto LIVE Trading - Scalping Strategy
================================================================================

    ██╗    ██╗ █████╗ ██████╗ ███╗   ██╗██╗███╗   ██╗ ██████╗
    ██║    ██║██╔══██╗██╔══██╗████╗  ██║██║████╗  ██║██╔════╝
    ██║ █╗ ██║███████║██████╔╝██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
    ██║███╗██║██╔══██║██╔══██╗██║╚██╗██║██║██║╚██╗██║██║   ██║
    ╚███╔███╔╝██║  ██║██║  ██║██║ ╚████║██║██║ ╚████║╚██████╔╝
     ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝

                !!! THIS TRADES WITH REAL MONEY !!!
                  ALL TRADES USE REAL FUNDS
                  LOSSES ARE PERMANENT

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


def run_setup_wizard(config: CryptoLiveTradingConfig = None) -> CryptoLiveTradingConfig:
    """Run interactive setup wizard for LIVE trading."""
    print("\n" + "=" * 70)
    print("  !!! LIVE TRADING SETUP - REAL MONEY !!!")
    print("=" * 70)
    print()
    print("This setup configures LIVE trading with real money.")
    print("Make sure you understand the risks before proceeding.")
    print()

    if config is None:
        config = CryptoLiveTradingConfig()

    # Safety confirmation
    confirm1 = input("Type 'I UNDERSTAND' to proceed with LIVE setup: ").strip()
    if confirm1 != "I UNDERSTAND":
        print("\nSetup cancelled. Use paper trading instead:")
        print("  python -m trading_system.run_crypto_paper_trading")
        sys.exit(0)

    # API Credentials
    print("\n--- Step 1: Alpaca LIVE API Credentials ---")
    print("  Get your LIVE API keys from: https://app.alpaca.markets/")
    print("  (Make sure these are LIVE keys, NOT paper keys)")
    print()

    api_key = input(f"  LIVE API Key [{mask_api_key(config.api_key) if config.api_key else 'not set'}]: ").strip()
    if api_key:
        config.api_key = api_key

    api_secret = input(f"  LIVE API Secret [{mask_api_key(config.api_secret) if config.api_secret else 'not set'}]: ").strip()
    if api_secret:
        config.api_secret = api_secret

    # Position sizing (REQUIRED and emphasized)
    print("\n--- Step 2: Position Sizing (CRITICAL FOR LIVE) ---")
    print("  How much REAL MONEY to invest per trade?")
    print("  CAUTION: With 8 symbols at max 3 positions = up to 3 x position_size at risk")
    print()
    while True:
        current = f" [${config.fixed_position_value:.0f}]" if config.fixed_position_value > 0 else " [NOT SET]"
        pos_size = input(f"  Position value per trade{current}: ").strip()
        if pos_size:
            try:
                val = float(pos_size)
                if val > 0:
                    if val > 1000:
                        confirm_large = input(f"  WARNING: ${val} is a large position. Confirm? (yes/no): ").strip().lower()
                        if confirm_large != 'yes':
                            continue
                    config.fixed_position_value = val
                    break
                else:
                    print("  Position size must be greater than $0")
            except ValueError:
                print("  Invalid value, please enter a number")
        elif config.fixed_position_value > 0:
            break
        else:
            print("  Position size is REQUIRED for live trading")

    # Symbols
    print("\n--- Step 3: Symbols to Trade ---")
    print(f"  Available: {', '.join(STRATEGY_SYMBOLS)}")
    print(f"  Current: {', '.join(config.symbols)}")
    symbols_input = input("  Enter symbols (comma-separated, or press Enter to keep current): ").strip()
    if symbols_input:
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        valid_symbols = [s for s in symbols if s in STRATEGY_SYMBOLS]
        if valid_symbols:
            config.symbols = valid_symbols
        else:
            print("  Warning: No valid symbols entered, keeping current selection")

    # Risk parameters
    print("\n--- Step 4: Risk Parameters ---")
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
    print("\n--- Step 5: Position Limits ---")
    max_pos = input(f"  Max concurrent positions [{config.max_concurrent_positions}]: ").strip()
    if max_pos:
        try:
            config.max_concurrent_positions = int(max_pos)
        except ValueError:
            pass

    # Daily loss limit
    print("\n--- Step 6: Daily Loss Limit (SAFETY) ---")
    daily_loss = input(f"  Max daily loss (will stop trading) [${config.max_daily_loss}]: ").strip()
    if daily_loss:
        try:
            config.max_daily_loss = float(daily_loss)
        except ValueError:
            pass

    # Summary
    print("\n" + "=" * 70)
    print("  LIVE TRADING CONFIGURATION SUMMARY")
    print("=" * 70)
    display_config(config)

    # Final confirmation
    print("\n" + "=" * 70)
    print("  FINAL CONFIRMATION")
    print("=" * 70)
    print("  This will save your LIVE trading configuration.")
    print("  You can start live trading at any time after this.")
    print()

    save = input("  Save LIVE configuration? (yes/no): ").strip().lower()
    if save == 'yes':
        config.save()
        print("\n  LIVE Configuration saved!")
        print(f"  File: {CRYPTO_LIVE_CONFIG_FILE}")
    else:
        print("\n  Configuration NOT saved.")

    return config


def display_config(config: CryptoLiveTradingConfig):
    """Display current configuration."""
    from trading_system.strategies.crypto_scalping import SYMBOL_RISK_PARAMS

    print("\n--- Current LIVE Configuration ---")
    print(f"  API Key:          {mask_api_key(config.api_key) if config.api_key else 'NOT SET'}")
    print(f"  Trading Mode:     >>> LIVE (REAL MONEY) <<<")
    print(f"  Position Size:    ${config.fixed_position_value:,.2f}")
    print(f"  Max Positions:    {config.max_concurrent_positions}")
    print(f"  Daily Loss Limit: ${config.max_daily_loss}")
    print(f"  Time Filter:      {'ON' if config.use_time_filter else 'OFF'}")
    print(f"  Exit Modes:       TP and SL only (V6)")
    print(f"  Taker Fee:        {config.taker_fee_pct}%")
    print()
    print("  Per-Symbol Settings (V6.1):")
    for symbol in config.symbols:
        params = SYMBOL_RISK_PARAMS.get(symbol, {})
        tp = params.get('target_profit_pct', config.target_profit_pct)
        sl = params.get('stop_loss_pct', config.stop_loss_pct)
        score = params.get('min_entry_score', config.min_entry_score)
        print(f"    {symbol}: TP={tp}% SL={sl}% MinScore={score}")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="THE VOLUME AI Crypto LIVE Trading Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
!!! WARNING: THIS TRADES WITH REAL MONEY !!!

Examples:
    python -m trading_system.run_crypto_live_trading --setup
    python -m trading_system.run_crypto_live_trading --test
    python -m trading_system.run_crypto_live_trading --symbols BTC/USD ETH/USD
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
        help='Test API connection only (no trading)'
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

    args = parser.parse_args()

    print_banner()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Load or create configuration
    config = CryptoLiveTradingConfig.load()

    # Handle setup mode
    if args.setup:
        config = run_setup_wizard(config)
    elif not config.is_configured():
        print("LIVE trading not configured yet.\n")
        setup_now = input("Run setup wizard now? (yes/no): ").strip().lower()
        if setup_now == 'yes':
            config = run_setup_wizard()
        else:
            print("\nUse paper trading instead:")
            print("  python -m trading_system.run_crypto_paper_trading")
            sys.exit(0)

    # Override symbols if provided
    if args.symbols:
        valid_symbols = [s.upper() for s in args.symbols if s.upper() in STRATEGY_SYMBOLS]
        if valid_symbols:
            config.symbols = valid_symbols
            print(f"Using symbols: {', '.join(valid_symbols)}")

    # Override position size if provided
    if args.position_size:
        config.fixed_position_value = args.position_size
        print(f"Using position size: ${args.position_size:.2f}")

    # Validate configuration
    if not config.is_configured():
        print("\nLIVE configuration incomplete. Please run setup wizard.")
        print("  python -m trading_system.run_crypto_live_trading --setup")
        sys.exit(1)

    # Display current configuration
    display_config(config)

    # Test connection mode
    if args.test:
        print("Testing LIVE API connection...")
        if test_connection(config.api_key, config.api_secret, paper=False):
            print("\nLIVE connection successful!")

            # Get account info
            from trading_system.engine.alpaca_client import AlpacaClient
            client = AlpacaClient(config.api_key, config.api_secret, paper=False)

            account = client.get_account()
            print(f"\nLIVE Account Info:")
            print(f"  Account ID: {account['id']}")
            print(f"  Cash: ${account['cash']:,.2f}")
            print(f"  Buying Power: ${account['buying_power']:,.2f}")
            print(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")

            # Test crypto data
            print("\nTesting crypto data access...")
            for symbol in config.symbols[:3]:
                quote = client.get_latest_crypto_quote(symbol)
                if quote:
                    print(f"  {symbol}: ${quote.mid:.4f} (bid: ${quote.bid:.4f}, ask: ${quote.ask:.4f})")
                else:
                    print(f"  {symbol}: No data available")

            print("\nAll tests passed! Ready for LIVE trading.")
        else:
            print("\nLIVE connection failed. Please check your API credentials.")
            sys.exit(1)
        sys.exit(0)

    # Multiple confirmation before starting LIVE trading
    print("=" * 70)
    print("  !!! FINAL WARNING - LIVE TRADING !!!")
    print("=" * 70)
    print()
    print("  You are about to start LIVE trading with REAL MONEY.")
    print()
    print(f"  - Position Size: ${config.fixed_position_value:,.2f} per trade")
    print(f"  - Max Positions: {config.max_concurrent_positions}")
    print(f"  - Max Risk: ${config.fixed_position_value * config.max_concurrent_positions:,.2f}")
    print(f"  - Symbols: {', '.join(config.symbols)}")
    print(f"  - Daily Loss Limit: ${config.max_daily_loss}")
    print()
    print("  Losses are REAL and PERMANENT.")
    print()

    # First confirmation
    confirm1 = input("  Type 'LIVE' to confirm you want to trade with real money: ").strip()
    if confirm1 != "LIVE":
        print("\n  LIVE trading cancelled.")
        print("  Use paper trading instead:")
        print("    python -m trading_system.run_crypto_paper_trading")
        sys.exit(0)

    # Second confirmation
    confirm2 = input("  Type 'START' to begin LIVE trading NOW: ").strip()
    if confirm2 != "START":
        print("\n  LIVE trading cancelled.")
        sys.exit(0)

    print("\n  Starting LIVE trading...")
    print()

    # Run live trading
    try:
        from trading_system.engine.crypto_live_trading_engine import run_crypto_live_trading
        run_crypto_live_trading(config)
    except KeyboardInterrupt:
        print("\nShutdown requested.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
