#!/usr/bin/env python
"""
Kraken Futures ETH Margin - Paper Trading Runner

Runs the paper trading engine on Kraken Futures Demo environment.
Uses demo-futures.kraken.com for simulated trading.

Usage:
    python -m trading_system.run_kraken_paper_trading
    python -m trading_system.run_kraken_paper_trading --setup

Arguments:
    --setup     Run the configuration wizard
    -y          Skip confirmation prompts
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Kraken Futures ETH Margin - Paper Trading"
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Run configuration setup wizard'
    )
    parser.add_argument(
        '-y',
        action='store_true',
        help='Skip confirmation prompts'
    )

    args = parser.parse_args()

    if args.setup:
        # Run setup wizard
        from trading_system.config.kraken_margin_config import run_kraken_demo_setup_wizard
        run_kraken_demo_setup_wizard()
        return

    # Import and run engine
    from trading_system.config.kraken_margin_config import KrakenDemoConfig
    from trading_system.engine.kraken_paper_trading_engine import KrakenPaperTradingEngine

    print("\n" + "=" * 60)
    print("KRAKEN FUTURES ETH MARGIN - PAPER TRADING")
    print("=" * 60)

    # Load config
    if not KrakenDemoConfig.exists():
        print("\nNo configuration found. Running setup wizard...")
        from trading_system.config.kraken_margin_config import run_kraken_demo_setup_wizard
        config = run_kraken_demo_setup_wizard()
    else:
        config = KrakenDemoConfig.load()

    if not config.is_configured():
        print("\nConfiguration incomplete.")
        print("API Key and Secret are required.")
        print("\nRun: python -m trading_system.run_kraken_paper_trading --setup")
        return

    # Show config summary
    from trading_system.config.kraken_margin_config import mask_api_key
    print(f"\nConfiguration:")
    print(f"  API Key: {mask_api_key(config.api_key)}")
    print(f"  Environment: DEMO (demo-futures.kraken.com)")
    print(f"  Symbol: {config.symbol}")
    print(f"  Position Size: ${config.position_value_usd}")
    print(f"  Leverage: {config.leverage}x")
    print(f"  Take Profit: {config.target_profit_pct}%")
    print(f"  Stop Loss: {config.stop_loss_pct}%")

    # Confirmation
    if not args.y:
        confirm = input("\nStart paper trading? [Y/n]: ").strip().lower()
        if confirm == 'n':
            print("Cancelled.")
            return

    # Create and start engine
    engine = KrakenPaperTradingEngine(config)
    engine.start()


if __name__ == "__main__":
    main()
