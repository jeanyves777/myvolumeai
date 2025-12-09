#!/usr/bin/env python
"""
Kraken Futures ETH Margin - LIVE Trading Runner

!!! WARNING: THIS TRADES WITH REAL MONEY !!!

Runs the LIVE trading engine on Kraken Futures Production.
Uses futures.kraken.com for real money trading.

Usage:
    python -m trading_system.run_kraken_live_trading
    python -m trading_system.run_kraken_live_trading --setup

Arguments:
    --setup     Run the configuration wizard

IMPORTANT:
- This uses REAL MONEY with LEVERAGE
- You can lose more than your position size
- Risk of LIQUIDATION exists
- Only trade what you can afford to lose
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Kraken Futures ETH Margin - LIVE Trading (REAL MONEY)"
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Run configuration setup wizard'
    )

    args = parser.parse_args()

    if args.setup:
        # Run setup wizard
        from trading_system.config.kraken_margin_config import run_kraken_live_setup_wizard
        run_kraken_live_setup_wizard()
        return

    # Import and run engine
    from trading_system.config.kraken_margin_config import KrakenLiveConfig
    from trading_system.engine.kraken_live_trading_engine import KrakenLiveTradingEngine

    print("\n" + "!" * 60)
    print("!" * 60)
    print("     KRAKEN FUTURES ETH MARGIN - LIVE TRADING")
    print("        *** REAL MONEY - MARGIN TRADING ***")
    print("!" * 60)
    print("!" * 60)

    # Load config
    if not KrakenLiveConfig.exists():
        print("\nNo LIVE configuration found. Running setup wizard...")
        from trading_system.config.kraken_margin_config import run_kraken_live_setup_wizard
        config = run_kraken_live_setup_wizard()
    else:
        config = KrakenLiveConfig.load()

    if not config.is_configured():
        print("\nLIVE configuration incomplete.")
        print("API Key, Secret, and Position Size are required.")
        print("\nRun: python -m trading_system.run_kraken_live_trading --setup")
        return

    # Show config summary
    from trading_system.config.kraken_margin_config import mask_api_key
    print(f"\nLIVE Configuration:")
    print(f"  API Key: {mask_api_key(config.api_key)}")
    print(f"  Environment: LIVE (futures.kraken.com)")
    print(f"  Symbol: {config.symbol}")
    print(f"  Position Size: ${config.position_value_usd}")
    print(f"  Leverage: {config.leverage}x")
    print(f"  Effective Exposure: ${config.position_value_usd * config.leverage}")
    print(f"  Take Profit: {config.target_profit_pct}%")
    print(f"  Stop Loss: {config.stop_loss_pct}%")
    print(f"  Max Daily Loss: {config.max_daily_loss_pct}%")

    # Create and start engine
    engine = KrakenLiveTradingEngine(config)
    engine.start()


if __name__ == "__main__":
    main()
