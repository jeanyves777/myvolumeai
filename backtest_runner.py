"""
NautilusTrader Backtest Runner - Production Version with Data Pipeline Integration
Executes backtests using the NautilusTrader BacktestEngine with pre-collected data
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List, Optional
from pathlib import Path
import math
import time

import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def clean_for_json(obj):
    """Recursively clean an object for JSON serialization, replacing NaN/Inf with None"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer, np.floating)):
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    elif pd.isna(obj):
        return None
    return obj


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


# ═══════════════════════════════════════════════════════════════════════
# NAUTILUS TRADER IMPORTS
# ═══════════════════════════════════════════════════════════════════════

from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD, USDT, EUR, GBP, BTC, ETH
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.test_kit.providers import TestInstrumentProvider, TestDataProvider
from nautilus_trader.persistence.wranglers import TradeTickDataWrangler, QuoteTickDataWrangler
from nautilus_trader.examples.strategies.ema_cross import EMACross, EMACrossConfig

# Import custom strategy
try:
    from nautilus_trader.examples.strategies.coin_daily_0dte_momentum_v2 import (
        COINDaily0DTEMomentum,
        COINDaily0DTEMomentumConfig
    )
except ImportError:
    print("⚠️  Warning: Could not import COIN Daily 0DTE Momentum V2 strategy")
    COINDaily0DTEMomentum = None
    COINDaily0DTEMomentumConfig = None

# Import data providers
from .data_providers import fetch_historical_data
from .options_utils import (
    generate_options_contracts_by_expiry,
    get_expiries_by_frequency,
    calculate_strike_price
)
from .data_collection_pipeline import DataCollectionPipeline


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Map asset classes to currencies
CURRENCY_MAP = {
    'stocks': USD,
    'crypto': USDT,
    'forex': USD,
    'commodities': USD,
    'options': USD,
}

# Map symbols to sample data files (for demo purposes)
SAMPLE_DATA_MAP = {
    'BTC/USD': 'binance/btcusdt-trades.csv',
    'ETH/USD': 'binance/ethusdt-trades.csv',
    'ETHUSDT': 'binance/ethusdt-trades.csv',
    'AUD/USD': 'truefx/audusd-ticks.csv',
    'USD/JPY': 'truefx/usdjpy-ticks.csv',
}


# ═══════════════════════════════════════════════════════════════════════
# MAIN BACKTEST RUNNER CLASS
# ═══════════════════════════════════════════════════════════════════════

class NautilusBacktestRunner:
    """Run backtests using NautilusTrader engine with data pipeline integration"""

    def __init__(self, db_service):
        self.db_service = db_service
        self.test_data_dir = Path("/var/www/nautilus_trader-develop/tests/test_data")

    async def run_backtest(
        self,
        backtest_id: str,
        strategy_id: str,
        symbols: List[str],
        venues: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
        asset_class: str = 'stocks',
        options_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a backtest using NautilusTrader

        Args:
            backtest_id: Unique backtest identifier
            strategy_id: Strategy to test
            symbols: Trading symbols (for options, use underlying stock symbol)
            venues: Trading venues
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            asset_class: Asset class (stocks, crypto, forex, commodities, options)
            options_params: For options, dict with expiry_frequency, option_types, etc.

        Returns:
            Backtest results dictionary
        """
        backtest_start_time = time.time()
        
        try:
            print("\n" + "=" * 100)
            print("🚀 NAUTILUS BACKTEST RUNNER - STARTING EXECUTION")
            print("=" * 100)
            print(f"Backtest ID:      {backtest_id}")
            print(f"Strategy ID:      {strategy_id}")
            print(f"Symbols:          {', '.join(symbols)}")
            print(f"Period:           {start_date} to {end_date}")
            print(f"Initial Capital:  ${initial_capital:,.2f}")
            print(f"Asset Class:      {asset_class.upper()}")
            print("=" * 100 + "\n")

            # Check environment variables
            polygon_key = os.getenv('POLYGON_API_KEY')
            alpaca_key = os.getenv('ALPACA_API_KEY')
            print(f"🔑 API Keys Check:")
            print(f"   POLYGON_API_KEY: {'✓ Set (' + polygon_key[:10] + '...)' if polygon_key else '✗ NOT SET'}")
            print(f"   ALPACA_API_KEY:  {'✓ Set (' + alpaca_key[:10] + '...)' if alpaca_key else '✗ NOT SET'}")
            print()

            # Update status to running
            await self.db_service.execute(
                "UPDATE strategy_backtests SET status = 'running', updated_at = NOW() WHERE id = $1",
                backtest_id
            )

            # ═══════════════════════════════════════════════════════════
            # PHASE 1: DATA PRE-COLLECTION (Options Only)
            # ═══════════════════════════════════════════════════════════
            collected_data = None
            
            if asset_class == 'options' and options_params:
                print("\n" + "=" * 100)
                print("📦 PHASE 1: DATA PRE-COLLECTION")
                print("=" * 100)
                print("ℹ️  Pre-collecting all data before backtest for robust execution...")
                print()

                phase_start = time.time()

                try:
                    pipeline = DataCollectionPipeline()
                    expiry_frequency = options_params.get('expiry_frequency', 'weekly')
                    option_types = options_params.get('option_types', ['CALL', 'PUT'])

                    collected_data = await pipeline.collect_all_data_for_backtest(
                        symbol=symbols[0],
                        start_date=start_date,
                        end_date=end_date,
                        expiry_frequency=expiry_frequency,
                        option_types=option_types,
                        strike_selection=options_params.get('strike_selection', 'ATM'),
                        underlying_timeframe='1Min'
                    )

                    print(f"\n✅ Phase 1 Complete ({format_duration(time.time() - phase_start)})")
                    print(f"   Underlying bars:      {len(collected_data['underlying']):,}")
                    print(f"   Options contracts:    {len(collected_data['options']):,}")
                    print(f"   Success rate:         {collected_data['quality_report']['success_rate']:.1%}")
                    print()

                except Exception as e:
                    print(f"\n⚠️  Data pre-collection failed: {e}")
                    print(f"   Will fall back to on-demand loading...")
                    collected_data = None
                    import traceback
                    traceback.print_exc()

            # ═══════════════════════════════════════════════════════════
            # PHASE 2: STRATEGY CONFIGURATION LOADING
            # ═══════════════════════════════════════════════════════════
            print("\n" + "=" * 100)
            print("📋 PHASE 2: STRATEGY CONFIGURATION LOADING")
            print("=" * 100)

            phase_start = time.time()

            # Fetch strategy from database
            strategy_data = await self.db_service.fetch_one(
                "SELECT * FROM trading_strategies WHERE id = $1",
                strategy_id
            )

            if not strategy_data:
                raise Exception(f"Strategy {strategy_id} not found")

            strategy_name = strategy_data['name']
            strategy_display = strategy_data['display_name']
            config_schema = strategy_data.get('config_schema', {})

            # Parse config_schema if it's a string
            if isinstance(config_schema, str):
                config_schema = json.loads(config_schema)

            print(f"✓ Strategy loaded: {strategy_display} ({strategy_name})")
            print(f"✅ Phase 2 Complete ({format_duration(time.time() - phase_start)})\n")

            # ═══════════════════════════════════════════════════════════
            # PHASE 3: BACKTEST ENGINE CONFIGURATION
            # ═══════════════════════════════════════════════════════════
            print("=" * 100)
            print("📋 PHASE 3: BACKTEST ENGINE CONFIGURATION")
            print("=" * 100)

            phase_start = time.time()

            # Get currency for asset class
            base_currency = CURRENCY_MAP.get(asset_class, USD)

            # Configure backtest engine
            config = BacktestEngineConfig(
                trader_id=TraderId(f"BACKTESTER-{backtest_id[:8]}"),
                logging=LoggingConfig(
                    log_level="INFO",
                    log_level_file="DEBUG",
                    bypass_logging=False,
                ),
            )

            # Build the backtest engine
            engine = BacktestEngine(config=config)
            print(f"✓ Engine initialized: {engine.trader.id}")

            # Determine venue
            venue_name = venues[0] if venues else self._get_venue_for_symbol(symbols[0], asset_class)
            venue = Venue(venue_name)
            print(f"✓ Venue: {venue_name}")

            # Create fill model
            fill_model = FillModel(
                prob_fill_on_limit=0.5,
                prob_fill_on_stop=0.95,
                prob_slippage=0.3,
                random_seed=42,
            )

            # Add trading venue
            engine.add_venue(
                venue=venue,
                oms_type=OmsType.NETTING,
                account_type=AccountType.MARGIN if asset_class in ['forex', 'crypto'] else AccountType.CASH,
                base_currency=base_currency,
                starting_balances=[Money(initial_capital, base_currency)],
                fill_model=fill_model,
            )
            print(f"✓ Venue added with ${initial_capital:,.2f} starting capital")
            print(f"✅ Phase 3 Complete ({format_duration(time.time() - phase_start)})\n")

            # ═══════════════════════════════════════════════════════════
            # PHASE 4: DATA LOADING & VALIDATION
            # ═══════════════════════════════════════════════════════════
            print("=" * 100)
            print("📋 PHASE 4: DATA LOADING & VALIDATION")
            print("=" * 100)

            phase_start = time.time()

            data_loaded = False
            instruments_added = 0
            total_data_points = 0

            # Load data based on asset class
            if asset_class == 'options' and options_params:
                
                # ═══════════════════════════════════════════════════════
                # OPTIONS: USE PRE-COLLECTED DATA IF AVAILABLE
                # ═══════════════════════════════════════════════════════
                if collected_data and len(collected_data.get('options', {})) > 0:
                    print("📊 Using pre-collected data from Phase 1...\n")

                    # STEP 1: Add underlying instrument and load its data
                    for symbol in symbols:
                        print(f"📈 Processing underlying: {symbol}")

                        underlying_instrument = self._get_instrument(symbol, venue_name, 'stocks')
                        if underlying_instrument:
                            engine.add_instrument(underlying_instrument)
                            instruments_added += 1
                            print(f"   ✓ Underlying instrument added: {underlying_instrument.id}")

                            # Convert pre-collected DataFrame to Nautilus Bar objects
                            underlying_bars = self._convert_dataframe_to_bars(
                                collected_data['underlying'],
                                underlying_instrument
                            )

                            if underlying_bars and len(underlying_bars) > 0:
                                engine.add_data(underlying_bars)
                                data_loaded = True
                                total_data_points += len(underlying_bars)
                                print(f"   ✓ Loaded {len(underlying_bars):,} underlying bars from cache\n")
                            else:
                                print(f"   ⚠️  Could not convert underlying data\n")
                        else:
                            print(f"   ⚠️  Could not create underlying instrument\n")

                    # STEP 2: Add options instruments and load their data
                    print(f"📊 Loading {len(collected_data['options'])} pre-collected options contracts...")

                    contracts_loaded = 0
                    contracts_failed = 0

                    for contract_symbol, contract_df in collected_data['options'].items():
                        # Parse contract symbol to extract components
                        # Format: COIN20241115C00150000
                        try:
                            underlying_symbol = symbols[0]
                            
                            # Extract expiry (8 digits after symbol)
                            expiry_str = contract_symbol[len(underlying_symbol):len(underlying_symbol)+8]
                            expiry_date = f"{expiry_str[0:4]}-{expiry_str[4:6]}-{expiry_str[6:8]}"
                            
                            # Extract option type (C or P)
                            option_type = contract_symbol[len(underlying_symbol)+8]
                            
                            # Extract strike (remaining digits / 1000)
                            strike_str = contract_symbol[len(underlying_symbol)+9:]
                            strike = int(strike_str) / 1000.0

                            # Create instrument
                            expiry_params = {
                                'expiry_date': expiry_date,
                                'strike_price': strike,
                                'option_type': option_type,
                                'trade_direction': 'BUY'
                            }

                            instrument = self._get_instrument(
                                underlying_symbol,
                                venue_name,
                                asset_class,
                                expiry_params
                            )

                            if not instrument:
                                contracts_failed += 1
                                continue

                            engine.add_instrument(instrument)
                            instruments_added += 1

                            # Convert DataFrame to Nautilus Bars
                            bars = self._convert_dataframe_to_bars(contract_df, instrument)

                            if bars and len(bars) > 0:
                                engine.add_data(bars)
                                data_loaded = True
                                total_data_points += len(bars)
                                contracts_loaded += 1
                            else:
                                contracts_failed += 1

                        except Exception as e:
                            print(f"   ⚠️  Failed to process {contract_symbol}: {e}")
                            contracts_failed += 1

                    print(f"\n✓ Options contracts loaded: {contracts_loaded}")
                    if contracts_failed > 0:
                        print(f"⚠️  Failed to load: {contracts_failed} contracts")

                else:
                    # ═══════════════════════════════════════════════════
                    # FALLBACK: PRE-COLLECTION FAILED - USE OLD METHOD
                    # ═══════════════════════════════════════════════════
                    print("⚠️  No pre-collected data available")
                    print("📡 Falling back to on-demand data loading...\n")

                    # Get expiries
                    expiry_frequency = options_params.get('expiry_frequency', 'weekly')
                    expiries = get_expiries_by_frequency(start_date, end_date, expiry_frequency)
                    
                    if len(expiries) == 0:
                        raise Exception(f"No {expiry_frequency} expiries found in period")

                    print(f"✓ Found {len(expiries)} {expiry_frequency} expiries")

                    # Get option parameters
                    trade_directions = options_params.get('trade_direction', ['BUY'])
                    option_types = options_params.get('option_types', ['CALL'])

                    # Handle backward compatibility
                    if 'option_type' in options_params:
                        opt_type = options_params['option_type'].upper()
                        if opt_type == 'AUTO':
                            option_types = ['CALL', 'PUT']
                        else:
                            option_types = [opt_type.replace('C', 'CALL').replace('P', 'PUT')]

                    for symbol in symbols:
                        print(f"\n📈 Processing underlying: {symbol}")

                        # Load underlying instrument
                        underlying_instrument = self._get_instrument(symbol, venue_name, 'stocks')
                        if underlying_instrument:
                            engine.add_instrument(underlying_instrument)
                            instruments_added += 1
                            print(f"   ✓ Underlying instrument added")

                            underlying_data = self._load_data_for_symbol(
                                symbol, underlying_instrument, 'stocks',
                                start_date, end_date, None, force_minute_bars=True
                            )

                            if underlying_data and len(underlying_data) > 0:
                                engine.add_data(underlying_data)
                                data_loaded = True
                                total_data_points += len(underlying_data)
                                print(f"   ✓ Loaded {len(underlying_data):,} underlying bars")

                        # Get strike parameters
                        base_strike_price = options_params.get('strike_price')
                        if base_strike_price == 'auto':
                            base_strike_price = None

                        strike_selection = options_params.get('strike_selection', 'ATM')
                        strike_offset = options_params.get('strike_offset', 0)

                        # Load options contracts
                        for expiry_date in expiries:
                            strike_price = base_strike_price

                            # Calculate ATM strike if needed
                            if strike_price is None:
                                try:
                                    import requests
                                    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{expiry_date}/{expiry_date}"
                                    response = requests.get(url, params={'apiKey': polygon_key}, timeout=10)
                                    data = response.json()

                                    if data.get('status') == 'OK' and data.get('results'):
                                        is_0dte = expiry_frequency == 'daily'
                                        spot_price = float(data['results'][0]['o' if is_0dte else 'c'])

                                        strike_price = calculate_strike_price(
                                            spot_price=spot_price,
                                            strike_selection=strike_selection,
                                            strike_offset=strike_offset,
                                            strike_interval=1.0 if spot_price < 100 else 5.0
                                        )
                                    else:
                                        strike_price = 100.0
                                except Exception:
                                    strike_price = 100.0

                            for option_type in option_types:
                                for trade_direction in trade_directions:
                                    option_type_code = 'C' if option_type == 'CALL' else 'P'

                                    expiry_options_params = {
                                        'expiry_date': expiry_date,
                                        'strike_price': strike_price,
                                        'strike_selection': strike_selection,
                                        'strike_offset': strike_offset,
                                        'option_type': option_type_code,
                                        'trade_direction': trade_direction
                                    }

                                    instrument = self._get_instrument(
                                        symbol, venue_name, asset_class, expiry_options_params
                                    )

                                    if not instrument:
                                        continue

                                    engine.add_instrument(instrument)
                                    instruments_added += 1

                                    data = self._load_data_for_symbol(
                                        symbol, instrument, asset_class,
                                        start_date, end_date, expiry_options_params
                                    )

                                    if data and len(data) > 0:
                                        engine.add_data(data)
                                        data_loaded = True
                                        total_data_points += len(data)

            else:
                # ═══════════════════════════════════════════════════════
                # NON-OPTIONS: STANDARD DATA LOADING
                # ═══════════════════════════════════════════════════════
                print("📊 Loading market data...\n")

                for symbol in symbols:
                    print(f"Processing symbol: {symbol}")

                    # Get instrument
                    instrument = self._get_instrument(symbol, venue_name, asset_class)
                    if not instrument:
                        print(f"   ⚠️  Could not create instrument, skipping...\n")
                        continue

                    engine.add_instrument(instrument)
                    instruments_added += 1
                    print(f"   ✓ Instrument added: {instrument.id}")

                    # Load data
                    data = self._load_data_for_symbol(
                        symbol, instrument, asset_class,
                        start_date, end_date, options_params
                    )

                    if data and len(data) > 0:
                        engine.add_data(data)
                        data_loaded = True
                        total_data_points += len(data)
                        print(f"   ✓ Data loaded: {len(data):,} data points\n")
                    else:
                        print(f"   ⚠️  No data available\n")

            # Validation
            if not data_loaded:
                raise Exception(
                    f"No historical data available for symbols: {', '.join(symbols)}. "
                    f"Check data provider configuration and API keys."
                )

            print(f"📊 Data Loading Summary:")
            print(f"   Instruments added: {instruments_added}")
            print(f"   Total data points: {total_data_points:,}")
            print(f"✅ Phase 4 Complete ({format_duration(time.time() - phase_start)})\n")

            # ═══════════════════════════════════════════════════════════
            # PHASE 5: STRATEGY INITIALIZATION
            # ═══════════════════════════════════════════════════════════
            print("=" * 100)
            print("📋 PHASE 5: STRATEGY INITIALIZATION")
            print("=" * 100)

            phase_start = time.time()

            # Determine first instrument for bar subscription
            if asset_class == 'options' and options_params:
                underlying_instrument = None
                for inst in engine.cache.instruments():
                    if 'O:' not in str(inst.id) and any(sym in str(inst.id) for sym in symbols):
                        underlying_instrument = inst
                        break

                first_instrument = underlying_instrument if underlying_instrument else engine.cache.instruments()[0]
                print(f"✓ Using underlying for bars: {first_instrument.id}")
            else:
                first_instrument = engine.cache.instruments()[0]
                print(f"✓ Using instrument: {first_instrument.id}")

            # Create bar type
            bar_type = BarType.from_str(f"{first_instrument.id}-1-MINUTE-LAST-EXTERNAL")

            # Configure strategy
            if strategy_name == 'coin_daily_0dte_momentum' and COINDaily0DTEMomentum:
                print("\n🎯 Configuring COIN Daily 0DTE Momentum Strategy V2...")

                risk_mgmt = config_schema.get('risk_management', {})
                indicators = config_schema.get('indicators', [])

                # Extract EMA periods
                fast_ema_period = 9
                slow_ema_period = 20
                for ind in indicators:
                    if ind['type'] == 'ema':
                        period = ind['params'].get('period', 9)
                        if period < 15:
                            fast_ema_period = period
                        else:
                            slow_ema_period = period

                strategy_config = COINDaily0DTEMomentumConfig(
                    instrument_id=first_instrument.id,
                    bar_type=bar_type,
                    fixed_position_value=2000.0,
                    target_profit_pct=Decimal(str(risk_mgmt.get('take_profit_value', 7.5))),
                    stop_loss_pct=Decimal(str(risk_mgmt.get('stop_loss_value', 25.0))),
                    min_hold_minutes=config_schema.get('min_hold_minutes', 5),
                    entry_time_start=config_schema.get('entry_time', '09:30:00'),
                    entry_time_end="10:00:00",
                    max_hold_minutes=config_schema.get('max_hold_time_minutes', 30),
                    force_exit_time="15:55:00",
                    fast_ema_period=fast_ema_period,
                    slow_ema_period=slow_ema_period,
                    max_bid_ask_spread_pct=30.0,
                    min_option_premium=2.0,
                    max_option_premium=30.0,
                    request_bars=True,
                )

                strategy = COINDaily0DTEMomentum(config=strategy_config)

                print(f"\n✅ Strategy V2 Configured:")
                print(f"   • Position Size: FIXED $2,000")
                print(f"   • Target Profit: +{risk_mgmt.get('take_profit_value', 7.5)}%")
                print(f"   • Stop Loss: -{risk_mgmt.get('stop_loss_value', 25.0)}%")
                print(f"   • Entry Window: 09:30-10:00 AM EST")
                print(f"   • Force Exit: 3:55 PM EST")
                print(f"   • Max Hold: {config_schema.get('max_hold_time_minutes', 30)} minutes")

            else:
                print("\nℹ️  Using default EMACross strategy...")
                strategy_config = EMACrossConfig(
                    instrument_id=first_instrument.id,
                    bar_type=bar_type,
                    trade_size=Decimal(1.0),
                    fast_ema_period=10,
                    slow_ema_period=20,
                    close_positions_on_stop=True,
                )
                strategy = EMACross(config=strategy_config)
                print(f"✓ Strategy configured: EMACross (10/20)")

            engine.add_strategy(strategy=strategy)
            print(f"\n✅ Strategy added to engine")
            print(f"✅ Phase 5 Complete ({format_duration(time.time() - phase_start)})\n")

            # ═══════════════════════════════════════════════════════════
            # PHASE 6: BACKTEST EXECUTION
            # ═══════════════════════════════════════════════════════════
            print("=" * 100)
            print("📋 PHASE 6: BACKTEST EXECUTION")
            print("=" * 100)

            phase_start = time.time()

            print("⚡ Running backtest...")
            print("   ℹ️  This may take several minutes...\n")

            try:
                engine.run()
                execution_time = time.time() - phase_start
                print(f"\n✅ Backtest execution completed ({format_duration(execution_time)})")
            except Exception as e:
                print(f"\n❌ Backtest execution failed: {e}")
                import traceback
                traceback.print_exc()
                raise

            print(f"✅ Phase 6 Complete ({format_duration(time.time() - phase_start)})\n")

            # ═══════════════════════════════════════════════════════════
            # PHASE 7: RESULTS EXTRACTION
            # ═══════════════════════════════════════════════════════════
            print("=" * 100)
            print("📋 PHASE 7: RESULTS EXTRACTION")
            print("=" * 100)

            phase_start = time.time()

            print("📊 Extracting backtest results...")
            results = self._extract_results(engine, venue, initial_capital)

            print(f"✅ Phase 7 Complete ({format_duration(time.time() - phase_start)})\n")

            # ═══════════════════════════════════════════════════════════
            # PHASE 8: DATABASE PERSISTENCE
            # ═══════════════════════════════════════════════════════════
            print("=" * 100)
            print("📋 PHASE 8: DATABASE PERSISTENCE")
            print("=" * 100)

            phase_start = time.time()

            print("💾 Saving results to database...")
            try:
                await self.db_service.execute(
                    """UPDATE strategy_backtests
                       SET status = 'completed',
                           results = $1,
                           final_equity = $2,
                           total_pnl = $3,
                           total_pnl_pct = $4,
                           sharpe_ratio = $5,
                           max_drawdown = $6,
                           win_rate = $7,
                           total_trades = $8,
                           winning_trades = $9,
                           losing_trades = $10,
                           trades = $11,
                           metrics = $12,
                           updated_at = NOW(),
                           completed_at = NOW()
                       WHERE id = $13""",
                    json.dumps(clean_for_json(results)),
                    results.get('final_equity', 0),
                    results.get('total_pnl', 0),
                    results.get('total_return_pct', 0),
                    results.get('sharpe_ratio', 0),
                    results.get('max_drawdown_pct', 0),
                    results.get('win_rate', 0),
                    results.get('total_trades', 0),
                    results.get('winning_trades', 0),
                    results.get('losing_trades', 0),
                    json.dumps(clean_for_json(results.get('trades', []))),
                    json.dumps(clean_for_json({
                        'positions': results.get('positions', []),
                        'account_summary': results.get('account_summary', {}),
                        'fills_count': results.get('fills_count', 0),
                        'positions_count': results.get('positions_count', 0)
                    })),
                    backtest_id
                )
                print(f"✓ Results saved to database")
            except Exception as e:
                print(f"⚠️  Warning: Could not save to database: {e}")

            print(f"✅ Phase 8 Complete ({format_duration(time.time() - phase_start)})\n")

            # ═══════════════════════════════════════════════════════════
            # PHASE 9: CLEANUP & FINAL SUMMARY
            # ═══════════════════════════════════════════════════════════
            print("=" * 100)
            print("📋 PHASE 9: CLEANUP & FINAL SUMMARY")
            print("=" * 100)

            phase_start = time.time()

            print("🧹 Cleaning up backtest engine...")
            try:
                engine.reset()
                engine.dispose()
                print("✓ Engine cleaned up")
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}")

            total_time = time.time() - backtest_start_time

            # Print final summary
            print("\n" + "=" * 100)
            print("✅ BACKTEST COMPLETED SUCCESSFULLY")
            print("=" * 100)
            print(f"Backtest ID:      {backtest_id}")
            print(f"Execution Time:   {format_duration(total_time)}")
            print(f"\nPERFORMANCE RESULTS:")
            print(f"Initial Capital:  ${initial_capital:,.2f}")
            print(f"Final Equity:     ${results.get('final_equity', 0):,.2f}")
            print(f"Total P&L:        ${results.get('total_pnl', 0):,.2f}")
            print(f"Total Return:     {results.get('total_return_pct', 0):+.2f}%")
            print(f"\nRISK METRICS:")
            print(f"Sharpe Ratio:     {results.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown:     {results.get('max_drawdown_pct', 0):.2f}%")
            print(f"\nTRADING ACTIVITY:")
            print(f"Total Trades:     {results.get('total_trades', 0)}")
            print(f"Winning Trades:   {results.get('winning_trades', 0)}")
            print(f"Losing Trades:    {results.get('losing_trades', 0)}")
            print(f"Win Rate:         {results.get('win_rate', 0):.2f}%")
            print("=" * 100 + "\n")

            print(f"✅ Phase 9 Complete ({format_duration(time.time() - phase_start)})")

            return {
                "success": True,
                "backtest_id": backtest_id,
                "execution_time": total_time,
                "results": results
            }

        except Exception as e:
            total_time = time.time() - backtest_start_time

            print("\n" + "=" * 100)
            print("❌ BACKTEST FAILED")
            print("=" * 100)
            print(f"Backtest ID:      {backtest_id}")
            print(f"Execution Time:   {format_duration(total_time)}")
            print(f"Error:            {str(e)}")
            print("=" * 100 + "\n")

            print("📋 FULL ERROR TRACEBACK:")
            print("-" * 100)
            import traceback
            traceback.print_exc()
            print("-" * 100 + "\n")

            # Update database status to failed
            try:
                await self.db_service.execute(
                    """UPDATE strategy_backtests
                       SET status = 'failed',
                           error_message = $1,
                           updated_at = NOW()
                       WHERE id = $2""",
                    str(e),
                    backtest_id
                )
                print("💾 Database updated with failure status")
            except Exception as db_error:
                print(f"⚠️  Could not update database: {db_error}")

            return {
                "success": False,
                "backtest_id": backtest_id,
                "execution_time": total_time,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def _convert_dataframe_to_bars(self, df: pd.DataFrame, instrument) -> List[Bar]:
        """
        Convert DataFrame from DataCollectionPipeline to Nautilus Bar objects.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns [timestamp, open, high, low, close, volume].
        instrument : Instrument
            Nautilus instrument object.
        
        Returns
        -------
        List[Bar]
            List of Nautilus Bar objects.
        """
        bars = []
        
        try:
            for idx, row in df.iterrows():
                # Convert timestamp to nanoseconds
                ts_event = int(row['timestamp'].timestamp() * 1_000_000_000)
                
                bar = Bar(
                    bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"),
                    open=Price.from_str(f"{row['open']:.2f}"),
                    high=Price.from_str(f"{row['high']:.2f}"),
                    low=Price.from_str(f"{row['low']:.2f}"),
                    close=Price.from_str(f"{row['close']:.2f}"),
                    volume=Quantity.from_int(int(row['volume'])),
                    ts_event=ts_event,
                    ts_init=ts_event,
                )
                bars.append(bar)
            
            print(f"      ✓ Converted {len(bars):,} bars for {instrument.id}")
            
        except Exception as e:
            print(f"      ❌ Failed to convert bars: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        return bars

    def _get_venue_for_symbol(self, symbol: str, asset_class: str) -> str:
        """Determine appropriate venue for symbol"""
        if asset_class == 'crypto' or '/' in symbol or 'USDT' in symbol:
            return 'BINANCE'
        elif asset_class == 'stocks':
            return 'NYSE'
        elif asset_class == 'options':
            return 'CBOE'
        elif asset_class == 'forex':
            return 'SIM'
        return 'SIM'

    def _get_instrument(self, symbol: str, venue: str, asset_class: str, options_params: Optional[Dict[str, Any]] = None):
        """Get or create instrument for symbol"""
        try:
            # Handle options contracts
            if asset_class == 'options' and options_params:
                from nautilus_trader.model.instruments import OptionContract
                from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue as VenueId
                from nautilus_trader.model.enums import AssetClass, OptionKind
                from nautilus_trader.model.objects import Currency, Price, Quantity, Money
                from datetime import datetime
                from decimal import Decimal
                import pandas as pd
                import pytz
                from nautilus_trader.model.currencies import USD

                # Extract options parameters
                expiry_date = options_params.get('expiry_date')
                strike_price = options_params.get('strike_price')
                option_type = options_params.get('option_type')  # 'C' or 'P'

                # Construct OCC symbol
                expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
                expiry_str = expiry_dt.strftime('%y%m%d')
                strike_int = int(strike_price * 1000)
                occ_symbol = f"{symbol}{expiry_str}{option_type}{strike_int:08d}"

                # Timestamps
                activation_ts = pd.Timestamp("2024-01-01", tz=pytz.utc)
                
                expiry_datetime = datetime.strptime(expiry_date, '%Y-%m-%d')
                expiry_ts = pd.Timestamp(
                    year=expiry_datetime.year,
                    month=expiry_datetime.month,
                    day=expiry_datetime.day,
                    hour=16,
                    minute=0,
                    tz='America/New_York'
                ).tz_convert(pytz.utc)

                now_ts = pd.Timestamp.now(tz=pytz.utc)

                option = OptionContract(
                    InstrumentId.from_str(f"{occ_symbol}.{venue}"),
                    Symbol(occ_symbol),
                    AssetClass.EQUITY,
                    USD,
                    2,
                    Price.from_str("0.01"),
                    Quantity.from_int(100),
                    Quantity.from_int(1),
                    symbol,
                    OptionKind.CALL if option_type == 'C' else OptionKind.PUT,
                    Price.from_str(f"{strike_price:.2f}"),
                    activation_ts.value,
                    expiry_ts.value,
                    now_ts.value,
                    now_ts.value,
                    margin_init=None,
                    margin_maint=None,
                    maker_fee=None,
                    taker_fee=None,
                    exchange=venue,
                    info=None,
                )

                return option

            # Handle equity instruments
            elif asset_class == 'stocks' or asset_class == 'equity':
                return TestInstrumentProvider.equity(symbol=symbol, venue=venue)

            # Handle crypto instruments
            elif asset_class == 'crypto':
                if symbol in ['ETHUSDT', 'ETH/USDT', 'ETH/USD']:
                    return TestInstrumentProvider.ethusdt_binance()
                elif symbol in ['BTCUSDT', 'BTC/USDT', 'BTC/USD']:
                    return TestInstrumentProvider.btcusdt_binance()
                else:
                    return TestInstrumentProvider.equity(symbol=symbol, venue=venue)

            # Handle forex instruments
            elif asset_class == 'forex':
                from nautilus_trader.model.identifiers import Venue as VenueId
                if symbol == 'AUD/USD':
                    return TestInstrumentProvider.default_fx_ccy("AUD/USD", VenueId(venue))
                elif symbol == 'USD/JPY':
                    return TestInstrumentProvider.default_fx_ccy("USD/JPY", VenueId(venue))
                else:
                    return TestInstrumentProvider.default_fx_ccy(symbol, VenueId(venue))

            # Default fallback
            else:
                return TestInstrumentProvider.equity(symbol=symbol, venue=venue)

        except Exception as e:
            print(f"      ❌ Error creating instrument: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_data_for_symbol(
        self,
        symbol: str,
        instrument,
        asset_class: str,
        start_date: str,
        end_date: str,
        options_params: Optional[Dict[str, Any]] = None,
        force_minute_bars: bool = False
    ):
        """Load historical data for symbol (fallback when pre-collection unavailable)"""
        try:
            # Calculate optimal timeframe
            from datetime import datetime
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            days = (end - start).days

            # Smart timeframe selection
            if force_minute_bars or (asset_class == 'options' and options_params):
                timeframe = '1Min'
            elif days > 60:
                timeframe = '1Day'
            elif days > 14:
                timeframe = '1Hour'
            else:
                timeframe = '1Min'

            # Try to fetch from providers
            bars = fetch_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                asset_class=asset_class,
                timeframe=timeframe,
                instrument=instrument,
                options_params=options_params
            )

            if bars and len(bars) > 0:
                return bars

            # Fall back to sample data
            data_file = SAMPLE_DATA_MAP.get(symbol)
            if not data_file:
                normalized = symbol.replace('/', '').upper()
                data_file = SAMPLE_DATA_MAP.get(normalized)

            if not data_file:
                return None

            data_path = self.test_data_dir / data_file
            if not data_path.exists():
                return None

            provider = TestDataProvider()

            if 'trades' in str(data_file):
                wrangler = TradeTickDataWrangler(instrument=instrument)
                return wrangler.process(provider.read_csv_ticks(data_file))
            else:
                wrangler = QuoteTickDataWrangler(instrument=instrument)
                return wrangler.process(provider.read_csv_ticks(data_file))

        except Exception as e:
            print(f"      ⚠️  Error loading data: {e}")
            return None

    def _extract_results(self, engine: BacktestEngine, venue: Venue, initial_capital: float) -> Dict[str, Any]:
        """Extract backtest results from engine"""
        try:
            # Get reports
            account_df = engine.trader.generate_account_report(venue)
            fills_df = engine.trader.generate_order_fills_report()
            positions_df = engine.trader.generate_positions_report()

            # Filter fills for V2 manual order tracking
            if len(fills_df) > 0:
                original_count = len(fills_df)
                fills_df = fills_df[
                    (fills_df['last_qty'].notna()) & 
                    (fills_df['last_qty'] > 0) &
                    (fills_df['last_px'].notna()) &
                    (fills_df['last_px'] > 0)
                ]
                filtered_count = len(fills_df)
                print(f"   Filtered: {filtered_count} fills (removed {original_count - filtered_count} cancelled)")

            # Calculate metrics
            final_equity = float(account_df['total'].iloc[-1]) if len(account_df) > 0 else initial_capital
            total_pnl = final_equity - initial_capital
            total_return_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0

            total_trades = len(positions_df)
            total_fills = len(fills_df)

            # Calculate wins/losses
            winning_trades = 0
            losing_trades = 0
            if len(positions_df) > 0 and 'realized_pnl' in positions_df.columns:
                positions_df['realized_pnl'] = pd.to_numeric(positions_df['realized_pnl'], errors='coerce')
                winning_trades = len(positions_df[positions_df['realized_pnl'] > 0])
                losing_trades = len(positions_df[positions_df['realized_pnl'] < 0])

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            # Calculate Sharpe & Drawdown
            sharpe_ratio = 0.0
            max_drawdown_pct = 0.0

            if len(account_df) > 1:
                equity_series = account_df['total'].astype(float)
                returns = equity_series.pct_change().dropna()

                if len(returns) > 0:
                    mean_return = returns.mean()
                    std_return = returns.std()

                    if std_return > 0:
                        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
                        sharpe_ratio = max(-10, min(10, sharpe_ratio))

                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.cummax()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown_pct = abs(drawdown.min() * 100)

            # Convert to JSON
            account_summary = []
            if len(account_df) > 0:
                account_df_copy = account_df.reset_index()
                if 'index' in account_df_copy.columns:
                    account_df_copy['index'] = account_df_copy['index'].astype(str)
                account_summary = clean_for_json(account_df_copy.to_dict(orient='records'))

            trades_list = []
            if len(fills_df) > 0:
                fills_copy = fills_df.reset_index()
                for col in fills_copy.columns:
                    if fills_copy[col].dtype == 'datetime64[ns]' or fills_copy[col].dtype == 'datetime64[ns, UTC]':
                        fills_copy[col] = fills_copy[col].astype(str)
                fills_copy = fills_copy.replace([np.inf, -np.inf], None)
                fills_copy = fills_copy.where(pd.notna(fills_copy), None)
                trades_list = clean_for_json(fills_copy.to_dict(orient='records'))

            positions_list = []
            if len(positions_df) > 0:
                positions_copy = positions_df.reset_index()
                for col in positions_copy.columns:
                    if positions_copy[col].dtype == 'datetime64[ns]' or positions_copy[col].dtype == 'datetime64[ns, UTC]':
                        positions_copy[col] = positions_copy[col].astype(str)
                positions_copy = positions_copy.replace([np.inf, -np.inf], None)
                positions_copy = positions_copy.where(pd.notna(positions_copy), None)
                positions_list = clean_for_json(positions_copy.to_dict(orient='records'))

            print(f"   Final Equity: ${final_equity:,.2f}")
            print(f"   Total P&L: ${total_pnl:,.2f} ({total_return_pct:+.2f}%)")
            print(f"   Total Trades: {total_trades}")
            print(f"   Win Rate: {win_rate:.2f}%")

            return {
                "final_equity": final_equity,
                "total_pnl": total_pnl,
                "total_return_pct": total_return_pct,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown_pct": max_drawdown_pct,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "account_summary": account_summary,
                "fills_count": total_fills,
                "positions_count": total_trades,
                "trades": trades_list,
                "positions": positions_list
            }

        except Exception as e:
            print(f"   ❌ Error extracting results: {e}")
            import traceback
            traceback.print_exc()

            return {
                "final_equity": initial_capital,
                "total_pnl": 0,
                "total_return_pct": 0,
                "sharpe_ratio": 0,
                "max_drawdown_pct": 0,
                "win_rate": 0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "account_summary": [],
                "fills_count": 0,
                "positions_count": 0,
                "trades": [],
                "positions": []
            }


async def execute_backtest_async(
    backtest_id: str,
    strategy_id: str,
    symbols: List[str],
    venues: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float,
    db_service,
    asset_class: str = 'stocks',
    options_params: Optional[Dict[str, Any]] = None
):
    """Execute backtest in background (async wrapper)"""
    runner = NautilusBacktestRunner(db_service)
    return await runner.run_backtest(
        backtest_id=backtest_id,
        strategy_id=strategy_id,
        symbols=symbols,
        venues=venues,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        asset_class=asset_class,
        options_params=options_params
    )