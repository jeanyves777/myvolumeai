"""
Test script for the trading system.

Run this to verify all components work correctly.
"""

import sys
import io
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("THEVOLUMEAI TRADING SYSTEM - COMPONENT TESTS")
print("=" * 80)

# Test 1: Core Models
print("\n[TEST 1] Core Models")
print("-" * 40)
try:
    from trading_system.core.models import (
        Bar, Order, OrderSide, OrderType, OrderStatus, TimeInForce,
        Position, Trade, Fill, Account, Instrument, OptionContract,
        InstrumentType, OptionType
    )

    # Create an instrument
    inst = Instrument(
        symbol="AAPL",
        instrument_type=InstrumentType.STOCK,
        currency="USD",
        exchange="NASDAQ",
    )
    print(f"   [OK] Created instrument: {inst.symbol}")

    # Create a bar
    bar = Bar(
        symbol="AAPL",
        timestamp=datetime.now(),
        open=150.0,
        high=152.0,
        low=149.0,
        close=151.0,
        volume=1000000,
    )
    print(f"   [OK] Created bar: O={bar.open} H={bar.high} L={bar.low} C={bar.close}")

    # Create an order
    order = Order(
        instrument=inst,
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
    )
    print(f"   [OK] Created order: {order.side.value} {order.quantity} @ MARKET")

    # Create an account
    account = Account(initial_balance=100000.0)
    print(f"   [OK] Created account: ${account.initial_balance:,.2f}")

    # Create an option contract
    option = OptionContract(
        symbol="AAPL241115C00150000",
        underlying_symbol="AAPL",
        option_type=OptionType.CALL,
        strike_price=150.0,
        expiration=datetime.now() + timedelta(days=7),
    )
    print(f"   [OK] Created option: {option.symbol} (CALL ${option.strike_price})")

    print("   >>> Core models: PASSED")
except Exception as e:
    print(f"   >>> Core models: FAILED - {e}")
    import traceback
    traceback.print_exc()

# Test 2: Indicators
print("\n[TEST 2] Indicators")
print("-" * 40)
try:
    from trading_system.indicators import (
        ExponentialMovingAverage,
        SimpleMovingAverage,
        RelativeStrengthIndex,
        MACD,
        BollingerBands,
        AverageTrueRange,
    )

    # Test EMA
    ema = ExponentialMovingAverage(period=10)
    for i in range(20):
        ema.update(100 + i)
    print(f"   [OK] EMA(10): {ema.value:.2f} (initialized={ema.initialized})")

    # Test RSI
    rsi = RelativeStrengthIndex(period=14)
    for i in range(30):
        rsi.update(100 + (i % 5) - 2)
    print(f"   [OK] RSI(14): {rsi.value:.2f} (initialized={rsi.initialized})")

    # Test MACD
    macd = MACD(12, 26, 9)
    for i in range(50):
        macd.update(100 + i * 0.5)
    print(f"   [OK] MACD: value={macd.value:.2f}, signal={macd.signal:.2f}")

    # Test Bollinger Bands
    bb = BollingerBands(period=20, k=2.0)
    for i in range(30):
        bb.update(100 + (i % 10) - 5)
    print(f"   [OK] BB: upper={bb.upper:.2f}, middle={bb.middle:.2f}, lower={bb.lower:.2f}")

    # Test ATR
    atr = AverageTrueRange(period=14)
    for i in range(20):
        bar = Bar(
            symbol="TEST",
            timestamp=datetime.now(),
            open=100.0,
            high=102.0 + i * 0.1,
            low=98.0 - i * 0.1,
            close=100.0 + i * 0.05,
            volume=1000,
        )
        atr.update_from_bar(bar)
    print(f"   [OK] ATR(14): {atr.value:.2f}")

    print("   >>> Indicators: PASSED")
except Exception as e:
    print(f"   >>> Indicators: FAILED - {e}")
    import traceback
    traceback.print_exc()

# Test 3: Order Manager
print("\n[TEST 3] Order Manager")
print("-" * 40)
try:
    from trading_system.engine.order_manager import OrderManager
    from trading_system.core.models import Account, Instrument, InstrumentType, OrderSide

    # Create account and order manager
    account = Account(initial_balance=100000.0)
    om = OrderManager(account)

    # Create instrument
    inst = Instrument(symbol="TEST", instrument_type=InstrumentType.STOCK)

    # Create and submit market order
    order = om.create_market_order(inst, OrderSide.BUY, 100)
    om.submit_order(order)
    print(f"   [OK] Submitted market order: {order.client_order_id}")

    # Create and submit limit order
    limit_order = om.create_limit_order(inst, OrderSide.SELL, 100, 105.0)
    om.submit_order(limit_order)
    print(f"   [OK] Submitted limit order @ $105.00")

    # Create and submit stop order
    stop_order = om.create_stop_market_order(inst, OrderSide.SELL, 100, 95.0)
    om.submit_order(stop_order)
    print(f"   [OK] Submitted stop order @ $95.00")

    # Check pending orders
    pending = om.get_pending_orders()
    print(f"   [OK] Pending orders: {len(pending)}")

    # Process a bar (should fill market order)
    bar = Bar(symbol="TEST", timestamp=datetime.now(),
              open=100.0, high=101.0, low=99.0, close=100.5, volume=1000)
    fills = om.process_bar(bar)
    print(f"   [OK] Processed bar: {len(fills)} fills")

    print("   >>> Order Manager: PASSED")
except Exception as e:
    print(f"   >>> Order Manager: FAILED - {e}")
    import traceback
    traceback.print_exc()

# Test 4: Backtest Engine
print("\n[TEST 4] Backtest Engine")
print("-" * 40)
try:
    from trading_system.engine.backtest_engine import BacktestEngine, BacktestConfig
    from trading_system.strategy.base import Strategy, StrategyConfig
    from trading_system.core.models import Bar, Instrument, InstrumentType, OrderSide

    # Create a simple test strategy
    class SimpleTestStrategy(Strategy):
        def __init__(self):
            super().__init__(StrategyConfig())
            self.bar_count = 0

        def on_bar(self, bar):
            self.bar_count += 1

            # Buy on first bar
            if self.bar_count == 1:
                order = self.order_factory.market(
                    instrument_id=bar.symbol,
                    order_side=OrderSide.BUY,
                    quantity=10,
                )
                self.submit_order(order)

            # Sell on 10th bar
            if self.bar_count == 10:
                order = self.order_factory.market(
                    instrument_id=bar.symbol,
                    order_side=OrderSide.SELL,
                    quantity=10,
                )
                self.submit_order(order)

    # Create engine
    config = BacktestConfig(initial_capital=10000.0)
    engine = BacktestEngine(config)

    # Add instrument
    inst = Instrument(symbol="TEST", instrument_type=InstrumentType.STOCK)
    engine.add_instrument(inst)

    # Add synthetic data
    bars = []
    base_time = datetime.now()
    for i in range(20):
        bar = Bar(
            symbol="TEST",
            timestamp=base_time + timedelta(minutes=i),
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=1000,
        )
        bars.append(bar)
    engine.add_data(bars)

    # Add strategy
    strategy = SimpleTestStrategy()
    engine.add_strategy(strategy)

    # Run backtest
    results = engine.run()

    print(f"   [OK] Final equity: ${results.final_equity:,.2f}")
    print(f"   [OK] Total P&L: ${results.total_pnl:,.2f}")
    print(f"   [OK] Total trades: {results.total_trades}")
    print(f"   [OK] Win rate: {results.win_rate:.1f}%")

    print("   >>> Backtest Engine: PASSED")
except Exception as e:
    print(f"   >>> Backtest Engine: FAILED - {e}")
    import traceback
    traceback.print_exc()

# Test 5: COIN Strategy
print("\n[TEST 5] COIN 0DTE Strategy")
print("-" * 40)
try:
    from trading_system.strategies.coin_0dte_momentum import (
        COINDaily0DTEMomentum, COINDaily0DTEMomentumConfig
    )

    # Create strategy config
    config = COINDaily0DTEMomentumConfig(
        instrument_id="COIN",
        fixed_position_value=2000.0,
        target_profit_pct=Decimal("7.5"),
        stop_loss_pct=Decimal("25.0"),
    )

    # Create strategy
    strategy = COINDaily0DTEMomentum(config)

    print(f"   [OK] Created COIN strategy")
    print(f"   [OK] Position size: ${config.fixed_position_value:,.2f}")
    print(f"   [OK] Target profit: {config.target_profit_pct}%")
    print(f"   [OK] Stop loss: {config.stop_loss_pct}%")

    print("   >>> COIN Strategy: PASSED")
except Exception as e:
    print(f"   >>> COIN Strategy: FAILED - {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("All core components have been tested.")
print("\nTo run a full backtest:")
print("  py -m trading_system.run_backtest --symbol COIN --start 2024-11-01 --end 2024-11-15")
print("=" * 80)
