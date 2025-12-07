"""
Backtesting Engine.

The core engine that simulates trading:
- Loads and feeds historical data bar by bar
- Manages strategy execution
- Simulates order fills
- Tracks performance metrics
- Generates comprehensive results
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
import json
import math
import time

import pandas as pd
import numpy as np

from ..core.models import (
    Bar, Order, OrderSide, OrderType, OrderStatus, TimeInForce,
    Fill, Position, Trade, Account, Instrument, OptionContract,
    InstrumentType, OptionType
)
from ..core.events import BarEvent, FillEvent, PositionEvent
from .order_manager import OrderManager


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.

    Attributes:
        initial_capital: Starting account balance
        commission_per_contract: Commission per contract traded
        slippage_pct: Simulated slippage percentage
        data_path: Path to data directory
    """
    initial_capital: float = 100000.0
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.1
    data_path: str = ""


@dataclass
class BacktestResults:
    """
    Comprehensive backtest results.

    Contains all performance metrics and trade history.
    """
    # Basic metrics
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Duration
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration_days: int = 0
    execution_time_seconds: float = 0.0

    # Trade details
    trades: List[Dict] = field(default_factory=list)
    fills: List[Dict] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.final_equity,
            'total_pnl': self.total_pnl,
            'total_return_pct': self.total_return_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'avg_trade_pnl': self.avg_trade_pnl,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'duration_days': self.duration_days,
            'execution_time_seconds': self.execution_time_seconds,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
        }


class BacktestEngine:
    """
    Core backtesting engine that simulates trading.

    Usage:
    ------
    engine = BacktestEngine(config)
    engine.add_instrument(instrument)
    engine.add_data(bars)
    engine.add_strategy(strategy)
    engine.run()
    results = engine.get_results()
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtest engine.

        Parameters
        ----------
        config : BacktestConfig, optional
            Backtest configuration. Uses defaults if not provided.
        """
        self.config = config or BacktestConfig()

        # Account and order management
        self.account = Account(
            initial_balance=self.config.initial_capital,
            cash_balance=self.config.initial_capital,
        )
        self.order_manager = OrderManager(
            account=self.account,
            slippage_pct=self.config.slippage_pct,
            commission_per_contract=self.config.commission_per_contract,
        )

        # Instruments and data
        self.instruments: Dict[str, Instrument] = {}
        self.data: Dict[str, List[Bar]] = {}  # symbol -> list of bars
        self.current_bars: Dict[str, Bar] = {}  # symbol -> current bar

        # Strategy
        self.strategy = None

        # State
        self.current_time: Optional[datetime] = None
        self.bar_count: int = 0
        self.is_running: bool = False

        # Results tracking
        self.equity_curve: List[Dict] = []
        self.peak_equity: float = self.config.initial_capital

        # Register callbacks
        self.order_manager.on_fill(self._on_fill)

    # ═══════════════════════════════════════════════════════════════════════
    # SETUP METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def add_instrument(self, instrument: Instrument) -> None:
        """
        Add a tradeable instrument.

        Parameters
        ----------
        instrument : Instrument
            Instrument to add (stock, option, etc.)
        """
        self.instruments[instrument.symbol] = instrument
        self.data[instrument.symbol] = []
        print(f"   ✓ Added instrument: {instrument.symbol}")

    def add_data(self, bars: List[Bar]) -> None:
        """
        Add historical bar data for backtesting.

        Parameters
        ----------
        bars : List[Bar]
            List of OHLCV bars sorted by timestamp
        """
        if not bars:
            return

        symbol = bars[0].symbol

        # Ensure instrument exists
        if symbol not in self.instruments:
            print(f"   ⚠️ Warning: Adding data for unknown instrument {symbol}")
            # Create a basic stock instrument
            self.instruments[symbol] = Instrument(
                symbol=symbol,
                instrument_type=InstrumentType.STOCK,
            )

        self.data[symbol].extend(bars)
        self.data[symbol].sort(key=lambda b: b.timestamp)
        print(f"   ✓ Added {len(bars):,} bars for {symbol}")

    def add_data_from_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        instrument: Optional[Instrument] = None
    ) -> None:
        """
        Add data from a pandas DataFrame.

        Expected columns: timestamp, open, high, low, close, volume

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        symbol : str
            Symbol for the data
        instrument : Instrument, optional
            Instrument object. Creates default if not provided.
        """
        if instrument:
            self.add_instrument(instrument)
        elif symbol not in self.instruments:
            self.instruments[symbol] = Instrument(
                symbol=symbol,
                instrument_type=InstrumentType.STOCK,
            )
            self.data[symbol] = []

        bars = []
        for _, row in df.iterrows():
            bar = Bar(
                symbol=symbol,
                timestamp=pd.to_datetime(row['timestamp']),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row.get('volume', 0)),
            )
            bars.append(bar)

        self.add_data(bars)

    def add_strategy(self, strategy) -> None:
        """
        Add a trading strategy.

        Parameters
        ----------
        strategy : Strategy
            Strategy instance to run
        """
        self.strategy = strategy
        self.strategy._set_engine(self)
        print(f"   ✓ Added strategy: {strategy.__class__.__name__}")

    # ═══════════════════════════════════════════════════════════════════════
    # CACHE METHODS (for strategy compatibility)
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def cache(self):
        """Return self as cache (for strategy compatibility)"""
        return self

    def instrument(self, instrument_id) -> Optional[Instrument]:
        """Get instrument by ID (for strategy compatibility)"""
        # Handle both string and InstrumentId objects
        symbol = str(instrument_id).split('.')[0] if '.' in str(instrument_id) else str(instrument_id)
        return self.instruments.get(symbol)

    def instruments_list(self) -> List[Instrument]:
        """Get all instruments (for strategy compatibility)"""
        return list(self.instruments.values())

    def quote_tick(self, instrument_id) -> Optional[Any]:
        """Get current quote for instrument (for strategy compatibility)"""
        symbol = str(instrument_id).split('.')[0] if '.' in str(instrument_id) else str(instrument_id)
        bar = self.current_bars.get(symbol)
        if bar:
            # Return a simple object with bid/ask from bar close
            class SimpleQuote:
                def __init__(self, price):
                    self.bid_price = price * 0.999  # Approximate bid
                    self.ask_price = price * 1.001  # Approximate ask
            return SimpleQuote(bar.close)
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # PORTFOLIO METHODS (for strategy compatibility)
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def portfolio(self):
        """Return self as portfolio (for strategy compatibility)"""
        return self

    def is_net_long(self, instrument_id) -> bool:
        """Check if net long for instrument"""
        symbol = str(instrument_id).split('.')[0] if '.' in str(instrument_id) else str(instrument_id)
        return self.account.is_net_long(symbol)

    def is_net_short(self, instrument_id) -> bool:
        """Check if net short for instrument"""
        symbol = str(instrument_id).split('.')[0] if '.' in str(instrument_id) else str(instrument_id)
        return self.account.is_net_short(symbol)

    # ═══════════════════════════════════════════════════════════════════════
    # EXECUTION
    # ═══════════════════════════════════════════════════════════════════════

    def run(self) -> BacktestResults:
        """
        Run the backtest simulation.

        Returns
        -------
        BacktestResults
            Comprehensive results object
        """
        start_time = time.time()

        print("\n" + "=" * 80)
        print("🚀 STARTING BACKTEST")
        print("=" * 80)
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        print(f"Instruments: {len(self.instruments)}")
        print(f"Total bars: {sum(len(d) for d in self.data.values()):,}")
        print("=" * 80 + "\n")

        # Validate setup
        if not self.strategy:
            raise ValueError("No strategy added. Call add_strategy() first.")

        if not self.data:
            raise ValueError("No data added. Call add_data() first.")

        # Merge all bars and sort by timestamp
        all_bars = []
        for symbol, bars in self.data.items():
            all_bars.extend(bars)
        all_bars.sort(key=lambda b: b.timestamp)

        if not all_bars:
            raise ValueError("No bars available for backtesting")

        # Initialize
        self.is_running = True
        self.bar_count = 0

        # Call strategy start
        try:
            self.strategy.on_start()
        except Exception as e:
            print(f"❌ Error in strategy on_start(): {e}")
            import traceback
            traceback.print_exc()

        # Process bars
        print("⚡ Processing bars...")
        total_bars = len(all_bars)

        for i, bar in enumerate(all_bars):
            self.current_time = bar.timestamp
            self.current_bars[bar.symbol] = bar
            self.bar_count += 1

            # Process pending orders
            fills = self.order_manager.process_bar(bar)

            # Call strategy on_bar
            try:
                self.strategy.on_bar(bar)
            except Exception as e:
                print(f"❌ Error in strategy on_bar(): {e}")
                import traceback
                traceback.print_exc()

            # Record equity
            self._record_equity(bar.timestamp)

            # Progress logging (every 10%)
            if i > 0 and i % (total_bars // 10) == 0:
                pct = (i / total_bars) * 100
                print(f"   {pct:.0f}% complete ({i:,}/{total_bars:,} bars)")

        # Call strategy stop
        try:
            self.strategy.on_stop()
        except Exception as e:
            print(f"❌ Error in strategy on_stop(): {e}")

        self.is_running = False
        execution_time = time.time() - start_time

        # Generate results
        results = self._generate_results(all_bars, execution_time)

        # Print summary
        self._print_summary(results)

        return results

    def _record_equity(self, timestamp: datetime) -> None:
        """Record equity point for equity curve"""
        equity = self.account.equity
        self.equity_curve.append({
            'timestamp': timestamp.isoformat(),
            'equity': equity,
        })

        # Track peak for drawdown calculation
        if equity > self.peak_equity:
            self.peak_equity = equity

    def _on_fill(self, event: FillEvent) -> None:
        """Handle fill events"""
        try:
            self.strategy.on_order_filled(event)
        except Exception as e:
            print(f"❌ Error in strategy on_order_filled(): {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # RESULTS GENERATION
    # ═══════════════════════════════════════════════════════════════════════

    def _generate_results(
        self,
        all_bars: List[Bar],
        execution_time: float
    ) -> BacktestResults:
        """Generate comprehensive backtest results"""
        results = BacktestResults()

        # Basic metrics
        results.initial_capital = self.config.initial_capital
        results.final_equity = self.account.equity
        results.total_pnl = results.final_equity - results.initial_capital
        results.total_return_pct = self.account.total_return_pct

        # Dates
        results.start_date = all_bars[0].timestamp if all_bars else None
        results.end_date = all_bars[-1].timestamp if all_bars else None
        if results.start_date and results.end_date:
            results.duration_days = (results.end_date - results.start_date).days

        results.execution_time_seconds = execution_time

        # Trade statistics
        trades = self.account.trades
        results.total_trades = len(trades)

        if trades:
            winners = [t for t in trades if t.is_winner]
            losers = [t for t in trades if not t.is_winner]

            results.winning_trades = len(winners)
            results.losing_trades = len(losers)
            results.win_rate = (len(winners) / len(trades)) * 100

            if winners:
                results.avg_win = sum(t.pnl for t in winners) / len(winners)
            if losers:
                results.avg_loss = sum(t.pnl for t in losers) / len(losers)

            total_wins = sum(t.pnl for t in winners)
            total_losses = abs(sum(t.pnl for t in losers))
            results.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

            results.avg_trade_pnl = results.total_pnl / len(trades)

        # Risk metrics
        results.max_drawdown, results.max_drawdown_pct = self._calculate_max_drawdown()
        results.sharpe_ratio = self._calculate_sharpe_ratio()
        results.sortino_ratio = self._calculate_sortino_ratio()

        # Trade details
        results.trades = [t.to_dict() for t in trades]
        results.fills = [f.to_dict() for f in self.account.fills]
        results.equity_curve = self.equity_curve

        return results

    def _calculate_max_drawdown(self) -> tuple:
        """Calculate maximum drawdown"""
        if not self.equity_curve:
            return 0.0, 0.0

        equities = [e['equity'] for e in self.equity_curve]
        peak = equities[0]
        max_dd = 0.0
        max_dd_pct = 0.0

        for equity in equities:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = (dd / peak) * 100 if peak > 0 else 0

            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        return max_dd, max_dd_pct

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (annualized)"""
        if len(self.equity_curve) < 2:
            return 0.0

        equities = pd.Series([e['equity'] for e in self.equity_curve])
        returns = equities.pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns.mean() - (risk_free_rate / 252)  # Daily risk-free rate
        sharpe = (excess_returns / returns.std()) * np.sqrt(252)

        # Clamp to reasonable range
        return max(-10, min(10, sharpe))

    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (annualized)"""
        if len(self.equity_curve) < 2:
            return 0.0

        equities = pd.Series([e['equity'] for e in self.equity_curve])
        returns = equities.pct_change().dropna()

        if len(returns) == 0:
            return 0.0

        # Only consider downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if returns.mean() > 0 else 0.0

        excess_returns = returns.mean() - (risk_free_rate / 252)
        sortino = (excess_returns / downside_returns.std()) * np.sqrt(252)

        return max(-10, min(10, sortino))

    def _print_summary(self, results: BacktestResults) -> None:
        """Print backtest summary"""
        print("\n" + "=" * 80)
        print("✅ BACKTEST COMPLETED")
        print("=" * 80)
        print(f"Execution Time:   {results.execution_time_seconds:.1f}s")
        print(f"Period:           {results.start_date} to {results.end_date}")
        print(f"Duration:         {results.duration_days} days")
        print()
        print("PERFORMANCE:")
        print(f"  Initial Capital: ${results.initial_capital:,.2f}")
        print(f"  Final Equity:    ${results.final_equity:,.2f}")
        print(f"  Total P&L:       ${results.total_pnl:,.2f} ({results.total_return_pct:+.2f}%)")
        print()
        print("TRADES:")
        print(f"  Total Trades:    {results.total_trades}")
        print(f"  Winning Trades:  {results.winning_trades}")
        print(f"  Losing Trades:   {results.losing_trades}")
        print(f"  Win Rate:        {results.win_rate:.1f}%")
        if results.avg_win > 0:
            print(f"  Avg Win:         ${results.avg_win:,.2f}")
        if results.avg_loss < 0:
            print(f"  Avg Loss:        ${results.avg_loss:,.2f}")
        print()
        print("RISK METRICS:")
        print(f"  Max Drawdown:    ${results.max_drawdown:,.2f} ({results.max_drawdown_pct:.2f}%)")
        print(f"  Sharpe Ratio:    {results.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:   {results.sortino_ratio:.2f}")
        if results.profit_factor != float('inf'):
            print(f"  Profit Factor:   {results.profit_factor:.2f}")
        print("=" * 80 + "\n")

    # ═══════════════════════════════════════════════════════════════════════
    # CLEANUP
    # ═══════════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """Reset engine for a new backtest"""
        self.account = Account(
            initial_balance=self.config.initial_capital,
            cash_balance=self.config.initial_capital,
        )
        self.order_manager = OrderManager(
            account=self.account,
            slippage_pct=self.config.slippage_pct,
            commission_per_contract=self.config.commission_per_contract,
        )
        self.order_manager.on_fill(self._on_fill)

        self.current_bars.clear()
        self.current_time = None
        self.bar_count = 0
        self.is_running = False
        self.equity_curve.clear()
        self.peak_equity = self.config.initial_capital

    def dispose(self) -> None:
        """Clean up resources"""
        self.instruments.clear()
        self.data.clear()
        self.strategy = None
        self.reset()
