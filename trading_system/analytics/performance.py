"""
Performance Analytics and Reporting Module.

Provides comprehensive analysis of backtest results including:
- Trade-by-trade analysis
- Daily/Weekly/Monthly performance
- Risk metrics (Sharpe, Sortino, Max Drawdown, etc.)
- Win/Loss streaks
- Strategy execution details
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import math

import pandas as pd
import numpy as np


@dataclass
class TradeAnalysis:
    """Analysis of a single trade"""
    trade_id: str
    symbol: str
    direction: str  # CALL or PUT
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    duration_minutes: float
    exit_reason: str  # TP, SL, TIME, FORCE_EXIT
    signal: str = ""  # BULLISH, BEARISH, or NEUTRAL
    underlying_price_at_entry: float = 0.0
    strike_price: float = 0.0


@dataclass
class DailyPerformance:
    """Daily performance summary"""
    date: datetime
    trades: int
    wins: int
    losses: int
    pnl: float
    pnl_pct: float
    direction: str  # CALL, PUT, or NONE
    exit_reason: str
    starting_equity: float
    ending_equity: float


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    # Basic Info
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    trading_days: int

    # Capital
    initial_capital: float
    final_equity: float
    total_pnl: float
    total_return_pct: float

    # Trade Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L Analysis
    gross_profit: float
    gross_loss: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade_pnl: float
    largest_win: float
    largest_loss: float
    avg_win_pct: float
    avg_loss_pct: float

    # Risk Metrics
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_date: Optional[datetime]
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Streaks
    max_win_streak: int
    max_loss_streak: int
    current_streak: int
    current_streak_type: str  # WIN or LOSS

    # Time Analysis
    avg_hold_time_minutes: float
    avg_winning_hold_time: float
    avg_losing_hold_time: float

    # Direction Analysis
    call_trades: int
    put_trades: int
    call_win_rate: float
    put_win_rate: float
    call_pnl: float
    put_pnl: float

    # Exit Analysis
    tp_exits: int
    sl_exits: int
    time_exits: int
    force_exits: int

    # Daily Performance
    best_day_pnl: float
    worst_day_pnl: float
    avg_daily_pnl: float
    profitable_days: int
    losing_days: int

    # ML Ensemble Metrics (Optional)
    ml_enabled: bool = False
    ml_signal_trades: int = 0
    ml_signal_win_rate: float = 0.0
    ml_avg_confidence: float = 0.0
    ml_high_conf_trades: int = 0
    ml_high_conf_win_rate: float = 0.0
    ml_contribution_pnl: float = 0.0

    # Trade Details
    trades: List[TradeAnalysis] = field(default_factory=list)
    daily_performance: List[DailyPerformance] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    drawdown_curve: List[Dict] = field(default_factory=list)


@dataclass
class DataSourceInfo:
    """Track data source information for reporting"""
    underlying_source: str = "Unknown"
    options_source: str = "Synthetic (Advanced Model)"
    underlying_bars: int = 0
    options_bars: int = 0

    def __str__(self):
        return f"Underlying: {self.underlying_source} | Options: {self.options_source}"


class PerformanceAnalyzer:
    """
    Analyzes backtest results and generates comprehensive reports.
    """

    def __init__(self, results: Any, strategy_name: str = "Strategy", data_source_info: DataSourceInfo = None):
        """
        Initialize analyzer with backtest results.

        Parameters
        ----------
        results : BacktestResults
            Results from backtest engine
        strategy_name : str
            Name of the strategy
        data_source_info : DataSourceInfo
            Information about data sources used
        """
        self.results = results
        self.strategy_name = strategy_name
        self.data_source_info = data_source_info or DataSourceInfo()
        self.trades_df = None
        self.equity_df = None

    def analyze(self) -> PerformanceReport:
        """
        Perform comprehensive analysis and return report.

        Returns
        -------
        PerformanceReport
            Complete performance analysis
        """
        # Convert to DataFrames for analysis
        self._prepare_data()

        # Calculate all metrics
        report = PerformanceReport(
            strategy_name=self.strategy_name,
            symbol=self._get_symbol(),
            start_date=self.results.start_date,
            end_date=self.results.end_date,
            trading_days=self.results.duration_days,

            # Capital
            initial_capital=self.results.initial_capital,
            final_equity=self.results.final_equity,
            total_pnl=self.results.total_pnl,
            total_return_pct=self.results.total_return_pct,

            # Trade Stats
            total_trades=self.results.total_trades,
            winning_trades=self.results.winning_trades,
            losing_trades=self.results.losing_trades,
            win_rate=self.results.win_rate,

            # P&L
            **self._calculate_pnl_metrics(),

            # Risk
            **self._calculate_risk_metrics(),

            # Streaks
            **self._calculate_streaks(),

            # Time
            **self._calculate_time_metrics(),

            # Direction
            **self._calculate_direction_metrics(),

            # Exit
            **self._calculate_exit_metrics(),

            # Daily
            **self._calculate_daily_metrics(),

            # ML Ensemble (if available)
            **self._calculate_ml_metrics(),

            # Details
            trades=self._get_trade_analyses(),
            daily_performance=self._get_daily_performance(),
            equity_curve=self.results.equity_curve,
            drawdown_curve=self._calculate_drawdown_curve(),
        )

        return report

    def _prepare_data(self):
        """Prepare DataFrames for analysis"""
        if self.results.trades:
            self.trades_df = pd.DataFrame(self.results.trades)
            if 'entry_time' in self.trades_df.columns:
                self.trades_df['entry_time'] = pd.to_datetime(self.trades_df['entry_time'])
            if 'exit_time' in self.trades_df.columns:
                self.trades_df['exit_time'] = pd.to_datetime(self.trades_df['exit_time'])
        else:
            self.trades_df = pd.DataFrame()

        if self.results.equity_curve:
            self.equity_df = pd.DataFrame(self.results.equity_curve)
            self.equity_df['timestamp'] = pd.to_datetime(self.equity_df['timestamp'])
        else:
            self.equity_df = pd.DataFrame()

    def _get_symbol(self) -> str:
        """Get primary symbol from trades"""
        if not self.trades_df.empty and 'symbol' in self.trades_df.columns:
            # Extract underlying from option symbol
            symbols = self.trades_df['symbol'].unique()
            if len(symbols) > 0:
                sym = symbols[0]
                # Extract underlying (e.g., COIN from COIN241108C00180000)
                for i, c in enumerate(sym):
                    if c.isdigit():
                        return sym[:i]
                return sym
        return "UNKNOWN"

    def _calculate_pnl_metrics(self) -> Dict[str, float]:
        """Calculate P&L related metrics"""
        if self.trades_df.empty:
            return {
                'gross_profit': 0.0,
                'gross_loss': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_trade_pnl': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_win_pct': 0.0,
                'avg_loss_pct': 0.0,
            }

        wins = self.trades_df[self.trades_df['pnl'] > 0]
        losses = self.trades_df[self.trades_df['pnl'] <= 0]

        gross_profit = wins['pnl'].sum() if not wins.empty else 0.0
        gross_loss = abs(losses['pnl'].sum()) if not losses.empty else 0.0

        return {
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'avg_win': wins['pnl'].mean() if not wins.empty else 0.0,
            'avg_loss': losses['pnl'].mean() if not losses.empty else 0.0,
            'avg_trade_pnl': self.trades_df['pnl'].mean(),
            'largest_win': wins['pnl'].max() if not wins.empty else 0.0,
            'largest_loss': losses['pnl'].min() if not losses.empty else 0.0,
            'avg_win_pct': wins['pnl_pct'].mean() if not wins.empty and 'pnl_pct' in wins.columns else 0.0,
            'avg_loss_pct': losses['pnl_pct'].mean() if not losses.empty and 'pnl_pct' in losses.columns else 0.0,
        }

    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate risk metrics"""
        max_dd = self.results.max_drawdown
        max_dd_pct = self.results.max_drawdown_pct
        max_dd_date = None

        # Find max drawdown date
        if not self.equity_df.empty:
            peak = self.equity_df['equity'].iloc[0]
            max_dd_val = 0
            for idx, row in self.equity_df.iterrows():
                if row['equity'] > peak:
                    peak = row['equity']
                dd = peak - row['equity']
                if dd > max_dd_val:
                    max_dd_val = dd
                    max_dd_date = row['timestamp']

        # Calmar ratio = Annual return / Max Drawdown
        annual_return = self.results.total_return_pct
        calmar = annual_return / max_dd_pct if max_dd_pct > 0 else 0.0

        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'max_drawdown_date': max_dd_date,
            'sharpe_ratio': self.results.sharpe_ratio,
            'sortino_ratio': self.results.sortino_ratio,
            'calmar_ratio': calmar,
        }

    def _calculate_streaks(self) -> Dict[str, Any]:
        """Calculate win/loss streaks"""
        if self.trades_df.empty:
            return {
                'max_win_streak': 0,
                'max_loss_streak': 0,
                'current_streak': 0,
                'current_streak_type': 'NONE',
            }

        results = (self.trades_df['pnl'] > 0).tolist()

        max_win = max_loss = current = 0
        current_type = 'NONE'
        prev = None

        for win in results:
            if win:
                if prev == True or prev is None:
                    current += 1
                else:
                    current = 1
                max_win = max(max_win, current)
                current_type = 'WIN'
            else:
                if prev == False or prev is None:
                    current += 1
                else:
                    current = 1
                max_loss = max(max_loss, current)
                current_type = 'LOSS'
            prev = win

        return {
            'max_win_streak': max_win,
            'max_loss_streak': max_loss,
            'current_streak': current,
            'current_streak_type': current_type,
        }

    def _calculate_time_metrics(self) -> Dict[str, float]:
        """Calculate time-related metrics"""
        if self.trades_df.empty or 'duration_seconds' not in self.trades_df.columns:
            return {
                'avg_hold_time_minutes': 0.0,
                'avg_winning_hold_time': 0.0,
                'avg_losing_hold_time': 0.0,
            }

        self.trades_df['duration_minutes'] = self.trades_df['duration_seconds'] / 60

        wins = self.trades_df[self.trades_df['pnl'] > 0]
        losses = self.trades_df[self.trades_df['pnl'] <= 0]

        return {
            'avg_hold_time_minutes': self.trades_df['duration_minutes'].mean(),
            'avg_winning_hold_time': wins['duration_minutes'].mean() if not wins.empty else 0.0,
            'avg_losing_hold_time': losses['duration_minutes'].mean() if not losses.empty else 0.0,
        }

    def _calculate_direction_metrics(self) -> Dict[str, Any]:
        """Calculate CALL vs PUT metrics"""
        if self.trades_df.empty or 'symbol' not in self.trades_df.columns:
            return {
                'call_trades': 0,
                'put_trades': 0,
                'call_win_rate': 0.0,
                'put_win_rate': 0.0,
                'call_pnl': 0.0,
                'put_pnl': 0.0,
            }

        # Determine direction from symbol (C = CALL, P = PUT)
        self.trades_df['direction'] = self.trades_df['symbol'].apply(
            lambda x: 'CALL' if 'C0' in x else ('PUT' if 'P0' in x else 'UNKNOWN')
        )

        calls = self.trades_df[self.trades_df['direction'] == 'CALL']
        puts = self.trades_df[self.trades_df['direction'] == 'PUT']

        call_wins = (calls['pnl'] > 0).sum() if not calls.empty else 0
        put_wins = (puts['pnl'] > 0).sum() if not puts.empty else 0

        return {
            'call_trades': len(calls),
            'put_trades': len(puts),
            'call_win_rate': (call_wins / len(calls) * 100) if len(calls) > 0 else 0.0,
            'put_win_rate': (put_wins / len(puts) * 100) if len(puts) > 0 else 0.0,
            'call_pnl': calls['pnl'].sum() if not calls.empty else 0.0,
            'put_pnl': puts['pnl'].sum() if not puts.empty else 0.0,
        }

    def _calculate_exit_metrics(self) -> Dict[str, int]:
        """Calculate exit reason metrics"""
        # For now, estimate based on P&L
        if self.trades_df.empty:
            return {
                'tp_exits': 0,
                'sl_exits': 0,
                'time_exits': 0,
                'force_exits': 0,
            }

        wins = len(self.trades_df[self.trades_df['pnl'] > 0])
        losses = len(self.trades_df[self.trades_df['pnl'] <= 0])

        return {
            'tp_exits': wins,  # Assume wins are TP
            'sl_exits': losses,  # Assume losses are SL
            'time_exits': 0,
            'force_exits': 0,
        }

    def _calculate_daily_metrics(self) -> Dict[str, Any]:
        """Calculate daily performance metrics"""
        if self.trades_df.empty:
            return {
                'best_day_pnl': 0.0,
                'worst_day_pnl': 0.0,
                'avg_daily_pnl': 0.0,
                'profitable_days': 0,
                'losing_days': 0,
            }

        # Group by date
        if 'entry_time' in self.trades_df.columns:
            self.trades_df['date'] = pd.to_datetime(self.trades_df['entry_time']).dt.date
            daily = self.trades_df.groupby('date')['pnl'].sum()

            return {
                'best_day_pnl': daily.max() if not daily.empty else 0.0,
                'worst_day_pnl': daily.min() if not daily.empty else 0.0,
                'avg_daily_pnl': daily.mean() if not daily.empty else 0.0,
                'profitable_days': (daily > 0).sum(),
                'losing_days': (daily <= 0).sum(),
            }

        return {
            'best_day_pnl': 0.0,
            'worst_day_pnl': 0.0,
            'avg_daily_pnl': 0.0,
            'profitable_days': 0,
            'losing_days': 0,
        }

    def _get_trade_analyses(self) -> List[TradeAnalysis]:
        """Get detailed trade analyses"""
        analyses = []

        for trade in self.results.trades:
            symbol = trade.get('symbol', '')
            direction = 'CALL' if 'C0' in symbol else ('PUT' if 'P0' in symbol else 'UNKNOWN')
            pnl = trade.get('pnl', 0)

            # Get exit_reason from trade data (TP/SL/TIME/FORCE) or infer from P&L
            exit_reason = trade.get('exit_reason', '')
            if not exit_reason:
                exit_reason = 'TP' if pnl > 0 else 'SL'

            # Get signal from trade data (BULLISH/BEARISH/NEUTRAL)
            signal = trade.get('signal', '')
            if not signal:
                # Infer from direction if not set
                signal = 'BULLISH' if direction == 'CALL' else ('BEARISH' if direction == 'PUT' else '')

            analyses.append(TradeAnalysis(
                trade_id=trade.get('trade_id', ''),
                symbol=symbol,
                direction=direction,
                entry_time=pd.to_datetime(trade.get('entry_time')),
                exit_time=pd.to_datetime(trade.get('exit_time')),
                entry_price=trade.get('entry_price', 0),
                exit_price=trade.get('exit_price', 0),
                quantity=trade.get('quantity', 0),
                pnl=pnl,
                pnl_pct=trade.get('pnl_pct', 0),
                duration_minutes=trade.get('duration_seconds', 0) / 60,
                exit_reason=exit_reason,
                signal=signal,
            ))

        return analyses

    def _get_daily_performance(self) -> List[DailyPerformance]:
        """Get daily performance breakdown"""
        if self.trades_df.empty or 'entry_time' not in self.trades_df.columns:
            return []

        self.trades_df['date'] = pd.to_datetime(self.trades_df['entry_time']).dt.date

        daily_perf = []
        for date, group in self.trades_df.groupby('date'):
            wins = (group['pnl'] > 0).sum()
            losses = (group['pnl'] <= 0).sum()
            total_pnl = group['pnl'].sum()

            # Direction from first trade
            first_symbol = group.iloc[0]['symbol'] if not group.empty else ''
            direction = 'CALL' if 'C0' in first_symbol else ('PUT' if 'P0' in first_symbol else 'NONE')

            daily_perf.append(DailyPerformance(
                date=date,
                trades=len(group),
                wins=wins,
                losses=losses,
                pnl=total_pnl,
                pnl_pct=(total_pnl / self.results.initial_capital) * 100,
                direction=direction,
                exit_reason='TP' if total_pnl > 0 else 'SL',
                starting_equity=0,  # Would need equity tracking
                ending_equity=0,
            ))

        return daily_perf

    def _calculate_drawdown_curve(self) -> List[Dict]:
        """Calculate drawdown curve"""
        if self.equity_df.empty:
            return []

        drawdowns = []
        peak = self.equity_df['equity'].iloc[0]

        for idx, row in self.equity_df.iterrows():
            if row['equity'] > peak:
                peak = row['equity']
            dd = peak - row['equity']
            dd_pct = (dd / peak * 100) if peak > 0 else 0

            drawdowns.append({
                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                'drawdown': dd,
                'drawdown_pct': dd_pct,
                'peak': peak,
            })

        return drawdowns

    def _calculate_ml_metrics(self) -> Dict[str, Any]:
        """Calculate ML ensemble performance metrics"""
        if self.trades_df.empty:
            return {
                'ml_enabled': False,
                'ml_signal_trades': 0,
                'ml_signal_win_rate': 0.0,
                'ml_avg_confidence': 0.0,
                'ml_high_conf_trades': 0,
                'ml_high_conf_win_rate': 0.0,
                'ml_contribution_pnl': 0.0,
            }

        # Check if trades have ML metadata
        has_ml_data = any(key in self.trades_df.columns for key in ['ml_probability', 'ml_signal', 'ml_score'])
        
        if not has_ml_data:
            return {
                'ml_enabled': False,
                'ml_signal_trades': 0,
                'ml_signal_win_rate': 0.0,
                'ml_avg_confidence': 0.0,
                'ml_high_conf_trades': 0,
                'ml_high_conf_win_rate': 0.0,
                'ml_contribution_pnl': 0.0,
            }

        # Trades with ML signals (probability >= 0.5)
        ml_trades = self.trades_df[
            self.trades_df.get('ml_probability', pd.Series([0])) >= 0.5
        ] if 'ml_probability' in self.trades_df.columns else pd.DataFrame()

        # High confidence ML trades (probability >= 0.65)
        high_conf_trades = self.trades_df[
            self.trades_df.get('ml_probability', pd.Series([0])) >= 0.65
        ] if 'ml_probability' in self.trades_df.columns else pd.DataFrame()

        # Calculate metrics
        ml_signal_wins = (ml_trades['pnl'] > 0).sum() if not ml_trades.empty else 0
        high_conf_wins = (high_conf_trades['pnl'] > 0).sum() if not high_conf_trades.empty else 0
        
        avg_confidence = self.trades_df.get('ml_probability', pd.Series([0])).mean()
        ml_contribution = ml_trades['pnl'].sum() if not ml_trades.empty else 0.0

        return {
            'ml_enabled': True,
            'ml_signal_trades': len(ml_trades),
            'ml_signal_win_rate': (ml_signal_wins / len(ml_trades) * 100) if len(ml_trades) > 0 else 0.0,
            'ml_avg_confidence': avg_confidence,
            'ml_high_conf_trades': len(high_conf_trades),
            'ml_high_conf_win_rate': (high_conf_wins / len(high_conf_trades) * 100) if len(high_conf_trades) > 0 else 0.0,
            'ml_contribution_pnl': ml_contribution,
        }

    def print_report(self, report: PerformanceReport) -> str:
        """
        Generate formatted report string.

        Parameters
        ----------
        report : PerformanceReport
            Performance report to format

        Returns
        -------
        str
            Formatted report string
        """
        lines = []

        # Header
        lines.append("\n" + "=" * 100)
        lines.append("                    COMPREHENSIVE BACKTEST PERFORMANCE REPORT")
        lines.append("=" * 100)

        # Data Source Info - IMPORTANT: Shows if data is real or synthetic
        lines.append(f"\n{'DATA SOURCE QUALITY':=^100}")
        ds = self.data_source_info
        is_real = "REAL" in ds.underlying_source.upper()
        is_synthetic = "SYNTHETIC" in ds.underlying_source.upper()
        if is_real:
            lines.append(f"  *** REAL MARKET DATA - Results reflect actual market conditions ***")
        elif is_synthetic:
            lines.append(f"  *** SYNTHETIC DATA - Results are simulated, may not reflect reality ***")
        else:
            lines.append(f"  *** UNKNOWN DATA SOURCE - Verify data quality ***")
        lines.append(f"  ")
        lines.append(f"  Underlying Data:    {ds.underlying_source}")
        lines.append(f"  Underlying Bars:    {ds.underlying_bars:,}")
        lines.append(f"  Options Data:       {ds.options_source}")
        lines.append(f"  Options Bars:       {ds.options_bars:,}")
        if "Synthetic" in ds.options_source:
            lines.append(f"  ")
            if "Advanced" in ds.options_source:
                lines.append(f"  NOTE: Options use ADVANCED synthetic pricing model with:")
                lines.append(f"        - IV smile/skew (OTM puts have higher IV)")
                lines.append(f"        - Intraday IV patterns (higher at open/close)")
                lines.append(f"        - 0DTE accelerated time decay")
                lines.append(f"        - Realistic bid-ask spreads")
                lines.append(f"        - Greeks-based OHLC generation")
            else:
                lines.append(f"  NOTE: Options are priced using Black-Scholes model with 80% IV.")
                lines.append(f"        Real options prices may differ significantly.")

        # Strategy Info
        lines.append(f"\n{'STRATEGY INFORMATION':=^100}")
        lines.append(f"  Strategy Name:      {report.strategy_name}")
        lines.append(f"  Symbol:             {report.symbol}")
        lines.append(f"  Period:             {report.start_date} to {report.end_date}")
        lines.append(f"  Trading Days:       {report.trading_days}")

        # Capital & Returns
        lines.append(f"\n{'CAPITAL & RETURNS':=^100}")
        lines.append(f"  Initial Capital:    ${report.initial_capital:,.2f}")
        lines.append(f"  Final Equity:       ${report.final_equity:,.2f}")
        lines.append(f"  Total P&L:          ${report.total_pnl:,.2f}")
        lines.append(f"  Total Return:       {report.total_return_pct:+.2f}%")

        # Trade Statistics
        lines.append(f"\n{'TRADE STATISTICS':=^100}")
        lines.append(f"  Total Trades:       {report.total_trades}")
        lines.append(f"  Winning Trades:     {report.winning_trades} ({report.win_rate:.1f}%)")
        lines.append(f"  Losing Trades:      {report.losing_trades} ({100-report.win_rate:.1f}%)")
        lines.append(f"  ")
        lines.append(f"  Gross Profit:       ${report.gross_profit:,.2f}")
        lines.append(f"  Gross Loss:         ${report.gross_loss:,.2f}")
        lines.append(f"  Profit Factor:      {report.profit_factor:.2f}" if report.profit_factor != float('inf') else "  Profit Factor:      Infinite")
        lines.append(f"  ")
        lines.append(f"  Average Win:        ${report.avg_win:,.2f} ({report.avg_win_pct:+.2f}%)")
        lines.append(f"  Average Loss:       ${report.avg_loss:,.2f} ({report.avg_loss_pct:+.2f}%)")
        lines.append(f"  Average Trade:      ${report.avg_trade_pnl:,.2f}")
        lines.append(f"  ")
        lines.append(f"  Largest Win:        ${report.largest_win:,.2f}")
        lines.append(f"  Largest Loss:       ${report.largest_loss:,.2f}")

        # Risk Metrics
        lines.append(f"\n{'RISK METRICS':=^100}")
        lines.append(f"  Max Drawdown:       ${report.max_drawdown:,.2f} ({report.max_drawdown_pct:.2f}%)")
        if report.max_drawdown_date:
            lines.append(f"  Max DD Date:        {report.max_drawdown_date}")
        lines.append(f"  Sharpe Ratio:       {report.sharpe_ratio:.2f}")
        lines.append(f"  Sortino Ratio:      {report.sortino_ratio:.2f}")
        lines.append(f"  Calmar Ratio:       {report.calmar_ratio:.2f}")

        # Streaks
        lines.append(f"\n{'STREAKS':=^100}")
        lines.append(f"  Max Win Streak:     {report.max_win_streak}")
        lines.append(f"  Max Loss Streak:    {report.max_loss_streak}")
        lines.append(f"  Current Streak:     {report.current_streak} ({report.current_streak_type})")

        # Time Analysis
        lines.append(f"\n{'TIME ANALYSIS':=^100}")
        lines.append(f"  Avg Hold Time:      {report.avg_hold_time_minutes:.1f} minutes")
        lines.append(f"  Avg Win Hold:       {report.avg_winning_hold_time:.1f} minutes")
        lines.append(f"  Avg Loss Hold:      {report.avg_losing_hold_time:.1f} minutes")

        # Direction Analysis
        lines.append(f"\n{'DIRECTION ANALYSIS (CALL vs PUT)':=^100}")
        lines.append(f"  CALL Trades:        {report.call_trades} (Win Rate: {report.call_win_rate:.1f}%)")
        lines.append(f"  CALL P&L:           ${report.call_pnl:,.2f}")
        lines.append(f"  PUT Trades:         {report.put_trades} (Win Rate: {report.put_win_rate:.1f}%)")
        lines.append(f"  PUT P&L:            ${report.put_pnl:,.2f}")

        # Exit Analysis
        lines.append(f"\n{'EXIT ANALYSIS':=^100}")
        lines.append(f"  Take Profit Exits:  {report.tp_exits}")
        lines.append(f"  Stop Loss Exits:    {report.sl_exits}")
        lines.append(f"  Time Exits:         {report.time_exits}")
        lines.append(f"  Force Exits:        {report.force_exits}")

        # Daily Performance
        lines.append(f"\n{'DAILY PERFORMANCE':=^100}")
        lines.append(f"  Best Day P&L:       ${report.best_day_pnl:,.2f}")
        lines.append(f"  Worst Day P&L:      ${report.worst_day_pnl:,.2f}")
        lines.append(f"  Average Day P&L:    ${report.avg_daily_pnl:,.2f}")
        lines.append(f"  Profitable Days:    {report.profitable_days}")
        lines.append(f"  Losing Days:        {report.losing_days}")

        # ML Ensemble Performance (if enabled)
        if report.ml_enabled:
            lines.append(f"\n{'ML ENSEMBLE PERFORMANCE':=^100}")
            lines.append(f"  🤖 ML Status:              ENABLED")
            lines.append(f"  ML Signal Trades:          {report.ml_signal_trades} ({report.ml_signal_trades/report.total_trades*100:.1f}% of total)")
            lines.append(f"  ML Signal Win Rate:        {report.ml_signal_win_rate:.1f}%")
            lines.append(f"  Average ML Confidence:     {report.ml_avg_confidence:.3f}")
            lines.append(f"  ")
            lines.append(f"  High Confidence Trades:    {report.ml_high_conf_trades} (prob >= 0.65)")
            lines.append(f"  High Conf Win Rate:        {report.ml_high_conf_win_rate:.1f}%")
            lines.append(f"  ML Contribution to P&L:    ${report.ml_contribution_pnl:,.2f}")
            
            # ML effectiveness comparison
            if report.ml_signal_win_rate > report.win_rate:
                improvement = report.ml_signal_win_rate - report.win_rate
                lines.append(f"  ")
                lines.append(f"  ✅ ML Improvement:         +{improvement:.1f}% vs overall win rate")
            elif report.ml_signal_win_rate < report.win_rate:
                decline = report.win_rate - report.ml_signal_win_rate
                lines.append(f"  ")
                lines.append(f"  ⚠️  ML Underperformance:    -{decline:.1f}% vs overall win rate")
            else:
                lines.append(f"  ")
                lines.append(f"  ℹ️  ML Performance:         Same as overall win rate")

        # Trade Details Table with Cost Analysis and Hold Time
        if report.trades:
            lines.append(f"\n{'TRADE-BY-TRADE DETAILS (WITH COST ANALYSIS)':=^156}")
            lines.append("-" * 156)
            lines.append(f"{'#':>3} | {'Date':^12} | {'Dir':^5} | {'Signal':^8} | {'Entry':>7} | {'Exit':>7} | {'Qty':>4} | {'Cost/Cont':>10} | {'Total Cost':>11} | {'Exit Value':>10} | {'P&L':>10} | {'%':>7} | {'Hold':>8} | {'Exit':^4}")
            lines.append("-" * 156)

            total_cost = 0.0
            total_exit_value = 0.0
            total_hold_time = 0.0
            for i, trade in enumerate(report.trades, 1):
                date_str = trade.entry_time.strftime('%Y-%m-%d') if trade.entry_time else 'N/A'
                # Cost per contract (premium * 100 shares per contract)
                cost_per_contract = trade.entry_price * 100
                total_trade_cost = cost_per_contract * trade.quantity
                # Exit value per contract
                exit_per_contract = trade.exit_price * 100
                total_exit = exit_per_contract * trade.quantity
                # Hold time
                hold_mins = trade.duration_minutes
                total_hold_time += hold_mins

                total_cost += total_trade_cost
                total_exit_value += total_exit

                # Format hold time
                if hold_mins >= 60:
                    hold_str = f"{hold_mins/60:.1f}h"
                else:
                    hold_str = f"{hold_mins:.1f}m"

                # Format signal (BULL/BEAR/NEUT)
                signal_short = trade.signal[:4].upper() if trade.signal else "N/A"

                lines.append(
                    f"{i:>3} | {date_str:^12} | {trade.direction:^5} | {signal_short:^8} | "
                    f"${trade.entry_price:>5.2f} | ${trade.exit_price:>5.2f} | "
                    f"{trade.quantity:>4} | ${cost_per_contract:>8.2f} | ${total_trade_cost:>9.2f} | "
                    f"${total_exit:>8.2f} | ${trade.pnl:>8.2f} | {trade.pnl_pct:>6.2f}% | {hold_str:>8} | {trade.exit_reason:^4}"
                )
            lines.append("-" * 156)

            # Summary totals
            avg_hold = total_hold_time / len(report.trades) if report.trades else 0
            avg_hold_str = f"{avg_hold:.1f}m" if avg_hold < 60 else f"{avg_hold/60:.1f}h"
            lines.append(f"{'TOTALS':>3} | {' ':^12} | {' ':^5} | {' ':>7} | {' ':>7} | {' ':>4} | {' ':>10} | ${total_cost:>9.2f} | ${total_exit_value:>8.2f} | ${report.total_pnl:>8.2f} | {' ':>7} | {avg_hold_str:>8} | {' ':^4}")
            lines.append("-" * 140)

            # Cost Summary Section
            lines.append(f"\n{'COST SUMMARY':=^140}")
            lines.append(f"  Total Capital Deployed:     ${total_cost:,.2f}")
            lines.append(f"  Total Exit Value:           ${total_exit_value:,.2f}")
            lines.append(f"  Net P&L:                    ${report.total_pnl:,.2f}")
            lines.append(f"  Return on Capital Deployed: {(report.total_pnl / total_cost * 100) if total_cost > 0 else 0:.2f}%")
            lines.append(f"  Average Cost per Trade:     ${total_cost / len(report.trades):,.2f}" if report.trades else "")
            lines.append(f"  Average Hold Time:          {avg_hold:.1f} minutes")

        # Daily Performance Table
        if report.daily_performance:
            lines.append(f"\n{'DAILY PERFORMANCE BREAKDOWN':=^100}")
            lines.append("-" * 100)
            lines.append(f"{'Date':^12} | {'Trades':>6} | {'W/L':>5} | {'Direction':^6} | {'P&L':>12} | {'%':>8} | {'Exit':^6}")
            lines.append("-" * 100)

            for day in report.daily_performance:
                date_str = str(day.date) if day.date else 'N/A'
                wl = f"{day.wins}/{day.losses}"
                lines.append(
                    f"{date_str:^12} | {day.trades:>6} | {wl:>5} | {day.direction:^6} | "
                    f"${day.pnl:>11.2f} | {day.pnl_pct:>7.2f}% | {day.exit_reason:^6}"
                )
            lines.append("-" * 100)

        # Footer
        lines.append("\n" + "=" * 100)
        lines.append("                              END OF PERFORMANCE REPORT")
        lines.append("=" * 100 + "\n")

        return "\n".join(lines)
