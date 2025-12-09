"""
Comprehensive Options Trade Logger

Records every options trade with full details for strategy analysis and improvement:
- Entry/exit timestamps, prices, quantities
- Option-specific data (strike, expiry, type, greeks)
- Underlying stock data at entry/exit
- Signal scores and confidence
- Exit reason and P&L
"""

import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptionsTradeRecord:
    """Complete record of a single options trade"""
    # Trade identifiers
    trade_id: str
    underlying_symbol: str
    option_symbol: str

    # Option details
    option_type: str  # 'call' or 'put'
    strike_price: float
    expiration_date: str
    days_to_expiry: int = 0

    # Entry details
    entry_time: str = ""
    entry_price: float = 0.0  # Option premium
    entry_qty: int = 0  # Number of contracts
    entry_value: float = 0.0  # Total premium paid (price * qty * 100)
    entry_order_id: str = ""

    # Underlying at entry
    entry_underlying_price: float = 0.0
    entry_underlying_volume: float = 0.0

    # Greeks at entry
    entry_delta: float = 0.0
    entry_gamma: float = 0.0
    entry_theta: float = 0.0
    entry_vega: float = 0.0
    entry_iv: float = 0.0  # Implied volatility

    # Exit details
    exit_time: str = ""
    exit_price: float = 0.0
    exit_qty: int = 0
    exit_value: float = 0.0
    exit_order_id: str = ""
    exit_reason: str = ""  # TP, SL, EXPIRY, MANUAL, DELTA_EXIT, TIME_EXIT

    # Underlying at exit
    exit_underlying_price: float = 0.0
    exit_underlying_change_pct: float = 0.0

    # Greeks at exit
    exit_delta: float = 0.0
    exit_iv: float = 0.0

    # P&L
    gross_pnl: float = 0.0
    fees_paid: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0  # % return on premium

    # Hold duration
    hold_minutes: float = 0.0
    hold_days: float = 0.0

    # Signal details at entry
    signal_score: int = 0
    signal_reasons: str = ""
    ml_prediction: float = 0.0
    ml_confidence: float = 0.0

    # Strategy parameters
    target_profit_pct: float = 0.0
    stop_loss_pct: float = 0.0
    max_hold_days: int = 0

    # Market conditions
    vix_at_entry: float = 0.0
    vix_at_exit: float = 0.0
    market_trend: str = ""  # bullish, bearish, sideways

    # Technical indicators on underlying at entry
    underlying_rsi: float = 0.0
    underlying_macd: float = 0.0
    underlying_bb_pct: float = 0.0  # % between bands

    # Additional metadata
    notes: str = ""
    strategy_version: str = "v1"


class OptionsTradeLogger:
    """
    Logs all options trades for analysis and improvement.

    Features:
    - JSON and CSV export
    - Real-time trade tracking
    - Performance summaries
    - Greeks tracking
    - Integration with trading engines
    """

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = Path.home() / ".thevolumeai" / "options_trade_logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Current day's trades
        self.today_str = datetime.now().strftime("%Y-%m-%d")
        self.trades: Dict[str, OptionsTradeRecord] = {}  # trade_id -> record
        self.completed_trades: List[OptionsTradeRecord] = []

        # Load existing trades for today
        self._load_today_trades()

        logger.info(f"OptionsTradeLogger initialized: {self.log_dir}")

    def _get_trade_file(self, date_str: str = None) -> Path:
        """Get the trade log file for a specific date"""
        if date_str is None:
            date_str = self.today_str
        return self.log_dir / f"options_trades_{date_str}.json"

    def _get_csv_file(self, date_str: str = None) -> Path:
        """Get the CSV export file for a specific date"""
        if date_str is None:
            date_str = self.today_str
        return self.log_dir / f"options_trades_{date_str}.csv"

    def _load_today_trades(self):
        """Load trades from today's file if it exists"""
        file_path = self._get_trade_file()
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for trade_data in data.get('completed_trades', []):
                        record = OptionsTradeRecord(**trade_data)
                        self.completed_trades.append(record)
                    for trade_id, trade_data in data.get('open_trades', {}).items():
                        record = OptionsTradeRecord(**trade_data)
                        self.trades[trade_id] = record
                logger.info(f"Loaded {len(self.completed_trades)} completed, {len(self.trades)} open options trades")
            except Exception as e:
                logger.error(f"Error loading options trades: {e}")

    def _save_trades(self):
        """Save all trades to JSON file"""
        file_path = self._get_trade_file()
        data = {
            'date': self.today_str,
            'open_trades': {tid: asdict(t) for tid, t in self.trades.items()},
            'completed_trades': [asdict(t) for t in self.completed_trades],
            'summary': self._calculate_summary()
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        # Also export to CSV for easy analysis
        self._export_to_csv()

    def _export_to_csv(self):
        """Export completed trades to CSV"""
        if not self.completed_trades:
            return

        csv_file = self._get_csv_file()
        df = pd.DataFrame([asdict(t) for t in self.completed_trades])
        df.to_csv(csv_file, index=False)
        logger.debug(f"Exported {len(self.completed_trades)} options trades to CSV")

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate daily performance summary"""
        if not self.completed_trades:
            return {'trades': 0}

        trades = self.completed_trades
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl < 0]

        calls = [t for t in trades if t.option_type == 'call']
        puts = [t for t in trades if t.option_type == 'put']

        total_pnl = sum(t.net_pnl for t in trades)
        total_fees = sum(t.fees_paid for t in trades)

        return {
            'trades': len(trades),
            'calls': len(calls),
            'puts': len(puts),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(trades) * 100 if trades else 0,
            'total_pnl': round(total_pnl, 2),
            'total_fees': round(total_fees, 2),
            'avg_pnl': round(total_pnl / len(trades), 2) if trades else 0,
            'avg_winner': round(sum(t.net_pnl for t in winners) / len(winners), 2) if winners else 0,
            'avg_loser': round(sum(t.net_pnl for t in losers) / len(losers), 2) if losers else 0,
            'avg_hold_days': round(sum(t.hold_days for t in trades) / len(trades), 2) if trades else 0,
            'profit_factor': abs(sum(t.net_pnl for t in winners) / sum(t.net_pnl for t in losers)) if losers and sum(t.net_pnl for t in losers) != 0 else float('inf'),
            'call_win_rate': len([t for t in calls if t.net_pnl > 0]) / len(calls) * 100 if calls else 0,
            'put_win_rate': len([t for t in puts if t.net_pnl > 0]) / len(puts) * 100 if puts else 0,
        }

    def log_entry(
        self,
        trade_id: str,
        underlying_symbol: str,
        option_symbol: str,
        option_type: str,
        strike_price: float,
        expiration_date: str,
        entry_time: datetime,
        entry_price: float,
        entry_qty: int,
        entry_order_id: str = "",
        entry_underlying_price: float = 0.0,
        greeks: Dict[str, float] = None,
        signal_score: int = 0,
        signal_reasons: str = "",
        ml_prediction: float = 0.0,
        ml_confidence: float = 0.0,
        target_profit_pct: float = 0.0,
        stop_loss_pct: float = 0.0,
        underlying_indicators: Dict[str, float] = None,
        fee_per_contract: float = 0.65,
        notes: str = ""
    ) -> OptionsTradeRecord:
        """
        Log a new options trade entry.

        Parameters
        ----------
        trade_id : str
            Unique identifier for this trade
        underlying_symbol : str
            Underlying stock symbol (e.g., 'AAPL')
        option_symbol : str
            Option contract symbol
        option_type : str
            'call' or 'put'
        strike_price : float
            Strike price
        expiration_date : str
            Expiration date (YYYY-MM-DD)
        entry_time : datetime
            Entry timestamp
        entry_price : float
            Option premium per share
        entry_qty : int
            Number of contracts
        entry_order_id : str
            Order ID from broker
        entry_underlying_price : float
            Underlying stock price at entry
        greeks : Dict[str, float]
            Option greeks (delta, gamma, theta, vega, iv)
        signal_score : int
            Entry signal score
        signal_reasons : str
            Reasons for entry signal
        ml_prediction : float
            ML model prediction
        ml_confidence : float
            ML model confidence
        target_profit_pct : float
            Target profit percentage
        stop_loss_pct : float
            Stop loss percentage
        underlying_indicators : Dict[str, float]
            Technical indicators on underlying
        fee_per_contract : float
            Fee per contract (default $0.65)
        notes : str
            Additional notes

        Returns
        -------
        OptionsTradeRecord
            The created trade record
        """
        greeks = greeks or {}
        underlying_indicators = underlying_indicators or {}

        entry_value = entry_price * entry_qty * 100  # Each contract = 100 shares
        entry_fee = fee_per_contract * entry_qty

        # Calculate days to expiry
        exp_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        entry_dt = entry_time if isinstance(entry_time, datetime) else datetime.fromisoformat(str(entry_time))
        days_to_expiry = (exp_date - entry_dt.replace(tzinfo=None)).days

        record = OptionsTradeRecord(
            trade_id=trade_id,
            underlying_symbol=underlying_symbol,
            option_symbol=option_symbol,
            option_type=option_type,
            strike_price=strike_price,
            expiration_date=expiration_date,
            days_to_expiry=days_to_expiry,

            entry_time=entry_time.isoformat() if isinstance(entry_time, datetime) else str(entry_time),
            entry_price=entry_price,
            entry_qty=entry_qty,
            entry_value=entry_value,
            entry_order_id=entry_order_id,

            entry_underlying_price=entry_underlying_price,

            # Greeks
            entry_delta=greeks.get('delta', 0.0),
            entry_gamma=greeks.get('gamma', 0.0),
            entry_theta=greeks.get('theta', 0.0),
            entry_vega=greeks.get('vega', 0.0),
            entry_iv=greeks.get('iv', 0.0),

            # Signal
            signal_score=signal_score,
            signal_reasons=signal_reasons,
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence,

            # Strategy
            target_profit_pct=target_profit_pct,
            stop_loss_pct=stop_loss_pct,

            # Underlying indicators
            underlying_rsi=underlying_indicators.get('rsi', 0.0),
            underlying_macd=underlying_indicators.get('macd', 0.0),
            underlying_bb_pct=underlying_indicators.get('bb_pct', 0.0),

            fees_paid=entry_fee,
            notes=notes
        )

        self.trades[trade_id] = record
        self._save_trades()

        logger.info(f"Options ENTRY logged: {trade_id} {option_type.upper()} {underlying_symbol} "
                   f"${strike_price} exp:{expiration_date} @ ${entry_price:.2f} x {entry_qty}")
        return record

    def log_exit(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_price: float,
        exit_qty: int,
        exit_order_id: str = "",
        exit_reason: str = "",
        exit_underlying_price: float = 0.0,
        exit_greeks: Dict[str, float] = None,
        fee_per_contract: float = 0.65,
        notes: str = ""
    ) -> Optional[OptionsTradeRecord]:
        """
        Log an options trade exit.

        Parameters
        ----------
        trade_id : str
            Trade ID to close
        exit_time : datetime
            Exit timestamp
        exit_price : float
            Exit premium per share
        exit_qty : int
            Number of contracts sold
        exit_order_id : str
            Order ID from broker
        exit_reason : str
            Why the trade was closed
        exit_underlying_price : float
            Underlying stock price at exit
        exit_greeks : Dict[str, float]
            Option greeks at exit
        fee_per_contract : float
            Fee per contract (default $0.65)
        notes : str
            Additional notes

        Returns
        -------
        OptionsTradeRecord or None
            The updated trade record, or None if trade not found
        """
        if trade_id not in self.trades:
            logger.warning(f"Options trade {trade_id} not found for exit")
            return None

        record = self.trades[trade_id]
        exit_greeks = exit_greeks or {}

        exit_value = exit_price * exit_qty * 100
        exit_fee = fee_per_contract * exit_qty

        # Calculate P&L
        gross_pnl = exit_value - record.entry_value
        total_fees = record.fees_paid + exit_fee
        net_pnl = gross_pnl - exit_fee
        pnl_pct = (exit_value - record.entry_value) / record.entry_value * 100 if record.entry_value > 0 else 0

        # Calculate underlying change
        underlying_change = 0.0
        if record.entry_underlying_price > 0:
            underlying_change = (exit_underlying_price - record.entry_underlying_price) / record.entry_underlying_price * 100

        # Calculate hold duration
        entry_dt = datetime.fromisoformat(record.entry_time.replace('Z', '+00:00')) if 'Z' in record.entry_time or '+' in record.entry_time else datetime.fromisoformat(record.entry_time)
        exit_dt = exit_time if isinstance(exit_time, datetime) else datetime.fromisoformat(str(exit_time))

        # Handle timezone-aware vs naive datetimes
        if entry_dt.tzinfo is not None and exit_dt.tzinfo is None:
            entry_dt = entry_dt.replace(tzinfo=None)
        elif entry_dt.tzinfo is None and exit_dt.tzinfo is not None:
            exit_dt = exit_dt.replace(tzinfo=None)

        hold_minutes = (exit_dt - entry_dt).total_seconds() / 60
        hold_days = hold_minutes / (60 * 24)

        # Update record
        record.exit_time = exit_time.isoformat() if isinstance(exit_time, datetime) else str(exit_time)
        record.exit_price = exit_price
        record.exit_qty = exit_qty
        record.exit_value = exit_value
        record.exit_order_id = exit_order_id
        record.exit_reason = exit_reason
        record.exit_underlying_price = exit_underlying_price
        record.exit_underlying_change_pct = underlying_change
        record.exit_delta = exit_greeks.get('delta', 0.0)
        record.exit_iv = exit_greeks.get('iv', 0.0)
        record.gross_pnl = gross_pnl
        record.fees_paid = total_fees
        record.net_pnl = net_pnl
        record.pnl_pct = pnl_pct
        record.hold_minutes = hold_minutes
        record.hold_days = hold_days

        if notes:
            record.notes = f"{record.notes} | EXIT: {notes}" if record.notes else notes

        # Move to completed
        self.completed_trades.append(record)
        del self.trades[trade_id]
        self._save_trades()

        logger.info(f"Options EXIT logged: {trade_id} {record.option_type.upper()} {record.underlying_symbol} "
                   f"@ ${exit_price:.2f} | P&L: ${net_pnl:.2f} ({pnl_pct:+.1f}%) | Reason: {exit_reason}")

        return record

    def get_open_trades(self) -> Dict[str, OptionsTradeRecord]:
        """Get all currently open options trades"""
        return self.trades.copy()

    def get_completed_trades(self) -> List[OptionsTradeRecord]:
        """Get all completed options trades for today"""
        return self.completed_trades.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        return self._calculate_summary()

    def get_trades_for_analysis(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get options trades in a DataFrame for analysis.

        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD format)
        end_date : str
            End date (YYYY-MM-DD format)

        Returns
        -------
        pd.DataFrame
            DataFrame with all trade records
        """
        all_trades = []

        if start_date is None:
            start_date = self.today_str
        if end_date is None:
            end_date = self.today_str

        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            file_path = self._get_trade_file(date_str)

            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        for trade_data in data.get('completed_trades', []):
                            all_trades.append(trade_data)
                except Exception as e:
                    logger.error(f"Error loading options trades for {date_str}: {e}")

            current = current + pd.Timedelta(days=1)

        if not all_trades:
            return pd.DataFrame()

        return pd.DataFrame(all_trades)

    def print_summary(self):
        """Print formatted summary to console"""
        summary = self._calculate_summary()

        print("\n" + "=" * 60)
        print("OPTIONS TRADE SUMMARY - " + self.today_str)
        print("=" * 60)
        print(f"Total Trades:     {summary['trades']}")
        print(f"  Calls:          {summary['calls']}")
        print(f"  Puts:           {summary['puts']}")
        print(f"Winners:          {summary['winners']} ({summary['win_rate']:.1f}%)")
        print(f"Losers:           {summary['losers']}")
        print(f"Total P&L:        ${summary['total_pnl']:.2f}")
        print(f"Total Fees:       ${summary['total_fees']:.2f}")
        print(f"Avg P&L/Trade:    ${summary['avg_pnl']:.2f}")
        print(f"Avg Winner:       ${summary['avg_winner']:.2f}")
        print(f"Avg Loser:        ${summary['avg_loser']:.2f}")
        print(f"Profit Factor:    {summary['profit_factor']:.2f}")
        print(f"Avg Hold Time:    {summary['avg_hold_days']:.2f} days")
        print(f"Call Win Rate:    {summary['call_win_rate']:.1f}%")
        print(f"Put Win Rate:     {summary['put_win_rate']:.1f}%")
        print("=" * 60 + "\n")


# Singleton instance
_options_logger_instance = None


def get_options_trade_logger() -> OptionsTradeLogger:
    """Get singleton options trade logger instance"""
    global _options_logger_instance
    if _options_logger_instance is None:
        _options_logger_instance = OptionsTradeLogger()
    return _options_logger_instance
