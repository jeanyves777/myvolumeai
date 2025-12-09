"""
Comprehensive Crypto Trade Logger

Records every trade with full details for strategy analysis and improvement:
- Entry/exit timestamps, prices, quantities
- Indicator values at entry (RSI, MACD, Bollinger, etc.)
- Signal scores and confidence
- Exit reason (TP, SL, Trailing Stop, Time Exit)
- Actual P&L (realized and unrealized)
- Fees paid
- Market conditions at entry/exit
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
class CryptoTradeRecord:
    """Complete record of a single crypto trade"""
    # Trade identifiers
    trade_id: str
    symbol: str

    # Entry details
    entry_time: str
    entry_price: float
    entry_qty: float
    entry_value: float  # entry_price * entry_qty
    entry_order_id: str = ""

    # Exit details (filled when closed)
    exit_time: str = ""
    exit_price: float = 0.0
    exit_qty: float = 0.0
    exit_value: float = 0.0
    exit_order_id: str = ""
    exit_reason: str = ""  # TP, SL, TRAILING_STOP, TIME_EXIT, MANUAL

    # P&L
    gross_pnl: float = 0.0
    fees_paid: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0

    # Hold duration
    hold_minutes: float = 0.0

    # Indicator values at entry
    entry_rsi: float = 0.0
    entry_macd: float = 0.0
    entry_macd_signal: float = 0.0
    entry_macd_hist: float = 0.0
    entry_bb_upper: float = 0.0
    entry_bb_middle: float = 0.0
    entry_bb_lower: float = 0.0
    entry_bb_pct: float = 0.0  # % between bands
    entry_stoch_k: float = 0.0
    entry_stoch_d: float = 0.0
    entry_adx: float = 0.0
    entry_atr: float = 0.0
    entry_volume: float = 0.0
    entry_volume_sma: float = 0.0

    # Signal details at entry
    signal_score: int = 0
    signal_reasons: str = ""
    ml_prediction: float = 0.0
    ml_confidence: float = 0.0

    # Price levels
    target_price: float = 0.0
    stop_loss_price: float = 0.0
    trailing_stop_price: float = 0.0

    # Market conditions
    market_trend: str = ""  # bullish, bearish, sideways
    volatility_level: str = ""  # low, medium, high

    # Indicator values at exit (for analysis)
    exit_rsi: float = 0.0
    exit_macd: float = 0.0
    exit_price_vs_entry: float = 0.0  # % change

    # Additional metadata
    notes: str = ""
    strategy_version: str = "v1"


class CryptoTradeLogger:
    """
    Logs all crypto trades for analysis and improvement.

    Features:
    - JSON and CSV export
    - Real-time trade tracking
    - Performance summaries
    - Integration with trading engines
    """

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = Path.home() / ".thevolumeai" / "crypto_trade_logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Current day's trades
        self.today_str = datetime.now().strftime("%Y-%m-%d")
        self.trades: Dict[str, CryptoTradeRecord] = {}  # trade_id -> record
        self.completed_trades: List[CryptoTradeRecord] = []

        # Load existing trades for today
        self._load_today_trades()

        logger.info(f"CryptoTradeLogger initialized: {self.log_dir}")

    def _get_trade_file(self, date_str: str = None) -> Path:
        """Get the trade log file for a specific date"""
        if date_str is None:
            date_str = self.today_str
        return self.log_dir / f"crypto_trades_{date_str}.json"

    def _get_csv_file(self, date_str: str = None) -> Path:
        """Get the CSV export file for a specific date"""
        if date_str is None:
            date_str = self.today_str
        return self.log_dir / f"crypto_trades_{date_str}.csv"

    def _load_today_trades(self):
        """Load trades from today's file if it exists"""
        file_path = self._get_trade_file()
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for trade_data in data.get('completed_trades', []):
                        record = CryptoTradeRecord(**trade_data)
                        self.completed_trades.append(record)
                    for trade_id, trade_data in data.get('open_trades', {}).items():
                        record = CryptoTradeRecord(**trade_data)
                        self.trades[trade_id] = record
                logger.info(f"Loaded {len(self.completed_trades)} completed, {len(self.trades)} open trades")
            except Exception as e:
                logger.error(f"Error loading trades: {e}")

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
        logger.debug(f"Exported {len(self.completed_trades)} trades to CSV")

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate daily performance summary"""
        if not self.completed_trades:
            return {'trades': 0}

        trades = self.completed_trades
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl < 0]

        total_pnl = sum(t.net_pnl for t in trades)
        total_fees = sum(t.fees_paid for t in trades)

        return {
            'trades': len(trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(trades) * 100 if trades else 0,
            'total_pnl': round(total_pnl, 2),
            'total_fees': round(total_fees, 2),
            'avg_pnl': round(total_pnl / len(trades), 2) if trades else 0,
            'avg_winner': round(sum(t.net_pnl for t in winners) / len(winners), 2) if winners else 0,
            'avg_loser': round(sum(t.net_pnl for t in losers) / len(losers), 2) if losers else 0,
            'avg_hold_minutes': round(sum(t.hold_minutes for t in trades) / len(trades), 1) if trades else 0,
            'profit_factor': abs(sum(t.net_pnl for t in winners) / sum(t.net_pnl for t in losers)) if losers and sum(t.net_pnl for t in losers) != 0 else float('inf'),
        }

    def log_entry(
        self,
        trade_id: str,
        symbol: str,
        entry_time: datetime,
        entry_price: float,
        entry_qty: float,
        entry_order_id: str = "",
        indicators: Dict[str, float] = None,
        signal_score: int = 0,
        signal_reasons: str = "",
        ml_prediction: float = 0.0,
        ml_confidence: float = 0.0,
        target_price: float = 0.0,
        stop_loss_price: float = 0.0,
        fee_pct: float = 0.25,
        notes: str = ""
    ) -> CryptoTradeRecord:
        """
        Log a new trade entry.

        Parameters
        ----------
        trade_id : str
            Unique identifier for this trade
        symbol : str
            Trading pair (e.g., 'BTC/USD')
        entry_time : datetime
            Entry timestamp
        entry_price : float
            Entry price
        entry_qty : float
            Quantity bought
        entry_order_id : str
            Order ID from exchange
        indicators : Dict[str, float]
            Indicator values at entry
        signal_score : int
            Entry signal score (e.g., 0-10)
        signal_reasons : str
            Reasons for entry signal
        ml_prediction : float
            ML model prediction (0-1)
        ml_confidence : float
            ML model confidence
        target_price : float
            Take profit target
        stop_loss_price : float
            Stop loss level
        fee_pct : float
            Fee percentage (default 0.25%)
        notes : str
            Additional notes

        Returns
        -------
        CryptoTradeRecord
            The created trade record
        """
        indicators = indicators or {}
        entry_value = entry_price * entry_qty
        entry_fee = entry_value * (fee_pct / 100)

        record = CryptoTradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            entry_time=entry_time.isoformat() if isinstance(entry_time, datetime) else str(entry_time),
            entry_price=entry_price,
            entry_qty=entry_qty,
            entry_value=entry_value,
            entry_order_id=entry_order_id,
            fees_paid=entry_fee,

            # Indicators
            entry_rsi=indicators.get('rsi', 0.0),
            entry_macd=indicators.get('macd', 0.0),
            entry_macd_signal=indicators.get('macd_signal', 0.0),
            entry_macd_hist=indicators.get('macd_hist', 0.0),
            entry_bb_upper=indicators.get('bb_upper', 0.0),
            entry_bb_middle=indicators.get('bb_middle', 0.0),
            entry_bb_lower=indicators.get('bb_lower', 0.0),
            entry_bb_pct=indicators.get('bb_pct', 0.0),
            entry_stoch_k=indicators.get('stoch_k', 0.0),
            entry_stoch_d=indicators.get('stoch_d', 0.0),
            entry_adx=indicators.get('adx', 0.0),
            entry_atr=indicators.get('atr', 0.0),
            entry_volume=indicators.get('volume', 0.0),
            entry_volume_sma=indicators.get('volume_sma', 0.0),

            # Signal
            signal_score=signal_score,
            signal_reasons=signal_reasons,
            ml_prediction=ml_prediction,
            ml_confidence=ml_confidence,

            # Targets
            target_price=target_price,
            stop_loss_price=stop_loss_price,

            notes=notes
        )

        self.trades[trade_id] = record
        self._save_trades()

        logger.info(f"Trade ENTRY logged: {trade_id} {symbol} @ ${entry_price:.4f} x {entry_qty:.6f}")
        return record

    def log_exit(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_price: float,
        exit_qty: float,
        exit_order_id: str = "",
        exit_reason: str = "",
        exit_indicators: Dict[str, float] = None,
        fee_pct: float = 0.25,
        notes: str = ""
    ) -> Optional[CryptoTradeRecord]:
        """
        Log a trade exit.

        Parameters
        ----------
        trade_id : str
            Trade ID to close
        exit_time : datetime
            Exit timestamp
        exit_price : float
            Exit price
        exit_qty : float
            Quantity sold
        exit_order_id : str
            Order ID from exchange
        exit_reason : str
            Why the trade was closed (TP, SL, TRAILING_STOP, TIME_EXIT, MANUAL)
        exit_indicators : Dict[str, float]
            Indicator values at exit
        fee_pct : float
            Fee percentage (default 0.25%)
        notes : str
            Additional notes

        Returns
        -------
        CryptoTradeRecord or None
            The updated trade record, or None if trade not found
        """
        if trade_id not in self.trades:
            logger.warning(f"Trade {trade_id} not found for exit")
            return None

        record = self.trades[trade_id]
        exit_indicators = exit_indicators or {}

        exit_value = exit_price * exit_qty
        exit_fee = exit_value * (fee_pct / 100)

        # Calculate P&L
        gross_pnl = exit_value - record.entry_value
        total_fees = record.fees_paid + exit_fee
        net_pnl = gross_pnl - exit_fee  # Only exit fee since entry fee already deducted
        pnl_pct = (exit_price - record.entry_price) / record.entry_price * 100

        # Calculate hold duration
        entry_dt = datetime.fromisoformat(record.entry_time.replace('Z', '+00:00')) if 'Z' in record.entry_time or '+' in record.entry_time else datetime.fromisoformat(record.entry_time)
        exit_dt = exit_time if isinstance(exit_time, datetime) else datetime.fromisoformat(str(exit_time))

        # Handle timezone-aware vs naive datetimes
        if entry_dt.tzinfo is not None and exit_dt.tzinfo is None:
            entry_dt = entry_dt.replace(tzinfo=None)
        elif entry_dt.tzinfo is None and exit_dt.tzinfo is not None:
            exit_dt = exit_dt.replace(tzinfo=None)

        hold_minutes = (exit_dt - entry_dt).total_seconds() / 60

        # Update record
        record.exit_time = exit_time.isoformat() if isinstance(exit_time, datetime) else str(exit_time)
        record.exit_price = exit_price
        record.exit_qty = exit_qty
        record.exit_value = exit_value
        record.exit_order_id = exit_order_id
        record.exit_reason = exit_reason
        record.gross_pnl = gross_pnl
        record.fees_paid = total_fees
        record.net_pnl = net_pnl
        record.pnl_pct = pnl_pct
        record.hold_minutes = hold_minutes
        record.exit_rsi = exit_indicators.get('rsi', 0.0)
        record.exit_macd = exit_indicators.get('macd', 0.0)
        record.exit_price_vs_entry = pnl_pct

        if notes:
            record.notes = f"{record.notes} | EXIT: {notes}" if record.notes else notes

        # Move to completed
        self.completed_trades.append(record)
        del self.trades[trade_id]
        self._save_trades()

        logger.info(f"Trade EXIT logged: {trade_id} {record.symbol} @ ${exit_price:.4f} | "
                   f"P&L: ${net_pnl:.2f} ({pnl_pct:+.2f}%) | Reason: {exit_reason}")

        return record

    def update_trailing_stop(self, trade_id: str, new_trailing_stop: float):
        """Update the trailing stop price for an open trade"""
        if trade_id in self.trades:
            self.trades[trade_id].trailing_stop_price = new_trailing_stop
            self._save_trades()

    def get_open_trades(self) -> Dict[str, CryptoTradeRecord]:
        """Get all currently open trades"""
        return self.trades.copy()

    def get_completed_trades(self) -> List[CryptoTradeRecord]:
        """Get all completed trades for today"""
        return self.completed_trades.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        return self._calculate_summary()

    def get_trades_for_analysis(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get trades in a DataFrame for analysis.

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

        # Determine date range
        if start_date is None:
            start_date = self.today_str
        if end_date is None:
            end_date = self.today_str

        # Load trades from each day
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
                    logger.error(f"Error loading trades for {date_str}: {e}")

            current = current + pd.Timedelta(days=1)

        if not all_trades:
            return pd.DataFrame()

        return pd.DataFrame(all_trades)

    def print_summary(self):
        """Print formatted summary to console"""
        summary = self._calculate_summary()

        print("\n" + "=" * 60)
        print("CRYPTO TRADE SUMMARY - " + self.today_str)
        print("=" * 60)
        print(f"Total Trades:     {summary['trades']}")
        print(f"Winners:          {summary['winners']} ({summary['win_rate']:.1f}%)")
        print(f"Losers:           {summary['losers']}")
        print(f"Total P&L:        ${summary['total_pnl']:.2f}")
        print(f"Total Fees:       ${summary['total_fees']:.2f}")
        print(f"Avg P&L/Trade:    ${summary['avg_pnl']:.2f}")
        print(f"Avg Winner:       ${summary['avg_winner']:.2f}")
        print(f"Avg Loser:        ${summary['avg_loser']:.2f}")
        print(f"Profit Factor:    {summary['profit_factor']:.2f}")
        print(f"Avg Hold Time:    {summary['avg_hold_minutes']:.1f} min")
        print("=" * 60 + "\n")


# Singleton instance
_logger_instance = None


def get_crypto_trade_logger() -> CryptoTradeLogger:
    """Get singleton trade logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = CryptoTradeLogger()
    return _logger_instance
