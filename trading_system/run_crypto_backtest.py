"""
Crypto Scalping Strategy Backtest Runner

Usage:
    # Quick backtest (November 2024)
    python -m trading_system.run_crypto_backtest --symbols BTC/USD ETH/USD --start 2024-11-01 --end 2024-11-30

    # Full 3-month backtest (Sept-Nov 2024)
    python -m trading_system.run_crypto_backtest --symbols BTC/USD ETH/USD SOL/USD --start 2024-09-01 --end 2024-11-30

    # All symbols
    python -m trading_system.run_crypto_backtest --start 2024-11-01 --end 2024-11-30 --all-symbols
"""

import argparse
import asyncio
import io
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from dotenv import load_dotenv
import pytz

# Load environment variables
load_dotenv()

from trading_system.core.models import Bar, Instrument, InstrumentType
from trading_system.strategies.crypto_scalping import (
    CryptoScalping, CryptoScalpingConfig, ALPACA_CRYPTO_SYMBOLS
)
from trading_system.analytics.performance import (
    PerformanceAnalyzer, DataSourceInfo
)


async def fetch_crypto_data(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = '1Min'
) -> pd.DataFrame:
    """
    Fetch crypto data from available sources.

    Tries (in order):
    1. Binance.US API (works in US)
    2. Binance Global API
    3. CryptoCompare API (FREE 1-minute data globally!)
    4. Alpaca Crypto API
    5. Yahoo Finance (1-hour only as last resort)
    """
    print(f"   Fetching {symbol}...")

    # Convert symbol format: BTC/USD -> BTCUSDT for Binance, BTC for CryptoCompare
    binance_symbol = symbol.replace('/USD', 'USDT').replace('/', '')
    cc_symbol = symbol.replace('/USD', '').replace('/', '')

    # Try Binance.US first (works in United States)
    try:
        df = await fetch_binance_us_klines(binance_symbol, start_date, end_date, '1m')
        if df is not None and len(df) > 0:
            print(f"      Binance.US: {len(df):,} bars (1m) - REAL DATA")
            return df
    except Exception as e:
        print(f"      Binance.US failed: {e}")

    # Try Binance Global
    try:
        df = await fetch_binance_klines(binance_symbol, start_date, end_date, '1m')
        if df is not None and len(df) > 0:
            print(f"      Binance: {len(df):,} bars (1m) - REAL DATA")
            return df
    except Exception as e:
        print(f"      Binance failed: {e}")

    # Try CryptoCompare (FREE global API for 1-minute data!)
    try:
        df = await fetch_cryptocompare_data(cc_symbol, start_date, end_date)
        if df is not None and len(df) > 0:
            print(f"      CryptoCompare: {len(df):,} bars (1m) - REAL DATA")
            return df
    except Exception as e:
        print(f"      CryptoCompare failed: {e}")

    # Try Alpaca as backup
    alpaca_key = os.getenv('ALPACA_PAPER_KEY') or os.getenv('ALPACA_API_KEY')
    alpaca_secret = os.getenv('ALPACA_PAPER_SECRET') or os.getenv('ALPACA_API_SECRET')

    if alpaca_key and alpaca_secret:
        try:
            from trading_system.engine.alpaca_client import AlpacaClient
            client = AlpacaClient(alpaca_key, alpaca_secret, paper=True)

            start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC) + timedelta(days=1)

            bars = client.get_crypto_bars(symbol, timeframe, start_dt, end_dt, limit=50000)

            if bars and len(bars) > 0:
                df = pd.DataFrame([{
                    'timestamp': b.timestamp,
                    'open': b.open,
                    'high': b.high,
                    'low': b.low,
                    'close': b.close,
                    'volume': b.volume,
                } for b in bars])
                print(f"      Alpaca: {len(df):,} bars (1m) - REAL DATA")
                return df

        except Exception as e:
            print(f"      Alpaca failed: {e}")

    # Try Yahoo Finance as last resort (1-hour only)
    try:
        import yfinance as yf

        yf_symbol = symbol.replace('/', '-')
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(start=start_date, end=end_date, interval='1h')

        if len(hist) > 0:
            df = pd.DataFrame({
                'timestamp': hist.index.tz_localize('UTC') if hist.index.tz is None else hist.index.tz_convert('UTC'),
                'open': hist['Open'],
                'high': hist['High'],
                'low': hist['Low'],
                'close': hist['Close'],
                'volume': hist['Volume'].astype(int),
            }).reset_index(drop=True)
            print(f"      Yahoo: {len(df):,} bars (1h) - REAL DATA (WARNING: 1h bars not ideal for scalping)")
            return df

    except Exception as e:
        print(f"      Yahoo failed: {e}")

    # NO SYNTHETIC DATA - Only use real data
    print(f"      ERROR: No real data available for {symbol} - SKIPPING")
    print(f"      (Synthetic data is disabled to ensure backtest accuracy)")
    return None


async def fetch_binance_us_klines(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = '1m'
) -> pd.DataFrame:
    """
    Fetch klines from Binance.US API (works in United States).
    """
    import requests

    base_url = "https://api.binance.us/api/v3/klines"

    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int((datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).timestamp() * 1000)

    all_klines = []
    current_start = start_ts
    limit = 1000

    print(f"      Fetching from Binance.US: {symbol} ({interval})...")

    while current_start < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_ts,
            'limit': limit
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            klines = response.json()

            if not klines:
                break

            all_klines.extend(klines)
            last_ts = klines[-1][0]
            if last_ts <= current_start:
                break
            current_start = last_ts + 1

            if len(all_klines) % 50000 == 0:
                print(f"         Fetched {len(all_klines):,} bars...")

        except Exception as e:
            print(f"      Binance.US API error: {e}")
            break

    if not all_klines:
        return None

    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


async def fetch_cryptocompare_data(
    symbol: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetch 1-minute data from CryptoCompare API.

    CryptoCompare provides FREE 1-minute historical data globally!
    No API key required for basic access (2000 calls/day limit).
    """
    import requests

    # CryptoCompare histominute endpoint
    base_url = "https://min-api.cryptocompare.com/data/v2/histominute"

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)

    all_data = []
    current_end = int(end_dt.timestamp())
    limit = 2000  # Max per request

    print(f"      Fetching from CryptoCompare: {symbol} (1m)...")

    # Fetch in reverse chronological order
    while True:
        params = {
            'fsym': symbol,
            'tsym': 'USD',
            'limit': limit,
            'toTs': current_end,
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get('Response') != 'Success' or not data.get('Data', {}).get('Data'):
                break

            bars = data['Data']['Data']
            if not bars:
                break

            all_data = bars + all_data

            # Check if we've gone past start date
            oldest_ts = bars[0]['time']
            if oldest_ts <= start_dt.timestamp():
                break

            current_end = oldest_ts - 1

            if len(all_data) % 50000 == 0:
                print(f"         Fetched {len(all_data):,} bars...")

        except Exception as e:
            print(f"      CryptoCompare API error: {e}")
            break

    if not all_data:
        return None

    # Filter to date range
    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()
    all_data = [b for b in all_data if start_ts <= b['time'] < end_ts]

    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volumefrom': 'volume'
    })

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


async def fetch_binance_klines(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = '1m'
) -> pd.DataFrame:
    """
    Fetch klines (candlestick) data from Binance API.

    Binance provides FREE 1-minute data - no API key required for public endpoints!

    Parameters
    ----------
    symbol : str
        Binance symbol (e.g., 'BTCUSDT', 'ETHUSDT')
    start_date : str
        Start date 'YYYY-MM-DD'
    end_date : str
        End date 'YYYY-MM-DD'
    interval : str
        Kline interval: '1m', '3m', '5m', '15m', '1h', '4h', '1d'

    Returns
    -------
    pd.DataFrame
        DataFrame with OHLCV data
    """
    import requests

    # Binance API endpoint (public - no auth required!)
    base_url = "https://api.binance.com/api/v3/klines"

    # Convert dates to milliseconds
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int((datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).timestamp() * 1000)

    all_klines = []
    current_start = start_ts
    limit = 1000  # Binance max per request

    print(f"      Fetching from Binance: {symbol} ({interval})...")

    while current_start < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_ts,
            'limit': limit
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            klines = response.json()

            if not klines:
                break

            all_klines.extend(klines)

            # Move to next batch
            last_ts = klines[-1][0]
            if last_ts <= current_start:
                break
            current_start = last_ts + 1

            # Progress indicator
            if len(all_klines) % 10000 == 0:
                print(f"         Fetched {len(all_klines):,} bars...")

        except Exception as e:
            print(f"      Binance API error: {e}")
            break

    if not all_klines:
        return None

    # Convert to DataFrame
    # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # Convert types
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # Select columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    return df


def generate_synthetic_crypto_data(
    symbol: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Generate realistic synthetic crypto data for backtesting."""
    import numpy as np

    # Base prices for different cryptos (approximate)
    base_prices = {
        'BTC/USD': 60000,
        'ETH/USD': 3000,
        'SOL/USD': 150,
        'DOGE/USD': 0.15,
        'LINK/USD': 15,
        'AVAX/USD': 35,
        'DOT/USD': 7,
        'LTC/USD': 80,
        'UNI/USD': 10,
        'SHIB/USD': 0.000025,
    }

    # Volatility for different cryptos
    volatilities = {
        'BTC/USD': 0.02,
        'ETH/USD': 0.025,
        'SOL/USD': 0.04,
        'DOGE/USD': 0.05,
        'LINK/USD': 0.035,
        'AVAX/USD': 0.04,
        'DOT/USD': 0.035,
        'LTC/USD': 0.03,
        'UNI/USD': 0.04,
        'SHIB/USD': 0.06,
    }

    base_price = base_prices.get(symbol, 100)
    volatility = volatilities.get(symbol, 0.03)

    # Generate 1-minute bars for 24/7 crypto
    dates = pd.date_range(start=start_date, end=end_date, freq='1T', tz='UTC')

    # Random walk with mean reversion
    n = len(dates)
    returns = np.random.normal(0, volatility / np.sqrt(1440), n)  # Daily vol scaled to 1-min

    # Add some trending behavior
    trend = np.cumsum(np.random.randn(n) * 0.0001)
    returns = returns + trend * 0.01

    # Generate prices
    log_prices = np.log(base_price) + np.cumsum(returns)
    prices = np.exp(log_prices)

    # Generate OHLC
    noise = np.abs(np.random.randn(n)) * volatility * 0.3
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 - noise * np.random.rand(n)),
        'high': prices * (1 + noise),
        'low': prices * (1 - noise),
        'close': prices,
        'volume': np.random.randint(100, 10000, n).astype(float),
    })

    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    print(f"      Generated {len(df):,} synthetic bars")
    return df


@dataclass
class CryptoBacktestResults:
    """Results object compatible with PerformanceAnalyzer."""
    start_date: datetime
    end_date: datetime
    duration_days: int
    initial_capital: float
    final_equity: float
    total_pnl: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    trades: list
    equity_curve: list


class SimpleCryptoBacktestEngine:
    """
    Simplified backtest engine for crypto scalping.

    Features:
    - Multi-symbol support
    - Simple order execution
    - P&L tracking
    - Trade logging
    - Full analytics integration
    """

    def __init__(self, initial_capital: float = 100000, commission_pct: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct

        self.capital = initial_capital
        self.positions = {}  # symbol -> {qty, entry_price, entry_time}
        self.trades = []
        self.equity_curve = []
        self.start_date = None
        self.end_date = None
        self.data_sources = {}  # symbol -> source info

    def run(
        self,
        strategy: CryptoScalping,
        data: dict  # symbol -> DataFrame
    ):
        """
        Run backtest with given strategy and data.

        Parameters
        ----------
        strategy : CryptoScalping
            The strategy instance
        data : dict
            Dict mapping symbol to DataFrame with OHLCV data
        """
        print("\n" + "=" * 80)
        print("RUNNING CRYPTO SCALPING BACKTEST")
        print("=" * 80)

        # Merge all data and sort by timestamp
        all_bars = []
        for symbol, df in data.items():
            for _, row in df.iterrows():
                bar = Bar(
                    symbol=symbol,
                    timestamp=row['timestamp'].to_pydatetime() if hasattr(row['timestamp'], 'to_pydatetime') else row['timestamp'],
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume']),
                )
                all_bars.append(bar)

        # Sort by timestamp
        all_bars.sort(key=lambda x: x.timestamp)
        print(f"Total bars to process: {len(all_bars):,}")

        # Set date range
        if all_bars:
            self.start_date = all_bars[0].timestamp
            self.end_date = all_bars[-1].timestamp

        # Initialize strategy
        strategy.on_start()

        # Process bars
        processed = 0
        for bar in all_bars:
            symbol = bar.symbol

            # FIRST: Check exit conditions BEFORE strategy.on_bar() updates state
            if symbol in strategy.symbol_states and symbol in self.positions:
                state = strategy.symbol_states[symbol]
                pos = self.positions[symbol]

                # Update highest price for trailing stop
                if state.highest_price_since_entry is not None:
                    state.highest_price_since_entry = max(state.highest_price_since_entry, bar.close)

                should_exit, exit_reason = self._check_exit_conditions(
                    state, bar, pos, strategy.config
                )

                if should_exit:
                    # Execute exit
                    proceeds = pos['qty'] * bar.close * (1 - self.commission_pct)
                    pnl = proceeds - (pos['qty'] * pos['entry_price'])

                    self.capital += proceeds
                    self.trades.append({
                        'symbol': symbol,
                        'entry_time': pos['entry_time'],
                        'exit_time': bar.timestamp,
                        'entry_price': pos['entry_price'],
                        'exit_price': bar.close,
                        'qty': pos['qty'],
                        'pnl': pnl,
                        'pnl_pct': (bar.close - pos['entry_price']) / pos['entry_price'] * 100,
                        'exit_reason': exit_reason,
                    })

                    del self.positions[symbol]
                    state.reset_position_state()

            # SECOND: Update strategy (will generate new entry signals)
            strategy.on_bar(bar)

            # THIRD: Check for entry signals
            if symbol in strategy.symbol_states:
                state = strategy.symbol_states[symbol]

                # Process entry signal (only if no position exists)
                if state.pending_entry_order_id and symbol not in self.positions:
                    # Execute entry
                    qty = strategy.config.fixed_position_value / bar.close
                    cost = qty * bar.close * (1 + self.commission_pct)

                    if cost <= self.capital:
                        self.capital -= cost
                        self.positions[symbol] = {
                            'qty': qty,
                            'entry_price': bar.close,
                            'entry_time': bar.timestamp,
                        }
                        state.entry_price = bar.close
                        state.entry_time = bar.timestamp
                        state.highest_price_since_entry = bar.close
                        state.position = type('Position', (), {'quantity': qty, 'is_flat': False})()
                        state.pending_entry_order_id = None

            # Track equity
            if processed % 10000 == 0:
                equity = self.capital + sum(
                    p['qty'] * bar.close for p in self.positions.values()
                )
                self.equity_curve.append({
                    'timestamp': bar.timestamp,
                    'equity': equity,
                })

            processed += 1
            if processed % 50000 == 0:
                print(f"   Processed {processed:,} / {len(all_bars):,} bars...")

        # Close remaining positions
        for symbol, pos in list(self.positions.items()):
            last_bar = [b for b in all_bars if b.symbol == symbol][-1]
            proceeds = pos['qty'] * last_bar.close * (1 - self.commission_pct)
            pnl = proceeds - (pos['qty'] * pos['entry_price'])

            self.capital += proceeds
            self.trades.append({
                'symbol': symbol,
                'entry_time': pos['entry_time'],
                'exit_time': last_bar.timestamp,
                'entry_price': pos['entry_price'],
                'exit_price': last_bar.close,
                'qty': pos['qty'],
                'pnl': pnl,
                'pnl_pct': (last_bar.close - pos['entry_price']) / pos['entry_price'] * 100,
                'exit_reason': 'End of backtest',
            })

        strategy.on_stop()

        return self._generate_report()

    def _check_exit_conditions(self, state, bar, pos, config) -> tuple:
        """Check exit conditions for position."""
        price = bar.close
        entry_price = pos['entry_price']
        pnl_pct = (price - entry_price) / entry_price * 100

        # Calculate hold time in minutes
        entry_time = pos['entry_time']
        hold_minutes = (bar.timestamp - entry_time).total_seconds() / 60

        # Update highest price
        if state.highest_price_since_entry is not None:
            state.highest_price_since_entry = max(state.highest_price_since_entry, price)

        # 1. STOP LOSS - Always check first (no hold time requirement)
        if pnl_pct <= -config.stop_loss_pct:
            return True, 'Stop Loss'

        # 2. TAKE PROFIT (no hold time requirement)
        if pnl_pct >= config.target_profit_pct:
            return True, 'Take Profit'

        # 3. TRAILING STOP (only if in sufficient profit)
        if config.use_trailing_stop and state.highest_price_since_entry is not None:
            current_profit = (state.highest_price_since_entry - entry_price) / entry_price * 100
            # Only activate trailing stop after reaching activation threshold
            if current_profit >= config.trailing_stop_activation:
                trailing_stop_price = state.highest_price_since_entry * (1 - config.trailing_stop_pct / 100)
                if price <= trailing_stop_price:
                    return True, 'Trailing Stop'

        # === TECHNICAL EXITS - REQUIRE MINIMUM HOLD TIME AND PROFIT ===
        # Get config values with defaults for backward compatibility
        min_hold = getattr(config, 'min_hold_minutes', 10)
        min_profit = getattr(config, 'min_profit_for_technical_exit', 0.4)

        can_technical_exit = (
            hold_minutes >= min_hold and
            pnl_pct >= min_profit
        )

        if not can_technical_exit:
            return False, None  # Only SL/TP/Trailing can exit early

        # 4. RSI OVERBOUGHT (requires min hold time and min profit)
        if state.rsi.initialized and state.rsi.value >= config.rsi_overbought:
            return True, 'RSI Overbought'

        # 5. UPPER BB (requires min hold and good profit)
        if state.bb.initialized and price >= state.bb.upper and pnl_pct >= 0.8:
            return True, 'Upper BB'

        return False, None

    def _generate_report(self) -> CryptoBacktestResults:
        """Generate backtest report as CryptoBacktestResults object."""
        if not self.trades:
            return CryptoBacktestResults(
                start_date=self.start_date or datetime.now(pytz.UTC),
                end_date=self.end_date or datetime.now(pytz.UTC),
                duration_days=0,
                initial_capital=self.initial_capital,
                final_equity=self.capital,
                total_pnl=0,
                total_return_pct=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                trades=[],
                equity_curve=self.equity_curve,
            )

        df = pd.DataFrame(self.trades)
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]

        total_pnl = df['pnl'].sum()
        final_capital = self.capital

        # Calculate max drawdown
        max_dd = 0
        max_dd_pct = 0
        if self.equity_curve:
            equity_series = [e['equity'] for e in self.equity_curve]
            peak = equity_series[0]
            for eq in equity_series:
                if eq > peak:
                    peak = eq
                dd = peak - eq
                dd_pct = (dd / peak) * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                    max_dd_pct = dd_pct

        # Calculate Sharpe/Sortino (simplified)
        if len(df) > 1:
            returns = df['pnl_pct'].values / 100
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            neg_std = np.std(returns[returns < 0]) if len(returns[returns < 0]) > 0 else std_ret
            sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0  # Annualized
            sortino = (mean_ret / neg_std) * np.sqrt(252) if neg_std > 0 else 0
        else:
            sharpe = 0
            sortino = 0

        # Calculate duration
        duration_days = 0
        if self.start_date and self.end_date:
            duration_days = (self.end_date - self.start_date).days

        return CryptoBacktestResults(
            start_date=self.start_date or datetime.now(pytz.UTC),
            end_date=self.end_date or datetime.now(pytz.UTC),
            duration_days=duration_days,
            initial_capital=self.initial_capital,
            final_equity=final_capital,
            total_pnl=total_pnl,
            total_return_pct=(final_capital - self.initial_capital) / self.initial_capital * 100,
            total_trades=len(df),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / len(df) * 100 if len(df) > 0 else 0,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            trades=self.trades,
            equity_curve=self.equity_curve,
        )


def print_report(results: CryptoBacktestResults, symbols: list, data_sources: dict = None):
    """Print formatted backtest report using PerformanceAnalyzer."""
    print("\n" + "=" * 100)
    print("CRYPTO SCALPING BACKTEST RESULTS")
    print("=" * 100)

    if results.total_trades == 0:
        print("\nNo trades were executed during the backtest period.")
        print("This could be due to:")
        print("  - Entry conditions not being met")
        print("  - Insufficient data for indicator initialization")
        print("  - Risk controls preventing trades")
        return

    # Create DataSourceInfo
    source_info = DataSourceInfo(
        underlying_source="REAL (Yahoo Finance 1h)" if data_sources else "REAL",
        options_source="N/A (Spot Crypto)",
        underlying_bars=sum(len(df) for df in data_sources.values()) if data_sources else 0,
        options_bars=0,
    )

    # Use PerformanceAnalyzer for detailed report
    analyzer = PerformanceAnalyzer(results, "Crypto Scalping Strategy", source_info)

    # Print custom crypto-focused summary
    print(f"\n{'DATA QUALITY':=^100}")
    print(f"  *** REAL MARKET DATA - Results reflect actual market conditions ***")
    print(f"  Data Source: Yahoo Finance (1-hour bars)")
    if data_sources:
        total_bars = sum(len(df) for df in data_sources.values())
        print(f"  Total Bars:  {total_bars:,}")
    print(f"  Symbols:     {', '.join(symbols)}")

    print(f"\n{'STRATEGY INFO':=^100}")
    print(f"  Period:            {results.start_date} to {results.end_date}")
    print(f"  Duration:          {results.duration_days} days")

    print(f"\n{'CAPITAL & RETURNS':=^100}")
    print(f"  Initial Capital:   ${results.initial_capital:,.2f}")
    print(f"  Final Equity:      ${results.final_equity:,.2f}")
    print(f"  Total P&L:         ${results.total_pnl:,.2f}")
    print(f"  Total Return:      {results.total_return_pct:+.2f}%")

    print(f"\n{'TRADE STATISTICS':=^100}")
    print(f"  Total Trades:      {results.total_trades}")
    print(f"  Winning Trades:    {results.winning_trades} ({results.win_rate:.1f}%)")
    print(f"  Losing Trades:     {results.losing_trades} ({100-results.win_rate:.1f}%)")

    # Calculate additional metrics from trades
    if results.trades:
        df = pd.DataFrame(results.trades)
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]

        gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        print(f"\n{'P&L ANALYSIS':=^100}")
        print(f"  Gross Profit:      ${gross_profit:,.2f}")
        print(f"  Gross Loss:        ${gross_loss:,.2f}")
        pf_str = f"{profit_factor:.2f}" if profit_factor != float('inf') else "Infinite"
        print(f"  Profit Factor:     {pf_str}")
        print(f"  Average Win:       ${wins['pnl'].mean() if len(wins) > 0 else 0:,.2f}")
        print(f"  Average Loss:      ${losses['pnl'].mean() if len(losses) > 0 else 0:,.2f}")
        print(f"  Best Trade:        ${df['pnl'].max():,.2f}")
        print(f"  Worst Trade:       ${df['pnl'].min():,.2f}")

        print(f"\n{'P&L BY SYMBOL':=^100}")
        for symbol, pnl in df.groupby('symbol')['pnl'].sum().items():
            print(f"  {symbol:<15} ${pnl:>15,.2f}")

        print(f"\n{'TRADES BY EXIT REASON':=^100}")
        for reason, count in df.groupby('exit_reason')['pnl'].count().items():
            print(f"  {reason:<20} {count:>10}")

    print(f"\n{'RISK METRICS':=^100}")
    print(f"  Max Drawdown:      ${results.max_drawdown:,.2f} ({results.max_drawdown_pct:.2f}%)")
    print(f"  Sharpe Ratio:      {results.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:     {results.sortino_ratio:.2f}")

    print("\n" + "=" * 100)


async def run_backtest(args):
    """Run the crypto backtest."""
    print("\n" + "=" * 80)
    print("THEVOLUMEAI CRYPTO SCALPING - BACKTEST RUNNER")
    print("=" * 80)

    # Determine symbols to test
    if args.all_symbols:
        symbols = ALPACA_CRYPTO_SYMBOLS
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = ['BTC/USD', 'ETH/USD']  # Default

    print(f"Symbols:          {', '.join(symbols)}")
    print(f"Period:           {args.start} to {args.end}")
    print(f"Initial Capital:  ${args.capital:,.2f}")
    print(f"Position Size:    ${args.position_size:,.2f}")
    print(f"TP/SL:            +{args.take_profit}% / -{args.stop_loss}%")
    print("=" * 80)

    # Fetch data for each symbol
    print("\nFetching market data...")
    data = {}
    for symbol in symbols:
        df = await fetch_crypto_data(symbol, args.start, args.end)
        if df is not None and len(df) > 0:
            data[symbol] = df
        else:
            print(f"   WARNING: No data for {symbol}")

    if not data:
        print("\nERROR: No data available for any symbol. Cannot run backtest.")
        return None

    # Create strategy config
    config = CryptoScalpingConfig(
        symbols=list(data.keys()),
        fixed_position_value=args.position_size,
        target_profit_pct=args.take_profit,
        stop_loss_pct=args.stop_loss,
        trailing_stop_pct=args.trailing_stop,
        use_trailing_stop=not args.no_trailing_stop,
        max_trades_per_day=args.max_trades,
        max_concurrent_positions=args.max_positions,
    )

    # Create strategy
    strategy = CryptoScalping(config)

    # Create and run backtest engine
    engine = SimpleCryptoBacktestEngine(
        initial_capital=args.capital,
        commission_pct=0.001  # 0.1% commission
    )

    results = engine.run(strategy, data)

    # Print report with analytics
    print_report(results, list(data.keys()), data)

    # Save results as JSON
    if args.output:
        import json
        output_path = Path(args.output)

        # Convert results to dict for JSON serialization
        report_dict = {
            'start_date': str(results.start_date),
            'end_date': str(results.end_date),
            'duration_days': results.duration_days,
            'initial_capital': results.initial_capital,
            'final_equity': results.final_equity,
            'total_pnl': results.total_pnl,
            'total_return_pct': results.total_return_pct,
            'total_trades': results.total_trades,
            'winning_trades': results.winning_trades,
            'losing_trades': results.losing_trades,
            'win_rate': results.win_rate,
            'max_drawdown': results.max_drawdown,
            'max_drawdown_pct': results.max_drawdown_pct,
            'sharpe_ratio': results.sharpe_ratio,
            'sortino_ratio': results.sortino_ratio,
            'trades': results.trades,
            'symbols_tested': list(data.keys()),
            'data_source': 'REAL (Yahoo Finance 1h)',
        }

        # Add trade-level metrics
        if results.trades:
            df = pd.DataFrame(results.trades)
            wins = df[df['pnl'] > 0]
            losses = df[df['pnl'] <= 0]
            report_dict['avg_pnl'] = df['pnl'].mean()
            report_dict['avg_win'] = wins['pnl'].mean() if len(wins) > 0 else 0
            report_dict['avg_loss'] = losses['pnl'].mean() if len(losses) > 0 else 0
            report_dict['best_trade'] = df['pnl'].max()
            report_dict['worst_trade'] = df['pnl'].min()
            report_dict['profit_factor'] = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf')
            report_dict['trades_by_symbol'] = df.groupby('symbol')['pnl'].sum().to_dict()
            report_dict['trades_by_exit'] = df.groupby('exit_reason')['pnl'].count().to_dict()

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='TheVolumeAI Crypto Scalping Strategy - Backtest Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick backtest - November 2024 with BTC and ETH
  python -m trading_system.run_crypto_backtest --start 2024-11-01 --end 2024-11-30

  # Full backtest - 3 months with all symbols
  python -m trading_system.run_crypto_backtest --all-symbols --start 2024-09-01 --end 2024-11-30

  # Custom symbols
  python -m trading_system.run_crypto_backtest --symbols BTC/USD ETH/USD SOL/USD --start 2024-11-01 --end 2024-11-30
        """
    )

    # Required arguments
    parser.add_argument('--start', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                       help='End date (YYYY-MM-DD)')

    # Symbol selection
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Crypto symbols to test (e.g., BTC/USD ETH/USD)')
    parser.add_argument('--all-symbols', action='store_true',
                       help='Test all supported symbols')

    # Capital and position sizing
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Initial capital (default: 100000)')
    parser.add_argument('--position-size', type=float, default=500.0,
                       help='Position size per trade (default: 500)')

    # Risk parameters
    parser.add_argument('--take-profit', type=float, default=1.0,
                       help='Take profit percentage (default: 1.0)')
    parser.add_argument('--stop-loss', type=float, default=0.5,
                       help='Stop loss percentage (default: 0.5)')
    parser.add_argument('--trailing-stop', type=float, default=0.3,
                       help='Trailing stop percentage (default: 0.3)')
    parser.add_argument('--no-trailing-stop', action='store_true',
                       help='Disable trailing stop')

    # Risk controls
    parser.add_argument('--max-trades', type=int, default=500,
                       help='Max trades per day (default: 500 for scalping)')
    parser.add_argument('--max-positions', type=int, default=20,
                       help='Max concurrent positions (default: 20 to allow all symbols)')

    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (JSON)')

    args = parser.parse_args()

    # Run backtest
    asyncio.run(run_backtest(args))


if __name__ == '__main__':
    main()
