"""
Backtest Runner CLI

Usage:
    python -m trading_system.run_backtest --symbol COIN --start 2024-11-01 --end 2024-11-15

This script runs backtests using our own trading system (no NautilusTrader).
"""

import argparse
import asyncio
import io
import os
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from trading_system.core.models import (
    Bar, Instrument, OptionContract, InstrumentType, OptionType
)
from trading_system.engine.backtest_engine import BacktestEngine, BacktestConfig
from trading_system.strategies.coin_0dte_momentum import (
    COINDaily0DTEMomentum, COINDaily0DTEMomentumConfig
)
from trading_system.analytics.performance import PerformanceAnalyzer, DataSourceInfo


def create_option_contracts(
    underlying_symbol: str,
    df: pd.DataFrame,
    expiry_frequency: str = 'weekly'
) -> list:
    """
    Create option contracts covering the price range of each trading day.

    Only creates strikes within 5% of each day's price range to keep
    the number of contracts manageable while ensuring ATM options are available.

    Parameters
    ----------
    underlying_symbol : str
        Underlying symbol (e.g., 'COIN')
    df : pd.DataFrame
        Underlying data with OHLCV
    expiry_frequency : str
        'daily', 'weekly', or 'monthly'

    Returns
    -------
    list
        List of OptionContract objects
    """
    contracts = []
    created_symbols = set()  # Track unique contracts to avoid duplicates

    # Get unique dates
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    trading_days = sorted(df['date'].unique())

    # Calculate the FULL price range across the entire period
    min_price = float(df['low'].min())
    max_price = float(df['high'].max())

    print(f"   Price range: ${min_price:.2f} - ${max_price:.2f}")

    # For each trading day, create strikes around that day's price
    for day in trading_days:
        day_data = df[df['date'] == day]
        if len(day_data) == 0:
            continue

        # Get opening price for this day (used to determine relevant strikes)
        day_open = float(day_data['open'].iloc[0])
        day_high = float(day_data['high'].max())
        day_low = float(day_data['low'].min())

        # Create strikes within 5% of day's price range
        # This ensures ATM options are always available
        price_buffer = day_open * 0.05  # 5% buffer
        strike_low = int((day_low - price_buffer) / 5) * 5
        strike_high = int((day_high + price_buffer) / 5) * 5 + 5

        # Determine expiry for this day
        if expiry_frequency == 'daily':
            expiry_date = day
        elif expiry_frequency == 'weekly':
            # Find THIS WEEK's Friday - options expire same week at 4PM Friday
            # Monday (0) -> Friday in 4 days
            # Tuesday (1) -> Friday in 3 days
            # Wednesday (2) -> Friday in 2 days
            # Thursday (3) -> Friday in 1 day
            # Friday (4) -> Friday TODAY (0DTE)
            weekday = day.weekday()
            if weekday <= 4:  # Monday to Friday
                days_to_friday = 4 - weekday  # 4=Friday weekday number
            else:  # Saturday/Sunday - skip to next Friday
                days_to_friday = (4 - weekday) % 7
            expiry_date = day + pd.Timedelta(days=days_to_friday)
            if hasattr(expiry_date, 'date'):
                expiry_date = expiry_date.date()
        else:
            # Monthly - third Friday
            import calendar
            cal = calendar.Calendar()
            fridays = [d for d in cal.itermonthdates(day.year, day.month)
                      if d.weekday() == 4 and d.month == day.month]
            expiry_date = fridays[2] if len(fridays) >= 3 else fridays[-1]

        expiry_dt = datetime.combine(expiry_date, datetime.min.time())
        expiry_str = expiry_dt.strftime('%y%m%d')

        # Create CALL and PUT contracts for strikes near this day's price
        for strike in range(strike_low, strike_high + 1, 5):
            for opt_type in [OptionType.CALL, OptionType.PUT]:
                type_char = 'C' if opt_type == OptionType.CALL else 'P'
                symbol = f"{underlying_symbol}{expiry_str}{type_char}{int(strike*1000):08d}"

                # Skip if already created
                if symbol in created_symbols:
                    continue
                created_symbols.add(symbol)

                contract = OptionContract(
                    symbol=symbol,
                    instrument_type=InstrumentType.OPTION,
                    currency="USD",
                    multiplier=100,
                    tick_size=0.01,
                    exchange="CBOE",
                    underlying_symbol=underlying_symbol,
                    option_type=opt_type,
                    strike_price=float(strike),
                    expiration=expiry_dt,
                )
                contracts.append(contract)

    print(f"   Created {len(contracts)} unique option contracts")
    return contracts


class RealisticOptionsModel:
    """
    Realistic options pricing model with:
    - IV smile/skew (OTM puts have higher IV)
    - Intraday IV patterns (higher at open, lower midday)
    - 0DTE accelerated time decay
    - Bid-ask spread based on moneyness and time
    - Price jitter/noise for realistic fills
    - Greeks calculation for P&L attribution
    """

    def __init__(self, base_iv: float = 0.80, underlying_symbol: str = "COIN"):
        """
        Initialize the realistic options model.

        Parameters
        ----------
        base_iv : float
            Base implied volatility (ATM level), default 80% for COIN
        underlying_symbol : str
            Symbol for adjusting IV characteristics
        """
        self.base_iv = base_iv
        self.underlying_symbol = underlying_symbol

        # COIN-specific IV parameters (high-vol crypto stock)
        self.iv_params = {
            'COIN': {'base': 0.85, 'skew': 0.15, 'smile': 0.08, 'term_slope': 0.02},
            'SPY':  {'base': 0.18, 'skew': 0.05, 'smile': 0.02, 'term_slope': 0.01},
            'QQQ':  {'base': 0.22, 'skew': 0.06, 'smile': 0.03, 'term_slope': 0.01},
            'DEFAULT': {'base': 0.40, 'skew': 0.10, 'smile': 0.05, 'term_slope': 0.015},
        }

        # Get parameters for this symbol
        params = self.iv_params.get(underlying_symbol, self.iv_params['DEFAULT'])
        self.base_iv = params['base']
        self.skew_factor = params['skew']  # How much higher IV is for OTM puts
        self.smile_factor = params['smile']  # How much IV increases for far OTM
        self.term_slope = params['term_slope']  # IV term structure slope

    def calculate_iv(
        self,
        S: float,
        K: float,
        T: float,
        is_call: bool,
        hour_of_day: int = 10
    ) -> float:
        """
        Calculate implied volatility with smile, skew, and intraday patterns.

        Parameters
        ----------
        S : float
            Spot price
        K : float
            Strike price
        T : float
            Time to expiry in years
        is_call : bool
            True for calls, False for puts
        hour_of_day : int
            Hour in EST (9-16)

        Returns
        -------
        float
            Adjusted implied volatility
        """
        import numpy as np

        # 1. Calculate moneyness (log-moneyness)
        moneyness = np.log(K / S)  # Positive = OTM call / ITM put

        # 2. IV Skew: OTM puts have higher IV (fear premium)
        # For puts: negative moneyness = OTM, should have higher IV
        # For calls: positive moneyness = OTM, slightly higher IV
        if is_call:
            skew_adjustment = self.skew_factor * max(0, moneyness) * 0.5  # Smaller effect for calls
        else:
            skew_adjustment = self.skew_factor * max(0, -moneyness)  # Full effect for puts

        # 3. IV Smile: Far OTM options (both calls and puts) have higher IV
        abs_moneyness = abs(moneyness)
        smile_adjustment = self.smile_factor * (abs_moneyness ** 2)

        # 4. Term Structure: Short-dated options have higher IV (especially 0DTE)
        # IV increases as T decreases (inverted term structure for short dates)
        days_to_expiry = T * 365
        if days_to_expiry <= 1:
            # 0DTE: IV spikes significantly
            term_adjustment = 0.15  # +15% IV for 0DTE
        elif days_to_expiry <= 7:
            # Weekly: moderate increase
            term_adjustment = 0.05 * (7 - days_to_expiry) / 6
        else:
            term_adjustment = 0.0

        # 5. Intraday IV Pattern (EST hours)
        # Higher IV at open (9:30) and close (3:30-4:00), lower midday
        intraday_pattern = {
            9: 0.08,   # Pre-market/open spike
            10: 0.05,  # Still elevated
            11: 0.02,  # Settling
            12: 0.00,  # Lunch lull
            13: 0.00,  # Quiet
            14: 0.02,  # Afternoon pickup
            15: 0.05,  # Close approach
            16: 0.08,  # Close spike
        }
        intraday_adjustment = intraday_pattern.get(hour_of_day, 0.02)

        # 6. Combine all adjustments
        final_iv = self.base_iv + skew_adjustment + smile_adjustment + term_adjustment + intraday_adjustment

        # Clamp to reasonable bounds
        return max(0.10, min(3.0, final_iv))

    def calculate_bid_ask_spread(
        self,
        S: float,
        K: float,
        option_price: float,
        T: float,
        volume_ratio: float = 1.0
    ) -> tuple:
        """
        Calculate realistic bid-ask spread.

        Spread is wider for:
        - OTM options (less liquid)
        - Near expiry (gamma risk)
        - Low volume periods
        - Lower-priced options (minimum tick impact)

        Returns
        -------
        tuple
            (bid_price, ask_price, spread_pct)
        """
        import numpy as np

        # Base spread as percentage (market makers need edge)
        base_spread_pct = 0.03  # 3% base spread

        # 1. Moneyness adjustment: OTM options have wider spreads
        moneyness = abs(np.log(K / S))
        if moneyness > 0.10:  # More than 10% OTM
            moneyness_mult = 1.0 + moneyness * 2
        else:
            moneyness_mult = 1.0

        # 2. Time to expiry: 0DTE has wider spreads (gamma risk)
        days_to_expiry = T * 365
        if days_to_expiry <= 1:
            time_mult = 2.0  # 2x spread for 0DTE
        elif days_to_expiry <= 7:
            time_mult = 1.5
        else:
            time_mult = 1.0

        # 3. Price-based: Cheap options have proportionally wider spreads
        # Minimum $0.05 spread for options under $1
        if option_price < 1.0:
            min_spread = 0.05
        elif option_price < 5.0:
            min_spread = 0.05
        else:
            min_spread = 0.10

        # 4. Volume adjustment: Lower volume = wider spread
        volume_mult = 1.0 / max(0.5, min(2.0, volume_ratio))

        # Calculate final spread
        spread_pct = base_spread_pct * moneyness_mult * time_mult * volume_mult
        spread_pct = max(spread_pct, 0.02)  # Minimum 2% spread
        spread_pct = min(spread_pct, 0.30)  # Maximum 30% spread

        half_spread = option_price * spread_pct / 2
        half_spread = max(half_spread, min_spread / 2)

        bid_price = max(0.01, option_price - half_spread)
        ask_price = option_price + half_spread

        actual_spread_pct = (ask_price - bid_price) / option_price * 100

        return bid_price, ask_price, actual_spread_pct

    def black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        is_call: bool
    ) -> float:
        """Calculate Black-Scholes option price."""
        from scipy.stats import norm
        import numpy as np

        if T <= 0:
            # At expiry - intrinsic value only
            if is_call:
                return max(0.01, S - K)
            else:
                return max(0.01, K - S)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if is_call:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(0.01, price)

    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        is_call: bool
    ) -> dict:
        """Calculate option Greeks for analysis."""
        from scipy.stats import norm
        import numpy as np

        if T <= 0.001:
            # At expiry
            intrinsic = max(0, S - K) if is_call else max(0, K - S)
            return {
                'delta': 1.0 if (is_call and S > K) or (not is_call and S < K) else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'iv': sigma,
            }

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Delta
        if is_call:
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * sqrt_T)

        # Theta (per day)
        theta_term1 = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
        if is_call:
            theta_term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta_term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = (theta_term1 + theta_term2) / 365  # Per day

        # Vega (per 1% IV change)
        vega = S * sqrt_T * norm.pdf(d1) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'iv': sigma,
        }

    def add_price_noise(
        self,
        price: float,
        T: float,
        volume: int
    ) -> float:
        """Add realistic price noise/jitter."""
        import numpy as np

        # More noise for 0DTE (gamma creates volatility)
        days_to_expiry = T * 365
        if days_to_expiry <= 1:
            noise_pct = 0.02  # 2% noise for 0DTE
        else:
            noise_pct = 0.01  # 1% noise otherwise

        # Less noise for high volume
        if volume > 10000:
            noise_pct *= 0.5

        noise = np.random.normal(0, noise_pct * price)
        return max(0.01, price + noise)


def generate_synthetic_options_bars(
    underlying_df: pd.DataFrame,
    contract: OptionContract
) -> list:
    """
    Generate realistic synthetic options bars from underlying data.

    OPTIMIZED: Uses vectorized numpy operations for ~100x speedup.

    Features:
    - IV smile/skew (OTM puts have higher IV)
    - Intraday IV patterns (higher at open/close)
    - 0DTE accelerated time decay
    - REALISTIC price movement (gradual, not instant SL/TP triggers)
    - Options price follows underlying movement with appropriate leverage
    """
    import numpy as np
    from scipy.stats import norm
    import pytz

    # Get symbol-specific IV parameters
    iv_params = {
        'COIN': {'base': 0.85, 'skew': 0.15, 'smile': 0.08},
        'SPY':  {'base': 0.18, 'skew': 0.05, 'smile': 0.02},
        'QQQ':  {'base': 0.22, 'skew': 0.06, 'smile': 0.03},
    }
    params = iv_params.get(contract.underlying_symbol or "COIN", {'base': 0.40, 'skew': 0.10, 'smile': 0.05})
    base_iv = params['base']
    skew_factor = params['skew']
    smile_factor = params['smile']

    # Extract arrays for vectorized operations
    n = len(underlying_df)
    S_close = underlying_df['close'].values.astype(float)
    S_open = underlying_df['open'].values.astype(float)
    S_high = underlying_df['high'].values.astype(float)
    S_low = underlying_df['low'].values.astype(float)
    volumes = underlying_df.get('volume', pd.Series([1000]*n)).values.astype(float)

    K = contract.strike_price
    is_call = contract.is_call
    r = 0.05  # Risk-free rate

    # Parse timestamps and calculate time to expiry
    expiry_dt = pd.Timestamp(contract.expiration, tz='UTC')
    est_tz = pytz.timezone('America/New_York')

    timestamps = pd.to_datetime(underlying_df['timestamp'])
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize('UTC')

    # Time to expiry (vectorized)
    T_seconds = (expiry_dt - timestamps).dt.total_seconds().values
    T = np.maximum(0.0001, T_seconds / (365.0 * 24 * 3600))

    # Hour of day for intraday IV pattern
    est_hours = timestamps.dt.tz_convert(est_tz).dt.hour.values

    # Intraday IV adjustment (vectorized lookup) - REDUCED for stability
    intraday_adj = np.zeros(n)
    intraday_adj[est_hours == 9] = 0.04   # Reduced from 0.08
    intraday_adj[est_hours == 10] = 0.02  # Reduced from 0.05
    intraday_adj[est_hours == 11] = 0.01  # Reduced from 0.02
    intraday_adj[(est_hours == 12) | (est_hours == 13)] = 0.0
    intraday_adj[est_hours == 14] = 0.01  # Reduced from 0.02
    intraday_adj[est_hours == 15] = 0.02  # Reduced from 0.05
    intraday_adj[est_hours == 16] = 0.04  # Reduced from 0.08

    # 0DTE IV boost - REDUCED for more realistic movement
    days_to_expiry = T * 365
    term_adj = np.where(days_to_expiry <= 1, 0.08, np.where(days_to_expiry <= 7, 0.03 * (7 - days_to_expiry) / 6, 0.0))

    # Moneyness
    moneyness = np.log(K / S_close)

    # IV skew (puts get more IV when OTM)
    if is_call:
        skew_adj = skew_factor * np.maximum(0, moneyness) * 0.5
    else:
        skew_adj = skew_factor * np.maximum(0, -moneyness)

    # IV smile
    smile_adj = smile_factor * (np.abs(moneyness) ** 2)

    # Final IV
    sigma = base_iv + skew_adj + smile_adj + term_adj + intraday_adj
    sigma = np.clip(sigma, 0.10, 2.0)  # Reduced max from 3.0 to 2.0

    # Black-Scholes pricing (vectorized)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S_close / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if is_call:
        prices = S_close * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        prices = K * np.exp(-r * T) * norm.cdf(-d2) - S_close * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1

    prices = np.maximum(0.01, prices)
    delta_abs = np.abs(delta)

    # ========================================================================
    # REALISTIC PRICE MOVEMENT - Key Fix
    # ========================================================================
    # Instead of random noise, options price should follow underlying movement
    # with appropriate leverage (delta). This creates gradual price development.

    # Calculate underlying percentage move per bar
    underlying_pct_change = np.zeros(n)
    underlying_pct_change[1:] = (S_close[1:] - S_close[:-1]) / S_close[:-1]

    # Options move based on delta (simplified leverage)
    # For ATM options (~0.50 delta), a 1% underlying move = ~1% option move
    # But options have leverage, so amplify by 1/delta (capped)
    leverage = np.clip(1.0 / np.maximum(delta_abs, 0.1), 1.0, 10.0)

    # Option percentage change follows underlying with leverage
    # Apply direction: calls gain when underlying rises, puts gain when it falls
    if is_call:
        option_pct_change = underlying_pct_change * leverage * delta_abs
    else:
        option_pct_change = -underlying_pct_change * leverage * delta_abs

    # Add small random noise (very small, just for realism)
    # This is 0.2% noise, not 2% - options don't jump randomly
    noise = np.random.normal(0, 0.002, n)
    option_pct_change = option_pct_change + noise

    # Build close prices by accumulating changes from theoretical price
    close_prices = np.zeros(n)
    close_prices[0] = prices[0]
    for i in range(1, n):
        # Start from theoretical price but apply accumulated movement
        # This ensures prices track Black-Scholes but with realistic intrabar movement
        theoretical = prices[i]
        previous_close = close_prices[i-1]

        # Move from previous close based on underlying movement
        moved_price = previous_close * (1 + option_pct_change[i])

        # Blend with theoretical to prevent drift (80% movement, 20% theoretical anchor)
        close_prices[i] = 0.8 * moved_price + 0.2 * theoretical

    close_prices = np.maximum(0.01, close_prices)

    # ========================================================================
    # REALISTIC OHLC - Based on underlying's intrabar movement
    # ========================================================================

    # Calculate underlying intrabar range as percentage
    underlying_bar_range_pct = (S_high - S_low) / S_close

    # Options intrabar range scales with delta and underlying range
    # But cap it to prevent extreme swings
    option_bar_range_pct = np.clip(underlying_bar_range_pct * leverage * delta_abs, 0.001, 0.03)

    # Open prices: start from previous close, adjusted for gap
    open_prices = np.zeros(n)
    open_prices[0] = close_prices[0] * 0.998  # Small gap on first bar
    for i in range(1, n):
        # Gap from previous close based on underlying gap
        underlying_gap = (S_open[i] - S_close[i-1]) / S_close[i-1]
        if is_call:
            gap_effect = underlying_gap * delta_abs[i]
        else:
            gap_effect = -underlying_gap * delta_abs[i]
        open_prices[i] = close_prices[i-1] * (1 + gap_effect)
    open_prices = np.maximum(0.01, open_prices)

    # High and low based on direction of bar
    bar_direction = np.sign(close_prices - open_prices)  # 1 if up bar, -1 if down
    half_range = close_prices * option_bar_range_pct * 0.5

    # For up bars: high is above close, low is below open
    # For down bars: high is above open, low is below close
    high_prices = np.maximum(open_prices, close_prices) + half_range * 0.3  # Reduced extension
    low_prices = np.minimum(open_prices, close_prices) - half_range * 0.3   # Reduced extension

    # Ensure high >= max(open, close) and low <= min(open, close)
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    low_prices = np.maximum(0.01, low_prices)

    # Option volume
    option_volumes = np.maximum(1, (volumes * 0.01 * np.random.uniform(0.5, 1.5, n)).astype(int))

    # Build bars
    bars = []
    for i in range(n):
        ts = timestamps.iloc[i]
        if hasattr(ts, 'to_pydatetime'):
            ts = ts.to_pydatetime()

        bar = Bar(
            symbol=contract.symbol,
            timestamp=ts,
            open=round(open_prices[i], 2),
            high=round(high_prices[i], 2),
            low=round(low_prices[i], 2),
            close=round(close_prices[i], 2),
            volume=int(option_volumes[i]),
        )
        bars.append(bar)

    return bars


async def fetch_underlying_data(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = '1Min',
    data_info: DataSourceInfo = None
) -> pd.DataFrame:
    """
    Fetch underlying data using available providers.

    Falls back to: Polygon -> Yahoo Finance -> Synthetic
    Returns DataFrame and updates data_info with source information.
    """
    print(f"\n📊 Fetching data for {symbol} ({start_date} to {end_date})")

    # Try Polygon first
    polygon_key = os.getenv('POLYGON_API_KEY')
    if polygon_key:
        try:
            print("   Trying Polygon API...")
            import requests

            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start_date}/{end_date}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apiKey': polygon_key
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == 'OK' and data.get('results'):
                df = pd.DataFrame([{
                    'timestamp': pd.Timestamp(r['t'], unit='ms', tz='UTC'),
                    'open': float(r['o']),
                    'high': float(r['h']),
                    'low': float(r['l']),
                    'close': float(r['c']),
                    'volume': int(r.get('v', 0))
                } for r in data['results']])
                print(f"   ✓ Polygon: Fetched {len(df):,} bars")
                if data_info:
                    data_info.underlying_source = "REAL (Polygon.io API)"
                    data_info.underlying_bars = len(df)
                return df

        except Exception as e:
            print(f"   ⚠️ Polygon failed: {e}")

    # Try Yahoo Finance
    try:
        print("   Trying Yahoo Finance...")
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date, interval='1m')

        if len(hist) > 0:
            df = pd.DataFrame({
                'timestamp': hist.index.tz_localize('UTC') if hist.index.tz is None else hist.index.tz_convert('UTC'),
                'open': hist['Open'],
                'high': hist['High'],
                'low': hist['Low'],
                'close': hist['Close'],
                'volume': hist['Volume'].astype(int),
            }).reset_index(drop=True)
            print(f"   ✓ Yahoo: Fetched {len(df):,} bars")
            if data_info:
                data_info.underlying_source = "REAL (Yahoo Finance API)"
                data_info.underlying_bars = len(df)
            return df

    except Exception as e:
        print(f"   ⚠️ Yahoo failed: {e}")

    # Generate synthetic data as last resort
    print("   ⚠️ Generating SYNTHETIC data (no real data available)...")
    dates = pd.date_range(start=start_date, end=end_date, freq='1T')
    # Filter to market hours (9:30 - 16:00 ET)
    dates = dates[(dates.hour >= 9) & (dates.hour < 16)]

    import numpy as np
    base_price = 200.0  # COIN approximate price
    prices = base_price + np.cumsum(np.random.randn(len(dates)) * 0.5)
    prices = np.maximum(prices, 50)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(len(dates)) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(len(dates))) * 0.005),
        'low': prices * (1 - np.abs(np.random.randn(len(dates))) * 0.005),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates)),
    })
    print(f"   ✓ Generated {len(df):,} synthetic bars")
    if data_info:
        data_info.underlying_source = "SYNTHETIC (Generated)"
        data_info.underlying_bars = len(df)
    return df


async def run_backtest(args):
    """Run the backtest"""
    print("\n" + "=" * 80)
    print("🚀 THEVOLUMEAI TRADING SYSTEM - BACKTEST RUNNER")
    print("=" * 80)
    print(f"Symbol:           {args.symbol}")
    print(f"Period:           {args.start} to {args.end}")
    print(f"Initial Capital:  ${args.capital:,.2f}")
    print(f"Position Size:    ${args.position_size:,.2f}")
    print("=" * 80 + "\n")

    # Track data source for reporting
    data_info = DataSourceInfo()

    # Fetch underlying data
    underlying_df = await fetch_underlying_data(args.symbol, args.start, args.end, data_info=data_info)

    if underlying_df is None or len(underlying_df) == 0:
        print("❌ Failed to fetch data. Exiting.")
        return

    print(f"\n📈 Underlying data: {len(underlying_df):,} bars")
    print(f"   First: {underlying_df['timestamp'].iloc[0]}")
    print(f"   Last:  {underlying_df['timestamp'].iloc[-1]}")

    # Create backtest engine
    config = BacktestConfig(
        initial_capital=args.capital,
        commission_per_contract=0.65,
        slippage_pct=0.1,
    )
    engine = BacktestEngine(config)

    # Add underlying instrument
    underlying_inst = Instrument(
        symbol=args.symbol,
        instrument_type=InstrumentType.STOCK,
        currency="USD",
        exchange="NASDAQ",
    )
    engine.add_instrument(underlying_inst)

    # Convert underlying data to bars
    underlying_bars = []
    for _, row in underlying_df.iterrows():
        bar = Bar(
            symbol=args.symbol,
            timestamp=row['timestamp'].to_pydatetime() if hasattr(row['timestamp'], 'to_pydatetime') else row['timestamp'],
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=int(row['volume']),
        )
        underlying_bars.append(bar)
    engine.add_data(underlying_bars)

    # Create option contracts
    print("\n📋 Creating option contracts...")
    contracts = create_option_contracts(args.symbol, underlying_df, args.expiry_frequency)
    print(f"   Created {len(contracts)} option contracts")

    # Add options and generate synthetic data
    print("\n📊 Generating options data...")
    total_options_bars = 0
    for contract in contracts:
        engine.add_instrument(contract)
        options_bars = generate_synthetic_options_bars(underlying_df, contract)
        engine.add_data(options_bars)
        total_options_bars += len(options_bars)
    data_info.options_bars = total_options_bars

    # Create strategy
    print("\n🎯 Configuring strategy...")
    strategy_config = COINDaily0DTEMomentumConfig(
        instrument_id=args.symbol,
        fixed_position_value=args.position_size,
        target_profit_pct=Decimal(str(args.take_profit)),
        stop_loss_pct=Decimal(str(args.stop_loss)),
        max_hold_minutes=args.max_hold,
        fast_ema_period=9,
        slow_ema_period=20,
    )
    strategy = COINDaily0DTEMomentum(strategy_config)
    engine.add_strategy(strategy)

    # Run backtest
    results = engine.run()

    # Generate comprehensive performance report
    print("\n" + "=" * 100)
    print("GENERATING COMPREHENSIVE PERFORMANCE ANALYSIS...")
    print("=" * 100)

    analyzer = PerformanceAnalyzer(results, strategy_name="COIN Daily 0DTE Momentum", data_source_info=data_info)
    report = analyzer.analyze()
    report_text = analyzer.print_report(report)
    print(report_text)

    # Save results
    if args.output:
        output_path = Path(args.output)
        import json
        with open(output_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_path}")

        # Also save the text report
        report_path = output_path.with_suffix('.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"💾 Report saved to: {report_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='TheVolumeAI Trading System - Backtest Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m trading_system.run_backtest --symbol COIN --start 2024-11-01 --end 2024-11-15
  python -m trading_system.run_backtest --symbol COIN --start 2024-10-01 --end 2024-11-01 --capital 50000
        """
    )

    parser.add_argument('--symbol', type=str, default='COIN',
                       help='Underlying symbol (default: COIN)')
    parser.add_argument('--start', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Initial capital (default: 100000)')
    parser.add_argument('--position-size', type=float, default=2000.0,
                       help='Position size per trade (default: 2000)')
    parser.add_argument('--take-profit', type=float, default=7.5,
                       help='Take profit percentage (default: 7.5)')
    parser.add_argument('--stop-loss', type=float, default=25.0,
                       help='Stop loss percentage (default: 25)')
    parser.add_argument('--max-hold', type=int, default=30,
                       help='Max hold time in minutes (default: 30)')
    parser.add_argument('--expiry-frequency', type=str, default='weekly',
                       choices=['daily', 'weekly', 'monthly'],
                       help='Option expiry frequency (default: weekly)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (JSON)')

    args = parser.parse_args()

    # Run backtest
    asyncio.run(run_backtest(args))


if __name__ == '__main__':
    main()
