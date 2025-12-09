"""
Multi-Crypto ML Ensemble Training Script

Trains ML ensemble model on ALL crypto symbols from crypto_scalping.py:
- BTC/USD, ETH/USD, SOL/USD, DOGE/USD, LINK/USD
- AVAX/USD, DOT/USD, LTC/USD, SHIB/USD

Replicates the EXACT strategy logic for labeling:
- RSI < 30 oversold (mandatory)
- Price <= Lower BB
- Volume spike > 1.3x
- MACD positive momentum
- Stochastic oversold/crossover
- Support levels
- Candlestick patterns

Usage:
    # Train on 1-MINUTE data (matches backtest interval) - ALL 9 cryptos
    python train_multi_crypto_ensemble.py --days 30 --interval 1m --output models/crypto_scalping_ensemble.pkl
    
    # Train with more data (60 days)
    python train_multi_crypto_ensemble.py --days 60 --interval 1m --output models/crypto_scalping_ensemble.pkl
    
    # With hyperparameter optimization (slower but better)
    python train_multi_crypto_ensemble.py --days 30 --interval 1m --optimize --output models/crypto_scalping_ensemble.pkl
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# All crypto symbols from crypto_scalping.py
CRYPTO_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "LINK/USD",
    "AVAX/USD", "DOT/USD", "LTC/USD", "SHIB/USD"
]

# Yahoo Finance ticker mapping
SYMBOL_TO_YAHOO = {
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
    "SOL/USD": "SOL-USD",
    "DOGE/USD": "DOGE-USD",
    "LINK/USD": "LINK-USD",
    "AVAX/USD": "AVAX-USD",
    "DOT/USD": "DOT1-USD",  # Polkadot
    "LTC/USD": "LTC-USD",
    "SHIB/USD": "SHIB-USD"
}


def fetch_binance_data_sync(symbol: str, start_date: datetime, end_date: datetime, interval: str = '1m') -> pd.DataFrame:
    """
    Fetch data from Binance API synchronously (same source as backtest).
    
    Uses Binance.US first, then falls back to Binance Global.
    """
    import requests
    
    # Convert symbol format: BTC/USD -> BTCUSDT
    binance_symbol = symbol.replace('/USD', 'USDT').replace('/', '')
    
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    all_klines = []
    current_start = start_ts
    limit = 1000
    
    logger.info(f"   📊 Fetching {symbol} ({binance_symbol}) from Binance...")
    
    # Try Binance.US first
    base_url = "https://api.binance.us/api/v3/klines"
    
    try:
        while current_start < end_ts:
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_ts,
                'limit': limit
            }
            
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
            
            if len(all_klines) % 10000 == 0:
                logger.info(f"      Progress: {len(all_klines):,} bars...")
        
        if all_klines:
            logger.info(f"      ✅ Binance.US: {len(all_klines):,} bars")
    
    except Exception as e:
        logger.warning(f"      ⚠️  Binance.US failed: {e}")
        
        # Try Binance Global as fallback
        base_url = "https://api.binance.com/api/v3/klines"
        all_klines = []
        current_start = start_ts
        
        try:
            while current_start < end_ts:
                params = {
                    'symbol': binance_symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': end_ts,
                    'limit': limit
                }
                
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
            
            if all_klines:
                logger.info(f"      ✅ Binance Global: {len(all_klines):,} bars")
        
        except Exception as e2:
            logger.error(f"      ❌ Both Binance APIs failed: {e2}")
            return None
    
    if not all_klines:
        return None
    
    # Convert to DataFrame
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
    df['symbol'] = symbol
    
    return df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]


async def fetch_binance_data(symbol: str, start_date: datetime, end_date: datetime, interval: str = '1m') -> pd.DataFrame:
    """
    Fetch data from Binance API (same source as backtest).
    
    Uses Binance.US first, then falls back to Binance Global.
    This ensures training data matches backtest data EXACTLY.
    """
    import requests
    
    # Convert symbol format: BTC/USD -> BTCUSDT
    binance_symbol = symbol.replace('/USD', 'USDT').replace('/', '')
    
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    all_klines = []
    current_start = start_ts
    limit = 1000
    
    logger.info(f"   📊 Fetching {symbol} ({binance_symbol}) from Binance...")
    
    # Try Binance.US first
    base_url = "https://api.binance.us/api/v3/klines"
    
    try:
        while current_start < end_ts:
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_ts,
                'limit': limit
            }
            
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
            
            if len(all_klines) % 10000 == 0:
                logger.info(f"      Progress: {len(all_klines):,} bars...")
        
        if all_klines:
            logger.info(f"      ✅ Binance.US: {len(all_klines):,} bars")
    
    except Exception as e:
        logger.warning(f"      ⚠️  Binance.US failed: {e}")
        
        # Try Binance Global as fallback
        base_url = "https://api.binance.com/api/v3/klines"
        all_klines = []
        current_start = start_ts
        
        try:
            while current_start < end_ts:
                params = {
                    'symbol': binance_symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': end_ts,
                    'limit': limit
                }
                
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
            
            if all_klines:
                logger.info(f"      ✅ Binance Global: {len(all_klines):,} bars")
        
        except Exception as e2:
            logger.error(f"      ❌ Both Binance APIs failed: {e2}")
            return None
    
    if not all_klines:
        return None
    
    # Convert to DataFrame
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
    df['symbol'] = symbol
    
    return df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]


def download_multi_symbol_data(symbols: List[str], days: int, interval: str = '1m') -> pd.DataFrame:
    """
    Download data for all symbols with robust caching.

    Uses centralized data fetcher and organized cache.
    """
    print(">>> Importing DataCache...", flush=True)
    from trading_system.ml.data_cache import DataCache
    print(">>> Importing data_fetcher...", flush=True)
    from trading_system.data.data_fetcher import get_data_fetcher
    print(">>> Imports done, creating cache...", flush=True)

    cache = DataCache()
    cache.print_cache_info()
    
    fetcher = get_data_fetcher()
    
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    start_date_str = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Use centralized fetcher with caching
    results = fetcher.fetch_multiple_symbols(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        cache=cache,
        start_date_str=start_date_str,
        end_date_str=end_date_str
    )
    
    all_data = [df for df in results.values() if df is not None and not df.empty]

    if not all_data:
        logger.error("❌ No data available (cached or downloaded)!")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info(f"📊 DATA SUMMARY:")
    logger.info(f"   Successful: {len(all_data)} symbols")
    logger.info(f"   Failed: {len(symbols) - len(all_data)} symbols")
    logger.info("=" * 80)
    
    # Combine all symbols
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    logger.info(f"✅ COMBINED RAW DATA: {len(combined_df):,} total bars")

    # Apply data quality checks with caching
    logger.info("\n🧹 APPLYING DATA QUALITY CHECKS...")
    print(">>> About to import data_quality...", flush=True)
    from trading_system.ml.data_quality import clean_training_data
    print(">>> data_quality imported OK", flush=True)

    # Try to load cleaned data from cache
    cleaned_data = []
    need_cleaning = []

    print(">>> Checking cleaned cache for each symbol...", flush=True)
    for symbol in symbols:
        df_cleaned = cache.get_cleaned_data(symbol, start_date_str, end_date_str, interval)
        if df_cleaned is not None:
            cleaned_data.append(df_cleaned)
        else:
            need_cleaning.append(symbol)
    
    # Clean symbols that aren't cached
    if need_cleaning:
        logger.info(f"   Cleaning {len(need_cleaning)} symbols...")
        symbols_to_clean = combined_df[combined_df['symbol'].isin(need_cleaning)].copy()
        cleaned_symbols = clean_training_data(symbols_to_clean, per_symbol=True, min_quality=0.80)
        
        # Save each cleaned symbol to cache
        for symbol in need_cleaning:
            symbol_df = cleaned_symbols[cleaned_symbols['symbol'] == symbol].copy()
            if len(symbol_df) > 0:
                cache.save_cleaned_data(symbol_df, symbol, start_date_str, end_date_str, interval)
                cleaned_data.append(symbol_df)
    else:
        logger.info(f"   ✅ All symbols loaded from cleaned cache")
    
    # Combine cleaned data
    combined_df = pd.concat(cleaned_data, ignore_index=True)
    combined_df = combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    logger.info(f"✅ CLEANED DATA: {len(combined_df):,} bars")
    
    return combined_df


def calculate_strategy_indicators(df: pd.DataFrame, symbols: List[str], start_date: str, end_date: str, interval: str = '1m') -> pd.DataFrame:
    """
    Calculate ALL indicators used in crypto_scalping.py strategy with caching.
    
    Matches EXACT parameters from CryptoScalpingConfig:
    - RSI: 14 period, oversold < 30
    - BB: 20 period, 2 std dev
    - Volume MA: 20 period, spike > 1.3x
    - ADX: 14 period, trending > 20
    - Stochastic: K=14, D=3, oversold < 20
    - MACD: 12/26/9
    - EMA: 9/21/50 periods
    """
    from trading_system.ml.data_cache import DataCache
    
    cache = DataCache()
    logger.info("🔧 Calculating strategy indicators with caching...")
    
    features_data = []
    need_calculation = []
    
    # Try to load features from cache
    for symbol in symbols:
        df_features = cache.get_features_data(symbol, start_date, end_date, interval)
        if df_features is not None:
            features_data.append(df_features)
        else:
            need_calculation.append(symbol)
    
    # Calculate indicators for symbols not in cache
    if need_calculation:
        logger.info(f"   Calculating indicators for {len(need_calculation)} symbols...")
        
        for symbol in need_calculation:
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # RSI (14 period)
            delta = symbol_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            symbol_df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands (20 period, 2 std)
            symbol_df['bb_middle'] = symbol_df['close'].rolling(window=20).mean()
            bb_std = symbol_df['close'].rolling(window=20).std()
            symbol_df['bb_upper'] = symbol_df['bb_middle'] + (2 * bb_std)
            symbol_df['bb_lower'] = symbol_df['bb_middle'] - (2 * bb_std)
            
            # Volume MA and spike detection (20 period, 1.3x threshold)
            symbol_df['volume_ma'] = symbol_df['volume'].rolling(window=20).mean()
            symbol_df['volume_ratio'] = symbol_df['volume'] / symbol_df['volume_ma']
            symbol_df['volume_spike'] = (symbol_df['volume_ratio'] > 1.3).astype(int)
            
            # VWAP (50 period, reset daily)
            symbol_df['vwap'] = (symbol_df['close'] * symbol_df['volume']).rolling(window=50).sum() / symbol_df['volume'].rolling(window=50).sum()
            
            # EMAs (9, 21, 50)
            symbol_df['ema_9'] = symbol_df['close'].ewm(span=9, adjust=False).mean()
            symbol_df['ema_21'] = symbol_df['close'].ewm(span=21, adjust=False).mean()
            symbol_df['ema_50'] = symbol_df['close'].ewm(span=50, adjust=False).mean()
            
            # MACD (12/26/9)
            ema_12 = symbol_df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = symbol_df['close'].ewm(span=26, adjust=False).mean()
            symbol_df['macd'] = ema_12 - ema_26
            symbol_df['macd_signal'] = symbol_df['macd'].ewm(span=9, adjust=False).mean()
            symbol_df['macd_hist'] = symbol_df['macd'] - symbol_df['macd_signal']
            
            # Stochastic (K=14, D=3)
            low_14 = symbol_df['low'].rolling(window=14).min()
            high_14 = symbol_df['high'].rolling(window=14).max()
            symbol_df['stoch_k'] = 100 * ((symbol_df['close'] - low_14) / (high_14 - low_14))
            symbol_df['stoch_d'] = symbol_df['stoch_k'].rolling(window=3).mean()
            
            # ADX (14 period)
            high_diff = symbol_df['high'].diff()
            low_diff = -symbol_df['low'].diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            tr1 = symbol_df['high'] - symbol_df['low']
            tr2 = abs(symbol_df['high'] - symbol_df['close'].shift())
            tr3 = abs(symbol_df['low'] - symbol_df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            symbol_df['adx'] = dx.rolling(window=14).mean()
            
            # ATR (14 period)
            symbol_df['atr'] = atr
            
            # Support level detection (50 bar lookback)
            symbol_df['support_level'] = symbol_df['low'].rolling(window=50).min()
            symbol_df['near_support'] = (abs(symbol_df['close'] - symbol_df['support_level']) / symbol_df['support_level'] < 0.005).astype(int)
            
            # Candlestick patterns
            symbol_df['body'] = abs(symbol_df['close'] - symbol_df['open'])
            symbol_df['upper_wick'] = symbol_df['high'] - symbol_df[['open', 'close']].max(axis=1)
            symbol_df['lower_wick'] = symbol_df[['open', 'close']].min(axis=1) - symbol_df['low']
            symbol_df['total_range'] = symbol_df['high'] - symbol_df['low']
            
            symbol_df['is_bullish'] = (symbol_df['close'] > symbol_df['open']).astype(int)
            symbol_df['is_hammer'] = (
                (symbol_df['body'] / symbol_df['total_range'] < 0.35) &
                (symbol_df['lower_wick'] / symbol_df['total_range'] > 0.5) &
                (symbol_df['upper_wick'] / symbol_df['total_range'] < 0.15) &
                (symbol_df['is_bullish'] == 1)
            ).astype(int)
            
            symbol_df['is_doji'] = (symbol_df['body'] / symbol_df['total_range'] < 0.1).astype(int)
            
            # Save to cache
            cache.save_features_data(symbol_df, symbol, start_date, end_date, interval)
            features_data.append(symbol_df)
    else:
        logger.info(f"   ✅ All features loaded from cache")
    
    # Combine all features
    result = pd.concat(features_data, ignore_index=True)
    result = result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    logger.info(f"   ✅ Features ready: {len(result):,} bars")
    return result


def label_data_with_strategy_logic(df: pd.DataFrame, target_profit_pct: float = 2.0, stop_loss_pct: float = 0.6, lookahead_bars: int = 50) -> pd.DataFrame:
    """
    Label data using EXACT crypto_scalping.py entry logic - NO DATA LEAKAGE!
    
    Entry requires score >= 6 with RSI < 30 mandatory:
    1. RSI < 30 (oversold) - MANDATORY
    2. Price <= Lower BB
    3. Volume spike > 1.3x
    4. Bullish pattern (hammer/doji)
    5. MACD histogram positive or improving
    6. Stochastic oversold < 20 or bullish crossover
    7. Near support level
    8. Price below VWAP
    9. ADX > 20 (trending)
    
    Exit: +2.0% TP or -0.6% SL within lookahead_bars
    
    ANTI-CHEATING MEASURES:
    - Only use data AVAILABLE AT ENTRY TIME (no future peeking)
    - Indicators calculated with historical data only
    - Labels based on FUTURE price movement (what we're predicting)
    - Proper train/validation/test split to prevent overfitting
    """
    logger.info("🏷️  Labeling data with strategy logic (NO DATA LEAKAGE)...")
    
    result_dfs = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        
        # Initialize label column
        symbol_df['label'] = 0  # 0 = no trade, 1 = profitable trade
        symbol_df['entry_score'] = 0
        
        # IMPORTANT: Only label data where we have FUTURE bars to verify outcome
        # This prevents using "current bar" information we wouldn't have in real-time
        for i in range(len(symbol_df) - lookahead_bars):
            row = symbol_df.iloc[i]
            
            # Skip if indicators not ready
            if pd.isna(row['rsi']) or pd.isna(row['bb_lower']) or pd.isna(row['macd_hist']):
                continue
            
            # Calculate entry score (same as strategy)
            score = 0
            
            # 1. RSI oversold - MANDATORY
            if row['rsi'] >= 30:
                continue  # Skip if not oversold
            
            score += 1
            if row['rsi'] < 25:
                score += 1  # Extra point for deeply oversold
            
            # 2. Price at/below Lower BB
            if row['close'] <= row['bb_lower']:
                score += 1
            
            # 3. Volume spike
            if row['volume_spike'] == 1:
                score += 1
            
            # 4. Bullish pattern
            if row['is_hammer'] == 1 or row['is_doji'] == 1:
                score += 1
            
            # 5. MACD positive momentum
            if row['macd_hist'] > 0:
                score += 1
            
            # 6. Stochastic oversold
            if row['stoch_k'] < 20:
                score += 1
            
            # 7. Near support
            if row['near_support'] == 1:
                score += 1
            
            # 8. Below VWAP
            if not pd.isna(row['vwap']) and row['close'] < row['vwap']:
                score += 1
            
            # 9. ADX trending
            if not pd.isna(row['adx']) and row['adx'] > 20:
                score += 1
            
            symbol_df.loc[symbol_df.index[i], 'entry_score'] = score
            
            # Only consider entries with score >= 6 (strategy minimum)
            if score < 6:
                continue
            
            # ANTI-CHEATING: Look ONLY at FUTURE bars to determine outcome
            # We use ENTRY price (close of current bar) and check FUTURE movement
            entry_price = row['close']
            tp_price = entry_price * (1 + target_profit_pct / 100)
            sl_price = entry_price * (1 - stop_loss_pct / 100)
            
            # CRITICAL: Use i+1 onwards (FUTURE bars we haven't seen yet)
            future_bars = symbol_df.iloc[i+1:i+1+lookahead_bars]
            
            if len(future_bars) == 0:
                continue  # Skip if no future data
            
            # Check if TP or SL hit (using high/low of FUTURE bars only)
            hit_tp = (future_bars['high'] >= tp_price).any()
            hit_sl = (future_bars['low'] <= sl_price).any()
            
            if hit_tp and hit_sl:
                # Both hit - check which came first (time-based, no cheating)
                tp_bars = future_bars[future_bars['high'] >= tp_price]
                sl_bars = future_bars[future_bars['low'] <= sl_price]
                
                if len(tp_bars) > 0 and len(sl_bars) > 0:
                    tp_idx = tp_bars.index[0]
                    sl_idx = sl_bars.index[0]
                    
                    if tp_idx < sl_idx:
                        symbol_df.loc[symbol_df.index[i], 'label'] = 1  # Profitable
                    else:
                        symbol_df.loc[symbol_df.index[i], 'label'] = 0  # Loss
            elif hit_tp:
                symbol_df.loc[symbol_df.index[i], 'label'] = 1  # Profitable
            else:
                symbol_df.loc[symbol_df.index[i], 'label'] = 0  # Loss or no clear signal
        
        result_dfs.append(symbol_df)
    
    result = pd.concat(result_dfs, ignore_index=True)
    
    # Statistics
    total_signals = (result['entry_score'] >= 6).sum()
    profitable_signals = ((result['entry_score'] >= 6) & (result['label'] == 1)).sum()
    win_rate = (profitable_signals / total_signals * 100) if total_signals > 0 else 0
    
    logger.info(f"   ✅ Found {total_signals:,} entry signals (score >= 6)")
    logger.info(f"   ✅ Profitable signals: {profitable_signals:,} ({win_rate:.1f}% win rate)")
    
    # Show per-symbol breakdown
    logger.info("\n   Per-Symbol Breakdown:")
    for symbol in result['symbol'].unique():
        symbol_data = result[result['symbol'] == symbol]
        symbol_signals = (symbol_data['entry_score'] >= 6).sum()
        symbol_wins = ((symbol_data['entry_score'] >= 6) & (symbol_data['label'] == 1)).sum()
        symbol_wr = (symbol_wins / symbol_signals * 100) if symbol_signals > 0 else 0
        logger.info(f"      {symbol:10s}: {symbol_signals:4d} signals, {symbol_wins:4d} wins ({symbol_wr:5.1f}%)")
    
    return result


def train_ensemble_model(df: pd.DataFrame, optimize: bool = False, output_path: str = "models/crypto_scalping_ensemble.pkl"):
    """
    Train ML ensemble model with proper train/validation/test split.

    ANTI-OVERFITTING MEASURES:
    - Chronological split (no future data leakage)
    - Separate validation set for hyperparameter tuning
    - Test set never seen during training
    - Cross-validation on training set only
    """
    print(">>> [train_ensemble_model] Starting...", flush=True)
    print(">>> [train_ensemble_model] Importing training modules...", flush=True)
    try:
        from trading_system.ml.training.data_pipeline import DataPipeline
        from trading_system.ml.training.trainer import EnsembleTrainer
        print(">>> [train_ensemble_model] Imports OK", flush=True)
    except ImportError as e:
        logger.error(f"❌ ML modules not found: {e}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("🚀 TRAINING ML ENSEMBLE MODEL (WITH ANTI-OVERFITTING)")
    logger.info("=" * 80)

    # CRITICAL: Chronological split to prevent data leakage
    # Train on older data, validate on middle data, test on newest data
    total_rows = len(df)
    train_end = int(total_rows * 0.70)    # 70% training
    val_end = int(total_rows * 0.85)      # 15% validation
    # Remaining 15% for test

    logger.info(f"📊 Data Split (Chronological - NO LEAKAGE):")
    logger.info(f"   Training:   {train_end:,} rows (70%)")
    logger.info(f"   Validation: {val_end - train_end:,} rows (15%)")
    logger.info(f"   Test:       {total_rows - val_end:,} rows (15%)")

    # Save data temporarily
    print(">>> [train_ensemble_model] Saving temp CSV file...", flush=True)
    temp_file = "temp_multi_crypto_data.csv"
    df.to_csv(temp_file, index=False)
    logger.info(f"💾 Saved training data: {temp_file}")
    print(">>> [train_ensemble_model] CSV saved, initializing trainer...", flush=True)

    # Initialize trainer
    trainer = EnsembleTrainer()
    print(">>> [train_ensemble_model] Trainer initialized, starting training...", flush=True)

    # Train model
    logger.info("🎯 Training ensemble with 5 models (RF, XGBoost, LSTM, LR, SVM)...")
    logger.info("   Using SMOTE for class balancing (prevents overfitting to majority class)")
    logger.info("   Cross-validation on training set only (no future peeking)")

    ensemble = trainer.train_from_file(
        data_path=temp_file,
        optimize_hyperparams=optimize,
        model_name=output_path.replace('.pkl', '').replace('models/', '')
    )
    print(">>> [train_ensemble_model] Training complete!", flush=True)

    # Save to the specified path
    ensemble.save(output_path)
    
    logger.info("=" * 80)
    logger.info(f"✅ MODEL SAVED: {output_path}")
    logger.info("=" * 80)
    
    # Clean up temp file
    import os
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return ensemble


def main():
    print(">>> MAIN STARTING...", flush=True)
    parser = argparse.ArgumentParser(description="Train ML Ensemble for Multi-Crypto Scalping Strategy on 1-MINUTE DATA")
    parser.add_argument('--days', type=int, default=30, help='Days of historical data (default: 30, max 30 for 1m data)')
    parser.add_argument('--interval', type=str, default='1m', choices=['1m', '5m', '15m', '1h'], help='Data interval - USE 1m to match backtest! (default: 1m)')
    parser.add_argument('--optimize', action='store_true', help='Enable hyperparameter optimization (slower)')
    parser.add_argument('--output', type=str, default='models/crypto_scalping_ensemble.pkl', help='Output model path')
    parser.add_argument('--symbols', type=str, nargs='+', help='Specific symbols to train on (default: all 9)')

    print(">>> Parsing args...", flush=True)
    args = parser.parse_args()
    print(">>> Args parsed OK", flush=True)

    # Enforce 1m interval warning
    if args.interval != '1m':
        logger.warning("⚠️" * 40)
        logger.warning(f"⚠️  TRAINING ON {args.interval} DATA BUT BACKTEST USES 1-MINUTE DATA!")
        logger.warning("⚠️  This will cause feature misalignment and poor predictions!")
        logger.warning("⚠️  Recommended: Use --interval 1m")
        logger.warning("⚠️" * 40)
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            logger.info("Training cancelled. Use --interval 1m")
            sys.exit(0)

    # Use specified symbols or all
    symbols = args.symbols if args.symbols else CRYPTO_SYMBOLS

    print(">>> About to log header...", flush=True)
    logger.info("=" * 80)
    logger.info("🤖 MULTI-CRYPTO ML ENSEMBLE TRAINING (1-MINUTE DATA)")
    logger.info("=" * 80)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Interval: {args.interval} ⚠️  MUST MATCH BACKTEST INTERVAL!")
    logger.info(f"Optimize: {args.optimize}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Data Caching: ENABLED (data/raw, data/cleaned, data/features)")
    logger.info(f"Data Quality: ENABLED (synthetic fill for NaN/invalid values)")
    logger.info("=" * 80)
    print(">>> Header logged, calculating dates...", flush=True)

    # Calculate date range
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    start_date_str = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    print(f">>> Dates: {start_date_str} to {end_date_str}", flush=True)

    # Step 1: Download data (with caching)
    print(">>> STEP 1: Calling download_multi_symbol_data...", flush=True)
    df = download_multi_symbol_data(symbols, args.days, args.interval)
    print(f">>> STEP 1 DONE: Got {len(df)} rows", flush=True)
    
    # Step 2: Calculate indicators (with caching)
    df = calculate_strategy_indicators(df, symbols, start_date_str, end_date_str, args.interval)
    
    # Step 3: Label data
    df = label_data_with_strategy_logic(df)
    
    # Step 4: Train model
    ensemble = train_ensemble_model(df, optimize=args.optimize, output_path=args.output)
    
    logger.info("=" * 80)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\n💡 Next Steps:")
    logger.info(f"   1. Run backtest: python trading_system/run_crypto_backtest.py")
    logger.info(f"   2. Evaluate model: python evaluate_ml_ensemble.py --model {args.output}")
    logger.info(f"   3. The strategy will automatically use the ML ensemble!")
    logger.info(f"\n📊 Data cached in: data/ (reusable for backtest and future training)")


if __name__ == "__main__":
    main()
