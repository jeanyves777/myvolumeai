"""
FAST ML Ensemble Training Script

Pre-loads ALL sklearn components at startup to avoid Windows DLL loading delays.
Uses only sklearn models (no XGBoost/TensorFlow) for faster training.
"""
print(">>> FAST ENSEMBLE TRAINER - Pre-loading sklearn...", flush=True)

import warnings
warnings.filterwarnings('ignore')

# PRE-LOAD sklearn BEFORE any lazy loading kicks in
# This avoids the Windows DLL loading issues during training
print(">>> Loading sklearn.ensemble...", flush=True)
from sklearn.ensemble import RandomForestClassifier
print(">>> Loading sklearn.linear_model...", flush=True)
from sklearn.linear_model import LogisticRegression
print(">>> Loading sklearn.preprocessing...", flush=True)
from sklearn.preprocessing import StandardScaler
print(">>> Loading sklearn.metrics...", flush=True)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
print(">>> sklearn loaded OK!", flush=True)

# Try to load xgboost, but make it optional
print(">>> Attempting XGBoost load...", flush=True)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print(">>> XGBoost loaded OK!", flush=True)
except ImportError:
    XGBOOST_AVAILABLE = False
    print(">>> XGBoost not available, using sklearn only", flush=True)

import argparse
import logging
import sys
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import joblib

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


class FastEnsemble:
    """
    Fast ML ensemble using only sklearn models.

    Models:
    - RandomForest (40%) - Strong tree-based model
    - Logistic Regression (30%) - Fast linear model
    - XGBoost (30%) - Optional, if available
    """

    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scalers = {}
        self.is_trained = False
        self.feature_columns = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict:
        """Train all models in the ensemble."""

        self.feature_columns = X_train.columns.tolist()
        metrics = {}

        # Prepare scaled data
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_val_scaled = self.scalers['standard'].transform(X_val) if X_val is not None else None

        total_start = time.time()

        # 1. Random Forest (40%)
        print("\n>>> Training RandomForest...", flush=True)
        start = time.time()
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,  # Reduced for speed
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.models['random_forest'].fit(X_train, y_train)
        self.weights['random_forest'] = 0.40
        rf_time = time.time() - start
        print(f">>> RandomForest done in {rf_time:.1f}s", flush=True)

        # Evaluate RF
        if X_val is not None:
            rf_pred = self.models['random_forest'].predict(X_val)
            metrics['random_forest'] = {
                'accuracy': accuracy_score(y_val, rf_pred),
                'f1_score': f1_score(y_val, rf_pred, zero_division=0)
            }
            print(f">>> RF Validation: acc={metrics['random_forest']['accuracy']:.4f}, f1={metrics['random_forest']['f1_score']:.4f}", flush=True)

        # 2. Logistic Regression (30%)
        print("\n>>> Training Logistic Regression...", flush=True)
        start = time.time()
        self.models['logistic'] = LogisticRegression(
            max_iter=500,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            solver='lbfgs'
        )
        self.models['logistic'].fit(X_train_scaled, y_train)
        self.weights['logistic'] = 0.30
        lr_time = time.time() - start
        print(f">>> Logistic done in {lr_time:.1f}s", flush=True)

        # Evaluate LR
        if X_val is not None:
            lr_pred = self.models['logistic'].predict(X_val_scaled)
            metrics['logistic'] = {
                'accuracy': accuracy_score(y_val, lr_pred),
                'f1_score': f1_score(y_val, lr_pred, zero_division=0)
            }
            print(f">>> LR Validation: acc={metrics['logistic']['accuracy']:.4f}, f1={metrics['logistic']['f1_score']:.4f}", flush=True)

        # 3. XGBoost (30%) - if available
        if XGBOOST_AVAILABLE:
            print("\n>>> Training XGBoost...", flush=True)
            start = time.time()

            # Calculate class weights
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,  # Reduced for speed
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            self.models['xgboost'].fit(X_train_scaled, y_train, verbose=False)
            self.weights['xgboost'] = 0.30
            xgb_time = time.time() - start
            print(f">>> XGBoost done in {xgb_time:.1f}s", flush=True)

            # Evaluate XGB
            if X_val is not None:
                xgb_pred = self.models['xgboost'].predict(X_val_scaled)
                metrics['xgboost'] = {
                    'accuracy': accuracy_score(y_val, xgb_pred),
                    'f1_score': f1_score(y_val, xgb_pred, zero_division=0)
                }
                print(f">>> XGB Validation: acc={metrics['xgboost']['accuracy']:.4f}, f1={metrics['xgboost']['f1_score']:.4f}", flush=True)
        else:
            # Redistribute weights if no XGBoost
            self.weights['random_forest'] = 0.55
            self.weights['logistic'] = 0.45
            print(">>> XGBoost not available, using RF(55%) + LR(45%)", flush=True)

        # Normalize weights
        total_weight = sum(self.weights.values())
        for model_name in self.weights:
            self.weights[model_name] /= total_weight

        self.is_trained = True
        total_time = time.time() - total_start
        print(f"\n>>> ALL MODELS TRAINED in {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble probability predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble not trained")

        X_scaled = self.scalers['standard'].transform(X)

        weighted_proba = np.zeros(len(X))

        for model_name, model in self.models.items():
            weight = self.weights[model_name]

            if model_name == 'random_forest':
                proba = model.predict_proba(X)[:, 1]
            else:
                proba = model.predict_proba(X_scaled)[:, 1]

            weighted_proba += weight * proba

        return weighted_proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Get binary predictions."""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        pred = self.predict(X_test)
        proba = self.predict_proba(X_test)

        return {
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1_score': f1_score(y_test, pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.0
        }

    def save(self, path: str):
        """Save the ensemble."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        data = {
            'models': self.models,
            'weights': self.weights,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        joblib.dump(data, path)
        print(f">>> Ensemble saved to {path}", flush=True)

    def load(self, path: str):
        """Load a saved ensemble."""
        data = joblib.load(path)
        self.models = data['models']
        self.weights = data['weights']
        self.scalers = data['scalers']
        self.feature_columns = data['feature_columns']
        self.is_trained = data['is_trained']
        print(f">>> Ensemble loaded from {path}", flush=True)


def download_and_prepare_data(symbols: List[str], days: int, interval: str = '1m') -> pd.DataFrame:
    """Download and prepare data using cached data."""

    print(">>> Importing data modules...", flush=True)
    from trading_system.ml.data_cache import DataCache
    from trading_system.data.data_fetcher import get_data_fetcher
    from trading_system.ml.data_quality import clean_training_data
    print(">>> Data modules loaded", flush=True)

    cache = DataCache()
    fetcher = get_data_fetcher()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')

    # Fetch data
    print(f">>> Fetching data for {len(symbols)} symbols...", flush=True)
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
        logger.error("No data available!")
        sys.exit(1)

    # Combine
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    print(f">>> Combined {len(combined_df):,} bars", flush=True)

    # Clean data
    cleaned_data = []
    for symbol in symbols:
        df_cleaned = cache.get_cleaned_data(symbol, start_date_str, end_date_str, interval)
        if df_cleaned is not None:
            cleaned_data.append(df_cleaned)
        else:
            symbol_df = combined_df[combined_df['symbol'] == symbol].copy()
            cleaned = clean_training_data(symbol_df, per_symbol=True, min_quality=0.80)
            cache.save_cleaned_data(cleaned, symbol, start_date_str, end_date_str, interval)
            cleaned_data.append(cleaned)

    combined_df = pd.concat(cleaned_data, ignore_index=True)
    combined_df = combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    print(f">>> Cleaned data: {len(combined_df):,} bars", flush=True)

    return combined_df, start_date_str, end_date_str


def calculate_features(df: pd.DataFrame, symbols: List[str], start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Calculate all technical indicators with caching."""

    from trading_system.ml.data_cache import DataCache
    cache = DataCache()

    features_data = []

    for symbol in symbols:
        # Try cache first
        df_features = cache.get_features_data(symbol, start_date, end_date, interval)
        if df_features is not None:
            features_data.append(df_features)
            continue

        # Calculate features
        symbol_df = df[df['symbol'] == symbol].copy()

        # RSI
        delta = symbol_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        symbol_df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        symbol_df['bb_middle'] = symbol_df['close'].rolling(window=20).mean()
        bb_std = symbol_df['close'].rolling(window=20).std()
        symbol_df['bb_upper'] = symbol_df['bb_middle'] + (2 * bb_std)
        symbol_df['bb_lower'] = symbol_df['bb_middle'] - (2 * bb_std)

        # Volume
        symbol_df['volume_ma'] = symbol_df['volume'].rolling(window=20).mean()
        symbol_df['volume_ratio'] = symbol_df['volume'] / symbol_df['volume_ma']
        symbol_df['volume_spike'] = (symbol_df['volume_ratio'] > 1.3).astype(int)

        # VWAP
        symbol_df['vwap'] = (symbol_df['close'] * symbol_df['volume']).rolling(window=50).sum() / symbol_df['volume'].rolling(window=50).sum()

        # EMAs
        symbol_df['ema_9'] = symbol_df['close'].ewm(span=9, adjust=False).mean()
        symbol_df['ema_21'] = symbol_df['close'].ewm(span=21, adjust=False).mean()
        symbol_df['ema_50'] = symbol_df['close'].ewm(span=50, adjust=False).mean()

        # MACD
        ema_12 = symbol_df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = symbol_df['close'].ewm(span=26, adjust=False).mean()
        symbol_df['macd'] = ema_12 - ema_26
        symbol_df['macd_signal'] = symbol_df['macd'].ewm(span=9, adjust=False).mean()
        symbol_df['macd_hist'] = symbol_df['macd'] - symbol_df['macd_signal']

        # Stochastic
        low_14 = symbol_df['low'].rolling(window=14).min()
        high_14 = symbol_df['high'].rolling(window=14).max()
        symbol_df['stoch_k'] = 100 * ((symbol_df['close'] - low_14) / (high_14 - low_14))
        symbol_df['stoch_d'] = symbol_df['stoch_k'].rolling(window=3).mean()

        # ADX
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
        symbol_df['atr'] = atr

        # Support level
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

    result = pd.concat(features_data, ignore_index=True)
    result = result.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    print(f">>> Features calculated: {len(result):,} bars", flush=True)

    return result


def label_data(df: pd.DataFrame, target_profit: float = 2.0, stop_loss: float = 0.6, lookahead: int = 50) -> pd.DataFrame:
    """Label data based on strategy entry logic."""

    print(">>> Labeling data with strategy logic...", flush=True)
    result_dfs = []

    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df['label'] = 0
        symbol_df['entry_score'] = 0

        for i in range(len(symbol_df) - lookahead):
            row = symbol_df.iloc[i]

            if pd.isna(row['rsi']) or pd.isna(row['bb_lower']) or pd.isna(row['macd_hist']):
                continue

            # Entry score
            score = 0

            if row['rsi'] >= 30:
                continue  # RSI oversold mandatory

            score += 1
            if row['rsi'] < 25:
                score += 1

            if row['close'] <= row['bb_lower']:
                score += 1

            if row['volume_spike'] == 1:
                score += 1

            if row['is_hammer'] == 1 or row['is_doji'] == 1:
                score += 1

            if row['macd_hist'] > 0:
                score += 1

            if row['stoch_k'] < 20:
                score += 1

            if row['near_support'] == 1:
                score += 1

            if not pd.isna(row['vwap']) and row['close'] < row['vwap']:
                score += 1

            if not pd.isna(row['adx']) and row['adx'] > 20:
                score += 1

            symbol_df.loc[symbol_df.index[i], 'entry_score'] = score

            if score < 6:
                continue

            # Check future outcome
            entry_price = row['close']
            tp_price = entry_price * (1 + target_profit / 100)
            sl_price = entry_price * (1 - stop_loss / 100)

            future_bars = symbol_df.iloc[i+1:i+1+lookahead]

            if len(future_bars) == 0:
                continue

            hit_tp = (future_bars['high'] >= tp_price).any()
            hit_sl = (future_bars['low'] <= sl_price).any()

            if hit_tp and hit_sl:
                tp_bars = future_bars[future_bars['high'] >= tp_price]
                sl_bars = future_bars[future_bars['low'] <= sl_price]

                if len(tp_bars) > 0 and len(sl_bars) > 0:
                    if tp_bars.index[0] < sl_bars.index[0]:
                        symbol_df.loc[symbol_df.index[i], 'label'] = 1
            elif hit_tp:
                symbol_df.loc[symbol_df.index[i], 'label'] = 1

        result_dfs.append(symbol_df)

    result = pd.concat(result_dfs, ignore_index=True)

    total_signals = (result['entry_score'] >= 6).sum()
    profitable = ((result['entry_score'] >= 6) & (result['label'] == 1)).sum()
    win_rate = (profitable / total_signals * 100) if total_signals > 0 else 0

    print(f">>> Signals: {total_signals:,}, Profitable: {profitable:,} ({win_rate:.1f}%)", flush=True)

    return result


def prepare_ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract ML features from labeled data."""

    feature_cols = [
        'rsi', 'bb_middle', 'bb_upper', 'bb_lower', 'volume_ratio', 'volume_spike',
        'vwap', 'ema_9', 'ema_21', 'ema_50', 'macd', 'macd_signal', 'macd_hist',
        'stoch_k', 'stoch_d', 'adx', 'atr', 'near_support', 'is_bullish',
        'is_hammer', 'is_doji', 'entry_score'
    ]

    # Add price-based features
    df['price_vs_bb_lower'] = (df['close'] - df['bb_lower']) / df['bb_lower']
    df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    df['price_vs_ema9'] = (df['close'] - df['ema_9']) / df['ema_9']
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)

    feature_cols.extend(['price_vs_bb_lower', 'price_vs_vwap', 'price_vs_ema9', 'rsi_oversold', 'stoch_oversold'])

    # Filter to entries only
    entry_mask = df['entry_score'] >= 6
    features = df.loc[entry_mask, feature_cols].copy()
    labels = df.loc[entry_mask, 'label'].copy()

    # Handle NaN
    features = features.fillna(0)

    print(f">>> ML features: {len(features):,} samples, {len(feature_cols)} features", flush=True)

    return features, labels


def main():
    print("\n" + "=" * 80, flush=True)
    print("FAST ML ENSEMBLE TRAINING", flush=True)
    print("=" * 80, flush=True)

    parser = argparse.ArgumentParser(description="Fast ML Ensemble Training")
    parser.add_argument('--days', type=int, default=30, help='Days of data')
    parser.add_argument('--interval', type=str, default='1m', help='Data interval')
    parser.add_argument('--output', type=str, default='models/crypto_scalping_ensemble.pkl', help='Output path')
    args = parser.parse_args()

    total_start = time.time()

    # Step 1: Download/load data
    print("\n>>> STEP 1: Loading data...", flush=True)
    step_start = time.time()
    df, start_date, end_date = download_and_prepare_data(CRYPTO_SYMBOLS, args.days, args.interval)
    print(f">>> Step 1 done in {time.time() - step_start:.1f}s", flush=True)

    # Step 2: Calculate features
    print("\n>>> STEP 2: Calculating features...", flush=True)
    step_start = time.time()
    df = calculate_features(df, CRYPTO_SYMBOLS, start_date, end_date, args.interval)
    print(f">>> Step 2 done in {time.time() - step_start:.1f}s", flush=True)

    # Step 3: Label data
    print("\n>>> STEP 3: Labeling data...", flush=True)
    step_start = time.time()
    df = label_data(df)
    print(f">>> Step 3 done in {time.time() - step_start:.1f}s", flush=True)

    # Step 4: Prepare ML features
    print("\n>>> STEP 4: Preparing ML features...", flush=True)
    step_start = time.time()
    features, labels = prepare_ml_features(df)
    print(f">>> Step 4 done in {time.time() - step_start:.1f}s", flush=True)

    # Step 5: Split data
    print("\n>>> STEP 5: Splitting data...", flush=True)
    n = len(features)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train = features.iloc[:train_end]
    X_val = features.iloc[train_end:val_end]
    X_test = features.iloc[val_end:]
    y_train = labels.iloc[:train_end]
    y_val = labels.iloc[train_end:val_end]
    y_test = labels.iloc[val_end:]

    print(f">>> Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}", flush=True)

    # Step 6: Handle class imbalance
    print("\n>>> STEP 6: Balancing classes...", flush=True)
    step_start = time.time()

    # Simple random oversampling
    y_array = y_train.values
    unique, counts = np.unique(y_array, return_counts=True)

    if len(unique) >= 2:
        max_count = counts.max()
        X_resampled_list = []
        y_resampled_list = []

        for cls, count in zip(unique, counts):
            cls_mask = y_array == cls
            X_cls = X_train[cls_mask]
            y_cls = y_train[cls_mask]

            if count < max_count:
                n_oversample = max_count - count
                indices = np.random.choice(len(X_cls), size=n_oversample, replace=True)
                X_oversampled = X_cls.iloc[indices]
                y_oversampled = y_cls.iloc[indices]

                X_resampled_list.append(X_cls)
                X_resampled_list.append(X_oversampled)
                y_resampled_list.append(y_cls)
                y_resampled_list.append(y_oversampled)
            else:
                X_resampled_list.append(X_cls)
                y_resampled_list.append(y_cls)

        X_train = pd.concat(X_resampled_list, ignore_index=True)
        y_train = pd.concat(y_resampled_list, ignore_index=True)

        # Shuffle
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train.iloc[shuffle_idx].reset_index(drop=True)
        y_train = y_train.iloc[shuffle_idx].reset_index(drop=True)

    print(f">>> Step 6 done in {time.time() - step_start:.1f}s - Resampled: {len(X_train):,}", flush=True)

    # Step 7: Train ensemble
    print("\n>>> STEP 7: Training ensemble...", flush=True)
    step_start = time.time()
    ensemble = FastEnsemble()
    train_metrics = ensemble.train(X_train, y_train, X_val, y_val)
    print(f">>> Step 7 done in {time.time() - step_start:.1f}s", flush=True)

    # Step 8: Evaluate
    print("\n>>> STEP 8: Evaluating on test set...", flush=True)
    step_start = time.time()
    test_metrics = ensemble.evaluate(X_test, y_test)
    print(f">>> Step 8 done in {time.time() - step_start:.1f}s", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("TEST SET RESULTS:", flush=True)
    print("=" * 80, flush=True)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}", flush=True)

    # Step 9: Save
    print("\n>>> STEP 9: Saving model...", flush=True)
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    ensemble.save(args.output)

    total_time = time.time() - total_start
    print("\n" + "=" * 80, flush=True)
    print(f"TRAINING COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"Model saved to: {args.output}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
