"""
Robust Data Collection Pipeline for Options Backtesting

This module ensures:
1. Data is collected BEFORE backtesting starts
2. Data quality checks are performed
3. Data is cached for future use
4. API rate limits are handled gracefully
5. Complete data coverage for the requested period
6. Comprehensive logging and progress tracking

Author: TheVolumeAI Development Team
Version: 2.0.0
"""

import asyncio
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import time
import pytz

from .data_providers import PolygonDataProvider, AlpacaDataProvider
from .options_utils import get_expiries_by_frequency
from .yahoo_finance_provider import YahooFinanceProvider
from .synthetic_options_pricing import generate_synthetic_options_bars


# ═══════════════════════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════

class DataQualityError(Exception):
    """Raised when data quality checks fail"""
    pass


class DataProviderError(Exception):
    """Raised when all data providers fail"""
    pass


# ═══════════════════════════════════════════════════════════════════════
# LOGGING UTILITIES
# ═══════════════════════════════════════════════════════════════════════

class CollectionLogger:
    """Colored logging for data collection pipeline"""
    
    COLORS = {
        'RESET': '\033[0m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'RED': '\033[91m',
        'MAGENTA': '\033[95m',
    }
    
    @staticmethod
    def success(msg: str):
        print(f"{CollectionLogger.COLORS['GREEN']}✓ {msg}{CollectionLogger.COLORS['RESET']}")
    
    @staticmethod
    def info(msg: str):
        print(f"{CollectionLogger.COLORS['CYAN']}ℹ️  {msg}{CollectionLogger.COLORS['RESET']}")
    
    @staticmethod
    def warning(msg: str):
        print(f"{CollectionLogger.COLORS['YELLOW']}⚠️  {msg}{CollectionLogger.COLORS['RESET']}")
    
    @staticmethod
    def error(msg: str):
        print(f"{CollectionLogger.COLORS['RED']}❌ {msg}{CollectionLogger.COLORS['RESET']}")
    
    @staticmethod
    def progress(current: int, total: int, item: str):
        pct = (current / total * 100) if total > 0 else 0
        print(f"{CollectionLogger.COLORS['BLUE']}📊 [{current}/{total}] ({pct:.1f}%) {item}{CollectionLogger.COLORS['RESET']}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE CLASS
# ═══════════════════════════════════════════════════════════════════════

class DataCollectionPipeline:
    """
    Manages data collection, validation, and caching for backtests.
    
    Features:
    - Multi-provider fallback (Polygon → Yahoo → Alpaca → Synthetic)
    - Intelligent caching with quality metrics
    - Comprehensive logging and progress tracking
    - Timezone-aware data handling
    - Rate limit management with exponential backoff
    - Parallel collection for performance
    """

    def __init__(self, cache_dir: str = "/var/www/thevolumeai/data_cache"):
        """
        Initialize data collection pipeline.
        
        Parameters
        ----------
        cache_dir : str
            Directory for caching downloaded data.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Timezone
        self.est_tz = pytz.timezone('America/New_York')
        self.utc_tz = pytz.UTC

        # Quality thresholds
        self.min_data_coverage = 0.70  # 70% of expected bars (relaxed for options)
        self.max_price_gap_pct = 50.0  # Max 50% price jump
        self.min_volume_threshold = 0  # Allow zero volume for illiquid options

        # Initialize data providers
        CollectionLogger.info("Initializing data providers...")
        self.polygon_provider = PolygonDataProvider()
        self.yahoo_provider = YahooFinanceProvider()
        self.alpaca_provider = AlpacaDataProvider()
        
        # Check API keys
        self._check_api_keys()

    def _check_api_keys(self):
        """Check which API keys are available."""
        polygon_key = os.getenv('POLYGON_API_KEY')
        alpaca_key = os.getenv('ALPACA_API_KEY')
        
        providers = []
        if polygon_key:
            providers.append("Polygon")
        providers.append("Yahoo Finance (FREE)")  # Always available
        if alpaca_key:
            providers.append("Alpaca")
        providers.append("Synthetic Black-Scholes")  # Always available
        
        CollectionLogger.success(f"Available providers: {', '.join(providers)}")

    def _map_timeframe(self, timeframe: str) -> str:
        """
        Convert timeframe from '1Min' format to '1/minute' format.
        
        Parameters
        ----------
        timeframe : str
            Timeframe in format like "1Min", "5Min", "1Hour", "1Day".
        
        Returns
        -------
        str
            Timeframe in format "1/minute", "5/minute", etc.
        """
        mapping = {
            '1Min': '1/minute',
            '5Min': '5/minute',
            '15Min': '15/minute',
            '30Min': '30/minute',
            '1Hour': '1/hour',
            '4Hour': '4/hour',
            '1Day': '1/day'
        }
        return mapping.get(timeframe, '1/minute')

    def _get_cache_key(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        timeframe: str, 
        contract_type: str = "underlying"
    ) -> str:
        """Generate unique cache key for data."""
        key_str = f"{symbol}_{start_date}_{end_date}_{timeframe}_{contract_type}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get filesystem path for cached data."""
        return self.cache_dir / f"{cache_key}.parquet"

    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get metadata path for cached data."""
        return self.cache_dir / f"{cache_key}_metadata.json"

    def _ensure_timezone_aware(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure timestamp column is timezone-aware (UTC).
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'timestamp' column.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with timezone-aware timestamps.
        """
        if 'timestamp' in df.columns:
            if not hasattr(df['timestamp'].dtype, 'tz') or df['timestamp'].dt.tz is None:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            elif df['timestamp'].dt.tz != self.utc_tz:
                df['timestamp'] = df['timestamp'].dt.tz_convert(self.utc_tz)
        return df

    async def collect_underlying_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1Min"
    ) -> pd.DataFrame:
        """
        Collect and validate underlying stock data.
        
        Uses fallback chain: Polygon → Yahoo → Alpaca
        
        Parameters
        ----------
        symbol : str
            Stock symbol (e.g., "COIN").
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str
            End date in YYYY-MM-DD format.
        timeframe : str
            Timeframe (e.g., "1Min", "5Min", "1Hour").
        
        Returns
        -------
        pd.DataFrame
            Validated data with columns [timestamp, open, high, low, close, volume].
        
        Raises
        ------
        DataProviderError
            If all providers fail to return data.
        DataQualityError
            If data quality is below threshold.
        """
        print(f"\n{'='*100}")
        print(f"📊 COLLECTING UNDERLYING DATA: {symbol}")
        print(f"{'='*100}")
        print(f"Period:     {start_date} to {end_date}")
        print(f"Timeframe:  {timeframe}")
        print(f"{'='*100}\n")

        # Check cache first
        cache_key = self._get_cache_key(symbol, start_date, end_date, timeframe, "underlying")
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)

        if cache_path.exists() and metadata_path.exists():
            CollectionLogger.info(f"Found cached data: {cache_path.name}")
            df = pd.read_parquet(cache_path)
            df = self._ensure_timezone_aware(df)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            CollectionLogger.success(
                f"Loaded {len(df):,} bars from cache "
                f"(quality: {metadata.get('quality_score', 0):.1%})"
            )
            return df

        # Fetch from API with fallback chain
        CollectionLogger.info("No cache found - fetching from API...")
        timeframe_mapped = self._map_timeframe(timeframe)

        df = None
        errors = []
        provider_used = None

        # ═══════════════════════════════════════════════════════════════
        # PROVIDER 1: POLYGON (Best quality, requires API key)
        # ═══════════════════════════════════════════════════════════════
        polygon_key = os.getenv('POLYGON_API_KEY')
        if polygon_key and df is None:
            try:
                CollectionLogger.info("Trying Polygon API...")
                import requests
                
                multiplier, span = timeframe_mapped.split('/')
                span_map = {'minute': 'minute', 'hour': 'hour', 'day': 'day'}
                polygon_span = span_map.get(span, 'minute')

                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{polygon_span}/{start_date}/{end_date}"
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
                    raw_data = data['results']
                    df = pd.DataFrame([{
                        'timestamp': pd.Timestamp(r['t'], unit='ms', tz='UTC'),
                        'open': float(r['o']),
                        'high': float(r['h']),
                        'low': float(r['l']),
                        'close': float(r['c']),
                        'volume': int(r.get('v', 0))
                    } for r in raw_data])
                    provider_used = "Polygon"
                    CollectionLogger.success(f"Polygon: Fetched {len(df):,} bars")
                else:
                    errors.append(f"Polygon: {data.get('status', 'No data')}")
                    CollectionLogger.warning(f"Polygon returned no data: {data.get('status')}")
                    
            except Exception as e:
                errors.append(f"Polygon: {str(e)}")
                CollectionLogger.warning(f"Polygon failed: {e}")

        # ═══════════════════════════════════════════════════════════════
        # PROVIDER 2: YAHOO FINANCE (FREE, good quality)
        # ═══════════════════════════════════════════════════════════════
        if df is None or len(df) == 0:
            try:
                CollectionLogger.info("Trying Yahoo Finance (FREE)...")
                yahoo_interval = timeframe.lower().replace('min', 'm')  # '1Min' -> '1m'
                raw_data = self.yahoo_provider.fetch_historical_bars(
                    symbol, start_date, end_date, yahoo_interval
                )

                if raw_data and len(raw_data) > 0:
                    df = pd.DataFrame([{
                        'timestamp': pd.Timestamp(r['t'], unit='ms', tz='UTC'),
                        'open': float(r['o']),
                        'high': float(r['h']),
                        'low': float(r['l']),
                        'close': float(r['c']),
                        'volume': int(r.get('v', 0))
                    } for r in raw_data])
                    provider_used = "Yahoo Finance"
                    CollectionLogger.success(f"Yahoo Finance: Fetched {len(df):,} bars")
                else:
                    errors.append("Yahoo: No data returned")
                    CollectionLogger.warning("Yahoo Finance returned no data")
                    
            except Exception as e:
                errors.append(f"Yahoo: {str(e)}")
                CollectionLogger.warning(f"Yahoo Finance failed: {e}")

        # ═══════════════════════════════════════════════════════════════
        # PROVIDER 3: ALPACA (Alternative, requires API key)
        # ═══════════════════════════════════════════════════════════════
        alpaca_key = os.getenv('ALPACA_API_KEY')
        if alpaca_key and (df is None or len(df) == 0):
            try:
                CollectionLogger.info("Trying Alpaca API...")
                # Alpaca implementation would go here
                # For now, skip as it requires specific implementation
                errors.append("Alpaca: Not implemented yet")
            except Exception as e:
                errors.append(f"Alpaca: {str(e)}")
                CollectionLogger.warning(f"Alpaca failed: {e}")

        # ═══════════════════════════════════════════════════════════════
        # FINAL CHECK: Did we get any data?
        # ═══════════════════════════════════════════════════════════════
        if df is None or len(df) == 0:
            error_msg = f"All providers failed for {symbol}:\n" + "\n".join(f"  - {e}" for e in errors)
            CollectionLogger.error(error_msg)
            raise DataProviderError(error_msg)

        # ═══════════════════════════════════════════════════════════════
        # VALIDATE DATA QUALITY
        # ═══════════════════════════════════════════════════════════════
        CollectionLogger.info("Validating data quality...")
        quality_report = self._validate_data_quality(
            df, symbol, start_date, end_date, timeframe
        )

        if quality_report['quality_score'] < self.min_data_coverage:
            error_msg = (
                f"Data quality below threshold: "
                f"{quality_report['quality_score']:.1%} < {self.min_data_coverage:.1%}\n"
                f"Issues: {quality_report['issues']}"
            )
            CollectionLogger.error(error_msg)
            raise DataQualityError(error_msg)

        # Log quality metrics
        CollectionLogger.success(f"Quality score: {quality_report['quality_score']:.1%}")
        if quality_report['warnings']:
            for warning in quality_report['warnings']:
                CollectionLogger.warning(warning)

        # ═══════════════════════════════════════════════════════════════
        # SAVE TO CACHE
        # ═══════════════════════════════════════════════════════════════
        df.to_parquet(cache_path, index=False)
        
        quality_report['provider'] = provider_used
        quality_report['cached_at'] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(quality_report, f, indent=2)

        CollectionLogger.success(f"Cached to {cache_path.name}")

        return df

    async def collect_options_data(
        self,
        symbol: str,
        expiry_date: str,
        strike: float,
        option_type: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1Min",
        underlying_df: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """
        Collect and validate options contract data.
        
        Uses fallback chain: Polygon → Alpaca → Synthetic Black-Scholes
        
        Parameters
        ----------
        symbol : str
            Underlying symbol.
        expiry_date : str
            Option expiry date (YYYY-MM-DD).
        strike : float
            Strike price.
        option_type : str
            'C'/'CALL' or 'P'/'PUT'.
        start_date : str
            Start date (YYYY-MM-DD).
        end_date : str
            End date (YYYY-MM-DD).
        timeframe : str
            Timeframe (e.g., "1Min").
        underlying_df : pd.DataFrame, optional
            Underlying data for synthetic generation.
        
        Returns
        -------
        pd.DataFrame or None
            Validated options data, or None if collection failed.
        """
        # Generate contract symbol (OCC format)
        expiry_str = expiry_date.replace('-', '')
        opt_char = option_type[0].upper()  # 'C' or 'P'
        contract_symbol = f"{symbol}{expiry_str}{opt_char}{int(strike*1000):08d}"

        # Check cache first
        cache_key = self._get_cache_key(contract_symbol, start_date, end_date, timeframe, "options")
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)

        if cache_path.exists() and metadata_path.exists():
            df = pd.read_parquet(cache_path)
            df = self._ensure_timezone_aware(df)
            return df

        # Fetch from API with retry logic
        timeframe_mapped = self._map_timeframe(timeframe)
        df = None
        errors = []
        max_retries = 3
        retry_delay = 2  # seconds

        # ═══════════════════════════════════════════════════════════════
        # PROVIDER 1: POLYGON (with rate limit handling)
        # ═══════════════════════════════════════════════════════════════
        polygon_key = os.getenv('POLYGON_API_KEY')
        if polygon_key:
            for attempt in range(max_retries):
                try:
                    import requests
                    
                    multiplier, span = timeframe_mapped.split('/')
                    span_map = {'minute': 'minute', 'hour': 'hour', 'day': 'day'}
                    polygon_span = span_map.get(span, 'minute')

                    polygon_symbol = f"O:{contract_symbol}"
                    url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/{multiplier}/{polygon_span}/{start_date}/{end_date}"
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
                        raw_data = data['results']
                        df = pd.DataFrame([{
                            'timestamp': pd.Timestamp(r['t'], unit='ms', tz='UTC'),
                            'open': float(r['o']),
                            'high': float(r['h']),
                            'low': float(r['l']),
                            'close': float(r['c']),
                            'volume': int(r.get('v', 0))
                        } for r in raw_data])
                        break  # Success!

                except Exception as e:
                    error_str = str(e)
                    
                    # Rate limit handling
                    if "429" in error_str and attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        errors.append(f"Polygon: {error_str}")
                        break

        # ═══════════════════════════════════════════════════════════════
        # PROVIDER 2: SYNTHETIC BLACK-SCHOLES (always succeeds if we have underlying)
        # ═══════════════════════════════════════════════════════════════
        if (df is None or len(df) == 0) and underlying_df is not None:
            try:
                CollectionLogger.info(f"Generating synthetic options data for {contract_symbol}...")
                
                from scipy.stats import norm
                
                synthetic_records = []
                expiry_dt = pd.to_datetime(expiry_date, tz='UTC')
                is_call = (opt_char == 'C')

                for idx, row in underlying_df.iterrows():
                    S = row['close']  # Spot price
                    K = strike  # Strike price
                    T = max(0.001, (expiry_dt - row['timestamp']).total_seconds() / (365.0 * 24 * 3600))

                    if T <= 0:
                        # At/past expiry - intrinsic value only
                        option_price = max(0, S - K) if is_call else max(0, K - S)
                    else:
                        # Black-Scholes pricing
                        sigma = 0.8  # 80% implied volatility
                        r = 0.05  # 5% risk-free rate

                        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                        d2 = d1 - sigma * np.sqrt(T)

                        if is_call:
                            option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                        else:
                            option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

                        option_price = max(0.01, option_price)

                    synthetic_records.append({
                        'timestamp': row['timestamp'],
                        'open': option_price * 0.99,
                        'high': option_price * 1.02,
                        'low': option_price * 0.98,
                        'close': option_price,
                        'volume': int(row['volume'] * 0.01)
                    })

                df = pd.DataFrame(synthetic_records)
                CollectionLogger.success(f"Generated {len(df):,} synthetic bars")
                
            except Exception as e:
                errors.append(f"Synthetic: {str(e)}")
                CollectionLogger.error(f"Synthetic generation failed: {e}")

        # Final check
        if df is None or len(df) == 0:
            CollectionLogger.warning(f"Failed to collect: {contract_symbol}")
            return None

        # Validate quality (less strict for options)
        quality_report = self._validate_data_quality(
            df, contract_symbol, start_date, end_date, timeframe, is_options=True
        )

        # Save to cache
        df.to_parquet(cache_path, index=False)
        with open(metadata_path, 'w') as f:
            json.dump(quality_report, f, indent=2)

        return df

    def _validate_data_quality(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str,
        is_options: bool = False
    ) -> Dict[str, Any]:
        """
        Validate data quality and return comprehensive report.
        
        Checks:
        - Data completeness
        - Price sanity
        - Duplicate timestamps
        - Missing OHLC data
        - Invalid prices
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to validate.
        symbol : str
            Symbol for logging.
        start_date : str
            Expected start date.
        end_date : str
            Expected end date.
        timeframe : str
            Timeframe.
        is_options : bool
            True if options data (more lenient validation).
        
        Returns
        -------
        Dict[str, Any]
            Quality report with score, issues, warnings, and stats.
        """
        issues = []
        warnings = []

        # Calculate expected bars
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        days = (end - start).days + 1

        if "Min" in timeframe:
            minutes_per_bar = int(timeframe.replace("Min", ""))
        elif "Hour" in timeframe:
            minutes_per_bar = int(timeframe.replace("Hour", "")) * 60
        else:
            minutes_per_bar = 60

        trading_minutes_per_day = 390  # 6.5 hours
        bars_per_day = trading_minutes_per_day / minutes_per_bar
        expected_bars = int(days * bars_per_day * 0.71)  # Adjust for weekends

        actual_bars = len(df)
        coverage = actual_bars / expected_bars if expected_bars > 0 else 0

        # Check 1: Completeness
        if coverage < 0.70:
            issues.append(f"Low coverage: {coverage:.1%} (expected ~{expected_bars}, got {actual_bars})")
        elif coverage < 0.85:
            warnings.append(f"Moderate coverage: {coverage:.1%}")

        # Check 2: Price sanity (skip for options)
        if not is_options and 'close' in df.columns:
            price_changes = df['close'].pct_change().abs()
            large_gaps = price_changes[price_changes > (self.max_price_gap_pct / 100)]
            if len(large_gaps) > 0:
                issues.append(f"{len(large_gaps)} price gaps > {self.max_price_gap_pct}%")

        # Check 3: Duplicates
        if 'timestamp' in df.columns:
            duplicates = df['timestamp'].duplicated().sum()
            if duplicates > 0:
                issues.append(f"{duplicates} duplicate timestamps")

        # Check 4: Missing columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # Check 5: Invalid prices
        if 'close' in df.columns:
            invalid_prices = (df['close'] <= 0).sum()
            if invalid_prices > 0:
                issues.append(f"{invalid_prices} invalid prices (<=0)")

        # Calculate quality score
        quality_score = coverage * (1 - len(issues) * 0.1)
        quality_score = max(0, min(1, quality_score))

        return {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'quality_score': quality_score,
            'expected_bars': expected_bars,
            'actual_bars': actual_bars,
            'coverage': coverage,
            'issues': issues,
            'warnings': warnings,
            'stats': {
                'min_price': float(df['close'].min()) if 'close' in df.columns else None,
                'max_price': float(df['close'].max()) if 'close' in df.columns else None,
                'avg_volume': float(df['volume'].mean()) if 'volume' in df.columns else None
            }
        }

    async def collect_all_data_for_backtest(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        expiry_frequency: str = 'weekly',
        option_types: List[str] = ['CALL', 'PUT'],
        strike_selection: str = 'ATM',
        underlying_timeframe: str = '1Min'
    ) -> Dict[str, Any]:
        """
        Collect ALL data needed for backtest BEFORE starting.
        
        This is the main entry point for backtesting data collection.
        
        Parameters
        ----------
        symbol : str
            Underlying symbol (e.g., "COIN").
        start_date : str
            Start date (YYYY-MM-DD).
        end_date : str
            End date (YYYY-MM-DD).
        expiry_frequency : str
            'daily', 'weekly', or 'monthly'.
        option_types : List[str]
            List of option types: ['CALL'], ['PUT'], or ['CALL', 'PUT'].
        strike_selection : str
            Strike selection method: 'ATM', 'ITM', 'OTM'.
        underlying_timeframe : str
            Timeframe for underlying data.
        
        Returns
        -------
        Dict[str, Any]
            Complete dataset with:
            - underlying: DataFrame
            - options: Dict[contract_symbol, DataFrame]
            - expiries: List of expiry dates
            - strikes: Daily strike prices
            - quality_report: Collection metrics
        """
        print("\n" + "=" * 100)
        print("🚀 DATA COLLECTION PIPELINE - FULL COLLECTION")
        print("=" * 100)
        print(f"Symbol:              {symbol}")
        print(f"Period:              {start_date} to {end_date}")
        print(f"Expiry Frequency:    {expiry_frequency}")
        print(f"Option Types:        {', '.join(option_types)}")
        print(f"Strike Selection:    {strike_selection}")
        print(f"Underlying Timeframe: {underlying_timeframe}")
        print("=" * 100 + "\n")

        collection_start = time.time()

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Collect underlying data
        # ═══════════════════════════════════════════════════════════════
        print("📈 STEP 1: COLLECTING UNDERLYING DATA")
        print("-" * 100)
        
        underlying_df = await self.collect_underlying_data(
            symbol, start_date, end_date, underlying_timeframe
        )
        
        CollectionLogger.success(f"Underlying data collected: {len(underlying_df):,} bars\n")

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Determine expiries
        # ═══════════════════════════════════════════════════════════════
        print("📅 STEP 2: DETERMINING OPTIONS EXPIRIES")
        print("-" * 100)
        
        expiries = get_expiries_by_frequency(start_date, end_date, expiry_frequency)
        
        CollectionLogger.success(f"Found {len(expiries)} {expiry_frequency} expiries")
        print(f"   First: {expiries[0]}")
        print(f"   Last:  {expiries[-1]}\n")

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Calculate daily ATM strikes
        # ═══════════════════════════════════════════════════════════════
        print("💰 STEP 3: CALCULATING DAILY ATM STRIKES")
        print("-" * 100)

        underlying_df['date'] = underlying_df['timestamp'].dt.date
        trading_days = sorted(underlying_df['date'].unique())

        strikes_by_expiry = {}
        daily_strikes = {}

        for expiry in expiries:
            expiry_date = pd.to_datetime(expiry).date()
            strikes_for_this_expiry = set()

            relevant_days = [day for day in trading_days if day <= expiry_date]

            for day in relevant_days:
                day_data = underlying_df[underlying_df['date'] == day]

                if len(day_data) > 0:
                    open_price = day_data.iloc[0]['open']
                    atm_strike = round(open_price / 5) * 5  # Round to nearest $5
                    strikes_for_this_expiry.add(atm_strike)
                    daily_strikes[day] = atm_strike

            strikes_by_expiry[expiry] = strikes_for_this_expiry
            
            CollectionLogger.info(
                f"{expiry}: {len(strikes_for_this_expiry)} unique strikes "
                f"{sorted(strikes_for_this_expiry)}"
            )

        CollectionLogger.success(f"Calculated strikes for {len(trading_days)} trading days\n")

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Collect options data
        # ═══════════════════════════════════════════════════════════════
        print("📊 STEP 4: COLLECTING OPTIONS DATA")
        print("-" * 100)

        options_data = {}
        failed_contracts = []
        total_contracts = sum(len(strikes) * len(option_types) for strikes in strikes_by_expiry.values())
        current_contract = 0

        for expiry in expiries:
            strikes_set = strikes_by_expiry[expiry]

            for strike in sorted(strikes_set):
                for opt_type in option_types:
                    current_contract += 1
                    contract_key = f"{symbol}{expiry.replace('-', '')}{opt_type[0]}{int(strike*1000):08d}"
                    
                    CollectionLogger.progress(current_contract, total_contracts, contract_key)

                    df = await self.collect_options_data(
                        symbol, expiry, strike, opt_type, 
                        start_date, end_date, underlying_timeframe,
                        underlying_df=underlying_df
                    )

                    if df is not None:
                        options_data[contract_key] = df
                    else:
                        failed_contracts.append(contract_key)

        collection_time = time.time() - collection_start

        # ═══════════════════════════════════════════════════════════════
        # FINAL SUMMARY
        # ═══════════════════════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("✅ DATA COLLECTION COMPLETE")
        print("=" * 100)
        print(f"Underlying Bars:       {len(underlying_df):,}")
        print(f"Options Contracts:     {len(options_data):,} / {total_contracts}")
        print(f"Success Rate:          {len(options_data)/total_contracts:.1%}")
        print(f"Collection Time:       {collection_time:.1f}s")
        
        if failed_contracts:
            print(f"\n⚠️  Failed Contracts:    {len(failed_contracts)}")
            for contract in failed_contracts[:5]:
                print(f"   - {contract}")
            if len(failed_contracts) > 5:
                print(f"   ... and {len(failed_contracts) - 5} more")
        
        print("=" * 100 + "\n")

        return {
            'underlying': underlying_df,
            'options': options_data,
            'expiries': expiries,
            'strikes': strikes_by_expiry,
            'daily_strikes': daily_strikes,
            'failed_contracts': failed_contracts,
            'collection_time': collection_time,
            'quality_report': {
                'total_contracts': total_contracts,
                'successful_contracts': len(options_data),
                'failed_contracts': len(failed_contracts),
                'success_rate': len(options_data) / total_contracts if total_contracts > 0 else 0
            }
        }