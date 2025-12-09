"""
Data caching system for training and backtesting.

Provides robust caching for:
- Raw OHLCV data from Binance
- Cleaned data (after quality checks)
- Engineered features (indicators)

Both training and backtesting use the same cached data.
Only downloads missing data.
"""
print(">>> [data_cache.py] Module loading...", flush=True)

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
print(">>> [data_cache.py] pandas imported, loading joblib...", flush=True)
import joblib
import hashlib
print(">>> [data_cache.py] Module loaded OK", flush=True)

logger = logging.getLogger(__name__)


class DataCache:
    """
    Manages cached data with organized folder structure.
    
    Structure:
        data/
            raw/           # Raw OHLCV from Binance
                BTC_USD_1m_20241125_20241201.pkl
                ETH_USD_1m_20241125_20241201.pkl
            cleaned/       # After data quality checks
                BTC_USD_1m_20241125_20241201_cleaned.pkl
            features/      # With indicators
                BTC_USD_1m_20241125_20241201_features.pkl
            metadata.json  # Cache metadata
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.cleaned_dir = self.base_dir / "cleaned"
        self.features_dir = self.base_dir / "features"
        self.metadata_file = self.base_dir / "metadata.json"
        
        # Create directories
        for dir_path in [self.raw_dir, self.cleaned_dir, self.features_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
    
    def _get_cache_key(self, symbol: str, start_date: str, end_date: str, interval: str = '1m') -> str:
        """Generate cache key for a data request"""
        # Normalize symbol (BTC/USD -> BTC_USD)
        symbol_clean = symbol.replace('/', '_')
        # Format: BTC_USD_1m_20241125_20241201
        start_clean = start_date.replace('-', '')
        end_clean = end_date.replace('-', '')
        return f"{symbol_clean}_{interval}_{start_clean}_{end_clean}"
    
    def _get_file_path(self, cache_key: str, stage: str) -> Path:
        """Get file path for a cache key and stage"""
        if stage == 'raw':
            return self.raw_dir / f"{cache_key}.pkl"
        elif stage == 'cleaned':
            return self.cleaned_dir / f"{cache_key}_cleaned.pkl"
        elif stage == 'features':
            return self.features_dir / f"{cache_key}_features.pkl"
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def get_raw_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1m') -> Optional[pd.DataFrame]:
        """
        Get raw OHLCV data from cache.
        
        Returns None if not cached or expired.
        """
        cache_key = self._get_cache_key(symbol, start_date, end_date, interval)
        file_path = self._get_file_path(cache_key, 'raw')
        
        if not file_path.exists():
            return None
        
        # Check age
        file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
        if file_age.total_seconds() > 24 * 3600:  # Expire after 24 hours
            logger.info(f"   ⏰ Cache expired for {symbol} (age: {file_age.total_seconds()/3600:.1f}h)")
            return None
        
        try:
            df = joblib.load(file_path)
            logger.info(f"   📦 Loaded {symbol} from cache ({len(df):,} bars)")
            return df
        except Exception as e:
            logger.warning(f"   ❌ Failed to load cache for {symbol}: {e}")
            return None
    
    def save_raw_data(self, df: pd.DataFrame, symbol: str, start_date: str, end_date: str, interval: str = '1m'):
        """Save raw OHLCV data to cache"""
        cache_key = self._get_cache_key(symbol, start_date, end_date, interval)
        file_path = self._get_file_path(cache_key, 'raw')
        
        try:
            joblib.dump(df, file_path)
            
            # Update metadata
            self.metadata[cache_key] = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'interval': interval,
                'bars': len(df),
                'cached_at': datetime.now().isoformat(),
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            }
            self._save_metadata()
            
            logger.info(f"   💾 Saved {symbol} to cache ({len(df):,} bars)")
        except Exception as e:
            logger.warning(f"   ❌ Failed to save cache for {symbol}: {e}")
    
    def get_cleaned_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1m') -> Optional[pd.DataFrame]:
        """Get cleaned data from cache"""
        cache_key = self._get_cache_key(symbol, start_date, end_date, interval)
        file_path = self._get_file_path(cache_key, 'cleaned')
        
        if not file_path.exists():
            return None
        
        try:
            df = joblib.load(file_path)
            logger.info(f"   📦 Loaded cleaned {symbol} from cache ({len(df):,} bars)")
            return df
        except Exception as e:
            logger.warning(f"   ❌ Failed to load cleaned cache for {symbol}: {e}")
            return None
    
    def save_cleaned_data(self, df: pd.DataFrame, symbol: str, start_date: str, end_date: str, interval: str = '1m'):
        """Save cleaned data to cache"""
        cache_key = self._get_cache_key(symbol, start_date, end_date, interval)
        file_path = self._get_file_path(cache_key, 'cleaned')
        
        try:
            joblib.dump(df, file_path)
            logger.info(f"   💾 Saved cleaned {symbol} to cache ({len(df):,} bars)")
        except Exception as e:
            logger.warning(f"   ❌ Failed to save cleaned cache for {symbol}: {e}")
    
    def get_features_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1m') -> Optional[pd.DataFrame]:
        """Get features (with indicators) from cache"""
        cache_key = self._get_cache_key(symbol, start_date, end_date, interval)
        file_path = self._get_file_path(cache_key, 'features')
        
        if not file_path.exists():
            return None
        
        try:
            df = joblib.load(file_path)
            logger.info(f"   📦 Loaded features for {symbol} from cache ({len(df):,} bars)")
            return df
        except Exception as e:
            logger.warning(f"   ❌ Failed to load features cache for {symbol}: {e}")
            return None
    
    def save_features_data(self, df: pd.DataFrame, symbol: str, start_date: str, end_date: str, interval: str = '1m'):
        """Save features (with indicators) to cache"""
        cache_key = self._get_cache_key(symbol, start_date, end_date, interval)
        file_path = self._get_file_path(cache_key, 'features')
        
        try:
            joblib.dump(df, file_path)
            logger.info(f"   💾 Saved features for {symbol} to cache ({len(df):,} bars)")
        except Exception as e:
            logger.warning(f"   ❌ Failed to save features cache for {symbol}: {e}")
    
    def clear_cache(self, older_than_days: int = 7):
        """Clear cache files older than specified days"""
        cutoff = datetime.now() - timedelta(days=older_than_days)
        cleared = 0
        
        for dir_path in [self.raw_dir, self.cleaned_dir, self.features_dir]:
            for file_path in dir_path.glob("*.pkl"):
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff:
                    try:
                        file_path.unlink()
                        cleared += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
        
        if cleared > 0:
            logger.info(f"🗑️  Cleared {cleared} old cache files")
        
        return cleared
    
    def get_cache_info(self) -> Dict:
        """Get cache statistics"""
        info = {
            'total_files': 0,
            'total_size_mb': 0.0,
            'raw_files': 0,
            'cleaned_files': 0,
            'features_files': 0,
            'oldest_file': None,
            'newest_file': None
        }
        
        all_files = []
        
        for dir_name, dir_path in [('raw', self.raw_dir), ('cleaned', self.cleaned_dir), ('features', self.features_dir)]:
            for file_path in dir_path.glob("*.pkl"):
                all_files.append(file_path)
                info[f'{dir_name}_files'] += 1
                info['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
        
        info['total_files'] = len(all_files)
        
        if all_files:
            mtimes = [f.stat().st_mtime for f in all_files]
            info['oldest_file'] = datetime.fromtimestamp(min(mtimes))
            info['newest_file'] = datetime.fromtimestamp(max(mtimes))
        
        return info
    
    def print_cache_info(self):
        """Print cache statistics"""
        info = self.get_cache_info()
        
        logger.info("=" * 80)
        logger.info("📊 CACHE STATISTICS")
        logger.info("=" * 80)
        logger.info(f"   Total files: {info['total_files']}")
        logger.info(f"   Total size: {info['total_size_mb']:.2f} MB")
        logger.info(f"   Raw data: {info['raw_files']} files")
        logger.info(f"   Cleaned data: {info['cleaned_files']} files")
        logger.info(f"   Features: {info['features_files']} files")
        
        if info['oldest_file']:
            logger.info(f"   Oldest: {info['oldest_file'].strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"   Newest: {info['newest_file'].strftime('%Y-%m-%d %H:%M')}")
        
        logger.info("=" * 80)
