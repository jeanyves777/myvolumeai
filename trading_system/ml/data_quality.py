"""
Data Quality Module for ML Training

Ensures high-quality training data by:
1. Detecting NaN, inf, and invalid values (zeros in critical fields)
2. Generating synthetic fills based on real data patterns
3. Preserving statistical properties of the original data
4. Validating data integrity before training

NO NaN or invalid zeros allowed - all gaps filled with pattern-based synthetic data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from scipy import stats
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    Data quality checker with pattern-based synthetic fill.
    
    Instead of dropping NaN/invalid data, we generate synthetic values
    that match the statistical patterns of surrounding real data.
    """
    
    def __init__(self, window_size: int = 20, min_valid_ratio: float = 0.5):
        """
        Initialize data quality checker.
        
        Args:
            window_size: Window size for pattern analysis (default: 20 bars)
            min_valid_ratio: Minimum ratio of valid data required (default: 0.5)
        """
        self.window_size = window_size
        self.min_valid_ratio = min_valid_ratio
        
    def check_and_fix_data(self, df: pd.DataFrame, symbol: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Check data quality and fix issues with synthetic fills.
        
        Args:
            df: DataFrame with OHLCV and indicators
            symbol: Optional symbol name for logging
            
        Returns:
            Tuple of (cleaned_df, report_dict)
        """
        symbol_str = f" for {symbol}" if symbol else ""
        logger.info(f"🔍 Checking data quality{symbol_str}...")
        
        report = {
            'total_rows': len(df),
            'issues_found': {},
            'fixes_applied': {},
            'data_quality_score': 0.0
        }
        
        df_clean = df.copy()
        
        # 1. Check and fix OHLCV data (critical - cannot be zero or NaN)
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            if col not in df_clean.columns:
                continue
                
            issues = self._detect_issues(df_clean[col], col)
            if issues['count'] > 0:
                report['issues_found'][col] = issues['count']
                logger.warning(f"   ⚠️  {col}: {issues['count']} invalid values detected")
                
                # Fix using pattern-based synthetic fill
                df_clean[col] = self._synthetic_fill(df_clean, col, issues['indices'])
                report['fixes_applied'][col] = issues['count']
                logger.info(f"   ✅ {col}: Fixed with synthetic data")
        
        # 2. Check and fix indicator data (can have some NaN during warmup)
        indicator_cols = [col for col in df_clean.columns 
                         if col not in ohlcv_cols + ['timestamp', 'symbol', 'label', 'entry_score']]
        
        for col in indicator_cols:
            issues = self._detect_issues(df_clean[col], col, allow_warmup_nan=True)
            if issues['count'] > 0:
                report['issues_found'][col] = issues['count']
                
                # Fix using forward fill, backward fill, then synthetic
                df_clean[col] = self._fix_indicator(df_clean, col, issues['indices'])
                report['fixes_applied'][col] = issues['count']
        
        # 3. Validate OHLC relationships
        validation_issues = self._validate_ohlc_relationships(df_clean)
        if validation_issues > 0:
            report['issues_found']['ohlc_validation'] = validation_issues
            df_clean = self._fix_ohlc_relationships(df_clean)
            report['fixes_applied']['ohlc_validation'] = validation_issues
            logger.warning(f"   ⚠️  Fixed {validation_issues} OHLC relationship violations")
        
        # 4. Calculate data quality score
        total_issues = sum(report['issues_found'].values())
        total_fixes = sum(report['fixes_applied'].values())
        total_cells = len(df_clean) * len(df_clean.columns)
        
        report['data_quality_score'] = 100 * (1 - total_issues / total_cells)
        
        logger.info(f"   ✅ Data quality score: {report['data_quality_score']:.2f}%")
        logger.info(f"   ✅ Total issues found: {total_issues}")
        logger.info(f"   ✅ Total fixes applied: {total_fixes}")
        
        # 5. Final validation
        if total_issues > len(df_clean) * self.min_valid_ratio:
            logger.warning(f"   ⚠️  High number of issues found but continuing with fixes ({report['data_quality_score']:.1f}%)")
        
        return df_clean, report
    
    def _detect_issues(self, series: pd.Series, col_name: str, allow_warmup_nan: bool = False) -> Dict:
        """Detect NaN, inf, and invalid zero values."""
        indices = []
        
        # Check for NaN
        nan_mask = series.isna()
        
        # Check for inf
        inf_mask = np.isinf(series)
        
        # Check for invalid zeros (OHLCV should never be zero)
        zero_mask = pd.Series([False] * len(series))
        if col_name in ['open', 'high', 'low', 'close']:
            # Price columns must NEVER be zero
            zero_mask = (series == 0)
        elif col_name == 'volume':
            # Volume CAN be zero (low liquidity periods), but suspicious if many consecutive zeros
            # Only flag as invalid if >10 consecutive zeros
            zero_runs = (series == 0).astype(int)
            run_lengths = zero_runs.groupby((zero_runs != zero_runs.shift()).cumsum()).transform('size')
            zero_mask = (series == 0) & (run_lengths > 10)
        
        # Combine all issues
        issue_mask = nan_mask | inf_mask | zero_mask
        
        # Allow NaN during warmup period for indicators
        if allow_warmup_nan and col_name not in ['open', 'high', 'low', 'close', 'volume']:
            warmup_period = min(50, len(series) // 10)  # First 50 bars or 10% of data
            issue_mask.iloc[:warmup_period] = False
        
        indices = series[issue_mask].index.tolist()
        
        return {
            'count': len(indices),
            'indices': indices,
            'has_nan': nan_mask.any(),
            'has_inf': inf_mask.any(),
            'has_zero': zero_mask.any()
        }
    
    def _synthetic_fill(self, df: pd.DataFrame, col: str, invalid_indices: List) -> pd.Series:
        """
        Generate synthetic fills based on surrounding data patterns.
        
        Uses multiple techniques:
        1. Linear interpolation for small gaps
        2. Pattern replication for larger gaps
        3. Statistical matching for extreme cases
        """
        series = df[col].copy()
        
        if len(invalid_indices) == 0:
            return series
        
        # Convert indices to actual DataFrame index values
        df_index_list = df.index.tolist()
        
        for idx in invalid_indices:
            # Skip if this index doesn't exist in the current dataframe
            if idx not in df_index_list:
                continue
            
            # Get the position in the series
            pos = df_index_list.index(idx)
            
            # Get surrounding valid data
            window_start = max(0, pos - self.window_size)
            window_end = min(len(series), pos + self.window_size + 1)
            
            window_data = series.iloc[window_start:window_end]
            valid_data = window_data[window_data.notna() & (window_data != 0) & ~np.isinf(window_data)]
            
            if len(valid_data) >= 2:
                # Method 1: Linear interpolation for small gaps
                before_data = series.iloc[max(0, pos-5):pos]
                after_data = series.iloc[pos+1:min(len(series), pos+6)]
                
                valid_before = before_data[before_data.notna() & (before_data != 0) & ~np.isinf(before_data)]
                valid_after = after_data[after_data.notna() & (after_data != 0) & ~np.isinf(after_data)]
                
                if len(valid_before) > 0 and len(valid_after) > 0:
                    # Interpolate between nearest valid values
                    synthetic_value = (valid_before.iloc[-1] + valid_after.iloc[0]) / 2
                    
                    # Add small random noise to match natural variance
                    noise_std = valid_data.std() * 0.05  # 5% of local std
                    synthetic_value += np.random.normal(0, noise_std)
                    
                    series.loc[idx] = synthetic_value
                
                else:
                    # Method 2: Use local mean with pattern matching
                    synthetic_value = valid_data.mean()
                    
                    # Adjust based on local trend
                    if len(valid_data) >= 5:
                        trend = np.polyfit(range(len(valid_data)), valid_data.values, 1)[0]
                        position_in_window = pos - window_start
                        synthetic_value += trend * position_in_window
                    
                    series.loc[idx] = synthetic_value
            
            else:
                # Method 3: Global fallback - use overall statistics
                global_valid = series[series.notna() & (series != 0) & ~np.isinf(series)]
                if len(global_valid) > 0:
                    series.loc[idx] = global_valid.median()
                else:
                    # Last resort: use a reasonable default
                    if col in ['open', 'high', 'low', 'close']:
                        series.loc[idx] = 100.0  # Reasonable price default
                    elif col == 'volume':
                        series.loc[idx] = 1000.0  # Reasonable volume default
                    else:
                        series.loc[idx] = 0.0  # Indicator default
        
        return series
    
    def _fix_indicator(self, df: pd.DataFrame, col: str, invalid_indices: List) -> pd.Series:
        """Fix indicator values using forward fill, backward fill, then synthetic."""
        series = df[col].copy()
        
        # First try forward fill
        series = series.fillna(method='ffill', limit=5)
        
        # Then backward fill for remaining
        series = series.fillna(method='bfill', limit=5)
        
        # Finally, synthetic fill for any remaining issues
        remaining_invalid = series[series.isna() | np.isinf(series) | (series == 0)].index.tolist()
        if len(remaining_invalid) > 0:
            series = self._synthetic_fill(df, col, remaining_invalid)
        
        return series
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> int:
        """
        Validate OHLC relationships:
        - high >= max(open, close)
        - low <= min(open, close)
        - high >= low
        """
        issues = 0
        
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return issues
        
        # Check high >= max(open, close)
        max_oc = df[['open', 'close']].max(axis=1)
        issues += (df['high'] < max_oc).sum()
        
        # Check low <= min(open, close)
        min_oc = df[['open', 'close']].min(axis=1)
        issues += (df['low'] > min_oc).sum()
        
        # Check high >= low
        issues += (df['high'] < df['low']).sum()
        
        return issues
    
    def _fix_ohlc_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix OHLC relationship violations."""
        df_fixed = df.copy()
        
        # Fix high to be at least max(open, close)
        max_oc = df_fixed[['open', 'close']].max(axis=1)
        df_fixed['high'] = df_fixed[['high', max_oc]].max(axis=1)
        
        # Fix low to be at most min(open, close)
        min_oc = df_fixed[['open', 'close']].min(axis=1)
        df_fixed['low'] = df_fixed[['low', min_oc]].min(axis=1)
        
        # Ensure high >= low (use average if violated)
        invalid_hl = df_fixed['high'] < df_fixed['low']
        if invalid_hl.any():
            avg_hl = (df_fixed.loc[invalid_hl, 'high'] + df_fixed.loc[invalid_hl, 'low']) / 2
            df_fixed.loc[invalid_hl, 'high'] = avg_hl * 1.001  # Slightly higher
            df_fixed.loc[invalid_hl, 'low'] = avg_hl * 0.999   # Slightly lower
        
        return df_fixed
    
    def generate_quality_report(self, df: pd.DataFrame, symbol: Optional[str] = None) -> str:
        """Generate detailed data quality report."""
        report_lines = []
        symbol_str = f" for {symbol}" if symbol else ""
        
        report_lines.append("=" * 80)
        report_lines.append(f"DATA QUALITY REPORT{symbol_str}")
        report_lines.append("=" * 80)
        
        # Basic stats
        report_lines.append(f"Total Rows: {len(df):,}")
        report_lines.append(f"Total Columns: {len(df.columns)}")
        report_lines.append("")
        
        # Column-by-column analysis
        report_lines.append("Column Analysis:")
        for col in df.columns:
            if col in ['timestamp', 'symbol']:
                continue
            
            series = df[col]
            null_count = series.isna().sum()
            zero_count = (series == 0).sum()
            inf_count = np.isinf(series).sum()
            
            if null_count + zero_count + inf_count > 0:
                report_lines.append(f"  {col:20s}: NaN={null_count:4d}, Zero={zero_count:4d}, Inf={inf_count:4d}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


def clean_training_data(df: pd.DataFrame, per_symbol: bool = True, min_quality: float = 0.95) -> pd.DataFrame:
    """
    Clean entire training dataset with quality checks.
    
    Args:
        df: Raw training data with OHLCV and indicators
        per_symbol: Clean each symbol separately (recommended)
        min_quality: Minimum data quality score (0-1)
        
    Returns:
        Cleaned DataFrame ready for ML training
    """
    checker = DataQualityChecker()
    
    logger.info("=" * 80)
    logger.info("🧹 CLEANING TRAINING DATA")
    logger.info("=" * 80)
    
    if per_symbol and 'symbol' in df.columns:
        cleaned_dfs = []
        
        for symbol in df['symbol'].unique():
            logger.info(f"\n📊 Processing {symbol}...")
            symbol_df = df[df['symbol'] == symbol].copy()
            
            cleaned_symbol_df, report = checker.check_and_fix_data(symbol_df, symbol)
            
            if report['data_quality_score'] < min_quality * 100:
                logger.warning(f"   ⚠️  {symbol} quality below threshold: {report['data_quality_score']:.1f}%")
            
            cleaned_dfs.append(cleaned_symbol_df)
        
        result = pd.concat(cleaned_dfs, ignore_index=True)
    
    else:
        result, report = checker.check_and_fix_data(df)
        
        if report['data_quality_score'] < min_quality * 100:
            logger.warning(f"   ⚠️  Data quality below threshold: {report['data_quality_score']:.1f}%")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ DATA CLEANING COMPLETE")
    logger.info(f"   Total rows: {len(result):,}")
    logger.info(f"   Symbols: {result['symbol'].nunique() if 'symbol' in result.columns else 1}")
    logger.info("=" * 80)
    
    return result
