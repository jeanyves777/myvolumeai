"""
Generate Sample Training Data for ML Ensemble

Creates synthetic OHLCV data or downloads real crypto data for training.

Usage:
    # Download real data from Yahoo Finance
    python generate_training_data.py --symbol BTC-USD --days 180 --output training_data.csv
    
    # Generate synthetic data (for testing)
    python generate_training_data.py --synthetic --bars 10000 --output training_data.csv
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_crypto_data(symbol: str, days: int) -> pd.DataFrame:
    """Download real crypto data from Yahoo Finance."""
    try:
        import yfinance as yf
        
        logger.info(f"📥 Downloading {symbol} data for last {days} days...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval='1h'  # 1-hour bars
        )
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Rename columns to match our format
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Keep only required columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"✅ Downloaded {len(df)} bars")
        logger.info(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
        
    except ImportError:
        logger.error("❌ yfinance not installed. Install with: pip install yfinance")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to download data: {str(e)}")
        sys.exit(1)


def generate_synthetic_data(num_bars: int, start_price: float = 40000.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    logger.info(f"🔧 Generating {num_bars} synthetic bars...")
    
    np.random.seed(42)
    
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=num_bars),
        periods=num_bars,
        freq='1H'
    )
    
    # Generate random walk with trend
    returns = np.random.randn(num_bars) * 0.02  # 2% volatility
    trend = np.linspace(0, 0.3, num_bars)  # 30% upward trend
    prices = start_price * np.exp(np.cumsum(returns) + trend)
    
    # Generate OHLC from prices
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        volatility = close * 0.01  # 1% bar range
        
        open_price = close + np.random.randn() * volatility * 0.5
        high = max(open_price, close) + abs(np.random.randn()) * volatility
        low = min(open_price, close) - abs(np.random.randn()) * volatility
        volume = abs(np.random.randn() * 1000000 + 5000000)  # Random volume
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    logger.info(f"✅ Generated {len(df)} synthetic bars")
    logger.info(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate Training Data for ML Ensemble')
    
    # Data source
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC-USD',
        help='Yahoo Finance symbol (e.g., BTC-USD, ETH-USD)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=180,
        help='Number of days of data to download (default: 180)'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Generate synthetic data instead of downloading'
    )
    parser.add_argument(
        '--bars',
        type=int,
        default=10000,
        help='Number of bars for synthetic data (default: 10000)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='training_data.csv',
        help='Output CSV file (default: training_data.csv)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("GENERATE ML TRAINING DATA")
    logger.info("=" * 70)
    
    # Generate or download data
    if args.synthetic:
        df = generate_synthetic_data(args.bars)
    else:
        df = download_crypto_data(args.symbol, args.days)
    
    # Save to CSV
    logger.info(f"\n💾 Saving data to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    logger.info("✅ Data saved successfully")
    logger.info("")
    logger.info("Next step:")
    logger.info(f"  Train ML ensemble: python train_ml_ensemble.py --data-file {args.output}")
    logger.info("")


if __name__ == "__main__":
    main()
