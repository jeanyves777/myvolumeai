"""
Train ML Ensemble for Crypto Scalping Strategy

This script trains the ML ensemble models using historical crypto data.

Usage:
    python train_ml_ensemble.py --data-file crypto_data.csv --symbols "BTC/USD,ETH/USD"
    
    Or use existing backtest data:
    python train_ml_ensemble.py --use-backtest-data
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.ml.training import EnsembleTrainer, DataPipeline
from trading_system.ml import EnsemblePredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train ML Ensemble for Trading')
    
    # Data source
    parser.add_argument(
        '--data-file',
        type=str,
        help='Path to CSV file with OHLCV data'
    )
    parser.add_argument(
        '--use-backtest-data',
        action='store_true',
        help='Use existing backtest data from crypto_backtest_*.json files'
    )
    
    # Training options
    parser.add_argument(
        '--symbols',
        type=str,
        default='BTC/USD,ETH/USD,SOL/USD',
        help='Comma-separated list of symbols to train on'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='crypto_scalping_ensemble',
        help='Name for the trained model'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Optimize hyperparameters (takes longer)'
    )
    parser.add_argument(
        '--lookahead',
        type=int,
        default=5,
        help='Bars to look ahead for labeling (default: 5)'
    )
    parser.add_argument(
        '--profit-threshold',
        type=float,
        default=0.5,
        help='Minimum profit %% for positive label (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.data_file and not args.use_backtest_data:
        parser.error("Must specify either --data-file or --use-backtest-data")
    
    logger.info("=" * 70)
    logger.info("ML ENSEMBLE TRAINING FOR CRYPTO SCALPING")
    logger.info("=" * 70)
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Model Name: {args.model_name}")
    logger.info(f"Lookahead: {args.lookahead} bars")
    logger.info(f"Profit Threshold: {args.profit_threshold}%")
    logger.info(f"Hyperparameter Optimization: {args.optimize}")
    logger.info("")
    
    # Create trainer
    trainer = EnsembleTrainer(output_dir=args.output_dir)
    trainer.data_pipeline.lookahead = args.lookahead
    trainer.data_pipeline.profit_threshold = args.profit_threshold
    
    # Train model
    if args.use_backtest_data:
        logger.info("📊 Using backtest data...")
        logger.warning("⚠️  Feature: Loading from backtest data not yet fully implemented")
        logger.warning("⚠️  Please provide --data-file with OHLCV CSV data for now")
        logger.info("")
        logger.info("Expected CSV format:")
        logger.info("  timestamp,open,high,low,close,volume")
        logger.info("  2024-01-01 00:00:00,42000.0,42100.0,41900.0,42050.0,1000")
        logger.info("  ...")
        sys.exit(1)
    else:
        logger.info(f"📊 Loading data from {args.data_file}...")
        
        # Check if file exists
        if not Path(args.data_file).exists():
            logger.error(f"❌ Data file not found: {args.data_file}")
            sys.exit(1)
        
        # Train ensemble
        try:
            ensemble = trainer.train_from_file(
                data_path=args.data_file,
                optimize_hyperparams=args.optimize,
                model_name=args.model_name
            )
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"Model saved to: {args.output_dir}/{args.model_name}.pkl")
            logger.info("")
            logger.info("Next steps:")
            logger.info(f"  1. Test the model: python evaluate_ml_ensemble.py --model {args.output_dir}/{args.model_name}.pkl")
            logger.info(f"  2. Run backtest with ML: python run_crypto_backtest.py --use-ml")
            logger.info("")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
