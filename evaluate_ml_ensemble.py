"""
Evaluate ML Ensemble Model

This script evaluates a trained ML ensemble model and shows performance metrics.

Usage:
    python evaluate_ml_ensemble.py --model models/crypto_scalping_ensemble.pkl --test-data test_data.csv
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.ml import EnsemblePredictor
from trading_system.ml.training import ModelEvaluator, DataPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate ML Ensemble Model')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained ensemble model (.pkl file)'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        help='Path to test data CSV file (optional - will use train/test split if not provided)'
    )
    parser.add_argument(
        '--feature-importance',
        action='store_true',
        help='Show feature importance analysis'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run simple backtest of ML predictions'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("ML ENSEMBLE MODEL EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info("")
    
    # Load ensemble
    logger.info("📦 Loading ensemble...")
    try:
        ensemble = EnsemblePredictor()
        ensemble.load(args.model)
        logger.info("✅ Ensemble loaded successfully")
        logger.info(f"   Models: {len(ensemble.models)}")
        for model in ensemble.models:
            logger.info(f"     - {model.config.model_name} (weight: {model.config.weight:.2f})")
    except Exception as e:
        logger.error(f"❌ Failed to load ensemble: {str(e)}")
        sys.exit(1)
    
    # Load test data if provided
    if args.test_data:
        logger.info(f"\n📊 Loading test data from {args.test_data}...")
        
        if not Path(args.test_data).exists():
            logger.error(f"❌ Test data not found: {args.test_data}")
            sys.exit(1)
        
        try:
            # Load and prepare data
            pipeline = DataPipeline()
            bars = pipeline.load_historical_data(args.test_data)
            features, labels = pipeline.prepare_training_data(bars)
            
            # Evaluate
            logger.info("\n📈 Evaluating ensemble on test data...")
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_ensemble(ensemble, features, labels)
            
            # Feature importance
            if args.feature_importance:
                logger.info("\n🎯 Analyzing feature importance...")
                importance = evaluator.analyze_feature_importance(ensemble)
            
            # Backtest
            if args.backtest:
                logger.info("\n📊 Running ML prediction backtest...")
                backtest_results = evaluator.backtest_predictions(
                    ensemble, features, bars
                )
            
            logger.info("\n✅ Evaluation complete!")
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        logger.info("\nℹ️  No test data provided")
        logger.info("   Use --test-data to evaluate on specific data")
        logger.info("   Model is loaded and ready to use in strategies")


if __name__ == "__main__":
    main()
