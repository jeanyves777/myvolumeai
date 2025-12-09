"""
Ensemble trainer with hyperparameter optimization.
"""
print(">>> [trainer.py] Module loading...", flush=True)

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
import time
from pathlib import Path

print(">>> [trainer.py] Importing EnsemblePredictor...", flush=True)
from ..ensemble import EnsemblePredictor
print(">>> [trainer.py] Importing DataPipeline...", flush=True)
from .data_pipeline import DataPipeline
print(">>> [trainer.py] Module loaded OK", flush=True)

logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """
    Trainer for ML ensemble models.
    
    Handles:
    - Training workflow
    - Hyperparameter optimization (optional)
    - Model persistence
    - Performance tracking
    """
    
    def __init__(self, output_dir: str = "models"):
        """
        Initialize trainer.
        
        Args:
            output_dir: Directory to save trained models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_pipeline = DataPipeline()
        
    def train_from_file(self, data_path: str,
                       optimize_hyperparams: bool = False,
                       model_name: str = "ensemble") -> EnsemblePredictor:
        """
        Train ensemble from CSV file.

        Args:
            data_path: Path to CSV with OHLCV data
            optimize_hyperparams: Whether to optimize hyperparameters
            model_name: Name for saved model

        Returns:
            Trained EnsemblePredictor
        """
        overall_start = time.time()
        step_times = {}

        print(">>> [trainer.train_from_file] Starting...", flush=True)
        print(">>> [trainer] +=========================================================================+", flush=True)
        print(">>> [trainer] |                    ML ENSEMBLE TRAINING PIPELINE                      |", flush=True)
        print(">>> [trainer] +=========================================================================+", flush=True)
        logger.info("=" * 60)
        logger.info("STARTING ENSEMBLE TRAINING")
        logger.info("=" * 60)

        # Load and prepare data
        step_start = time.time()
        print(f">>> [trainer] Step 1/7: Loading data from CSV... [{time.strftime('%H:%M:%S')}]", flush=True)
        logger.info("\n📊 Loading data...")
        bars = self.data_pipeline.load_historical_data(data_path)
        step_times['load_data'] = time.time() - step_start
        print(f">>> [trainer] [OK] Step 1 done in {step_times['load_data']:.1f}s - Loaded {len(bars):,} rows", flush=True)

        step_start = time.time()
        print(f">>> [trainer] Step 2/7: Preparing features and labels... [{time.strftime('%H:%M:%S')}]", flush=True)
        logger.info("\n🔧 Preparing features and labels...")
        features, labels = self.data_pipeline.prepare_training_data(bars)
        step_times['prepare_data'] = time.time() - step_start
        print(f">>> [trainer] [OK] Step 2 done in {step_times['prepare_data']:.1f}s - Features shape: {features.shape}", flush=True)

        # Split data
        step_start = time.time()
        print(f">>> [trainer] Step 3/7: Splitting data... [{time.strftime('%H:%M:%S')}]", flush=True)
        logger.info("\n✂️ Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.data_pipeline.split_data(features, labels)
        step_times['split_data'] = time.time() - step_start
        print(f">>> [trainer] [OK] Step 3 done in {step_times['split_data']:.1f}s - Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}", flush=True)

        # Handle class imbalance - use random_oversample (fast) instead of SMOTE (slow)
        step_start = time.time()
        print(f">>> [trainer] Step 4/7: Handling class imbalance... [{time.strftime('%H:%M:%S')}]", flush=True)
        logger.info("\n⚖️ Handling class imbalance...")
        X_train, y_train = self.data_pipeline.handle_class_imbalance(
            X_train, y_train, method='random_oversample'  # Much faster than SMOTE!
        )
        step_times['smote'] = time.time() - step_start
        print(f">>> [trainer] [OK] Step 4 done in {step_times['smote']:.1f}s ({step_times['smote']/60:.1f} min) - Resampled: {len(X_train):,} samples", flush=True)

        # Create ensemble
        step_start = time.time()
        print(f">>> [trainer] Step 5/7: Creating ensemble predictor... [{time.strftime('%H:%M:%S')}]", flush=True)
        logger.info("\n🤖 Creating ensemble...")
        ensemble = EnsemblePredictor()
        step_times['create_ensemble'] = time.time() - step_start
        print(f">>> [trainer] [OK] Step 5 done in {step_times['create_ensemble']:.1f}s - Ensemble created", flush=True)

        # Train models
        step_start = time.time()
        print(f">>> [trainer] Step 6/7: Training models... [{time.strftime('%H:%M:%S')}]", flush=True)
        logger.info("\n🎓 Training models...")
        training_metrics = ensemble.train(X_train, y_train, X_val, y_val)
        step_times['train_models'] = time.time() - step_start
        print(f">>> [trainer] [OK] Step 6 done in {step_times['train_models']:.1f}s ({step_times['train_models']/60:.1f} min) - Training complete!", flush=True)

        # Evaluate on test set
        step_start = time.time()
        print(f">>> [trainer] Step 7/7: Evaluating on test set... [{time.strftime('%H:%M:%S')}]", flush=True)
        logger.info("\n📈 Evaluating on test set...")
        test_metrics = ensemble.evaluate(X_test, y_test)
        step_times['evaluate'] = time.time() - step_start
        print(f">>> [trainer] [OK] Step 7 done in {step_times['evaluate']:.1f}s - Evaluation complete!", flush=True)

        # Log results
        self._log_training_results(training_metrics, test_metrics)

        # Save model
        model_path = self.output_dir / f"{model_name}.pkl"
        print(f">>> [trainer] Saving model to {model_path}...", flush=True)
        logger.info(f"\n💾 Saving model to {model_path}...")
        ensemble.save(str(model_path))

        # Calculate total time
        total_time = time.time() - overall_start

        # Print summary
        print(">>> [trainer] +=========================================================================+", flush=True)
        print(">>> [trainer] |                    TRAINING COMPLETE - SUMMARY                        |", flush=True)
        print(">>> [trainer] +-------------------------------------------------------------------------+", flush=True)
        print(f">>> [trainer] |  Total time:      {total_time/60:>6.1f} minutes                                    |", flush=True)
        print(">>> [trainer] +-------------------------------------------------------------------------+", flush=True)
        for step_name, step_time in step_times.items():
            pct = (step_time / total_time) * 100
            print(f">>> [trainer] |  {step_name:15s}: {step_time:>6.1f}s ({pct:>4.1f}%)                              |", flush=True)
        print(">>> [trainer] +=========================================================================+", flush=True)

        logger.info("\n" + "=" * 60)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info("=" * 60)
        print(">>> [trainer.train_from_file] Done!", flush=True)

        return ensemble
    
    def train_from_dataframe(self, features: pd.DataFrame, labels: pd.Series,
                            model_name: str = "ensemble") -> EnsemblePredictor:
        """
        Train ensemble from prepared features and labels.
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            model_name: Name for saved model
            
        Returns:
            Trained EnsemblePredictor
        """
        logger.info("Training ensemble from DataFrame...")
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.data_pipeline.split_data(features, labels)
        
        # Handle class imbalance
        X_train, y_train = self.data_pipeline.handle_class_imbalance(
            X_train, y_train, method='smote'
        )
        
        # Create and train ensemble
        ensemble = EnsemblePredictor()
        training_metrics = ensemble.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        test_metrics = ensemble.evaluate(X_test, y_test)
        
        # Log results
        self._log_training_results(training_metrics, test_metrics)
        
        # Save model
        model_path = self.output_dir / f"{model_name}.pkl"
        ensemble.save(str(model_path))
        
        return ensemble
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series,
                                 n_trials: int = 50) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary of best hyperparameters
        """
        try:
            import optuna
            
            logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
            
            def objective(trial):
                # Define hyperparameter search space for each model
                # This is a simplified example - expand as needed
                
                from ..models import RandomForestModel
                from ..base import MLModelConfig
                
                config = MLModelConfig(
                    model_name="random_forest_optimized",
                    model_type="classifier",
                    weight=0.20,
                    hyperparameters={
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 5, 30),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                        'random_state': 42,
                        'n_jobs': -1,
                        'class_weight': 'balanced'
                    }
                )
                
                model = RandomForestModel(config)
                model.train(X_train, y_train, X_val, y_val)
                
                # Return validation F1 score
                metrics = model.evaluate(X_val, y_val)
                return metrics['f1_score']
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            logger.info(f"Best F1 score: {study.best_value:.4f}")
            logger.info(f"Best hyperparameters: {study.best_params}")
            
            return study.best_params
            
        except ImportError:
            logger.warning("Optuna not installed. Skipping hyperparameter optimization.")
            logger.warning("Install with: pip install optuna")
            return {}
    
    def _log_training_results(self, training_metrics: Dict, test_metrics: Dict):
        """Log training and test results."""
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 60)
        
        # Log individual model metrics
        for model_name, metrics in training_metrics.items():
            if 'error' in metrics:
                logger.error(f"❌ {model_name}: {metrics['error']}")
            else:
                logger.info(f"\n{model_name.upper()}:")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        logger.info(f"  {metric_name}: {value:.4f}")
        
        # Log ensemble metrics
        logger.info("\n" + "-" * 60)
        logger.info("ENSEMBLE TEST METRICS:")
        logger.info("-" * 60)
        for metric_name, value in test_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Determine if model is good enough
        accuracy = test_metrics.get('ensemble_accuracy', 0)
        f1_score = test_metrics.get('ensemble_f1_score', 0)
        
        if accuracy >= 0.60 and f1_score >= 0.55:
            logger.info("\n✅ Model performance is GOOD (accuracy >= 60%, F1 >= 55%)")
        elif accuracy >= 0.55 and f1_score >= 0.50:
            logger.info("\n⚠️ Model performance is ACCEPTABLE (accuracy >= 55%, F1 >= 50%)")
        else:
            logger.warning("\n❌ Model performance is POOR (accuracy < 55% or F1 < 50%)")
            logger.warning("Consider:")
            logger.warning("  - Collecting more training data")
            logger.warning("  - Adjusting profit threshold")
            logger.warning("  - Feature engineering improvements")
            logger.warning("  - Hyperparameter optimization")
