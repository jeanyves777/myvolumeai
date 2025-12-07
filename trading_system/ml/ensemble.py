"""
Ensemble prediction system with weighted voting.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from .base import BaseMLModel
from .models import (
    RandomForestModel,
    XGBoostModel,
    LSTMModel,
    LogisticModel,
    SVMModel
)

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor using weighted voting from multiple ML models.
    
    Combines predictions from 5 models:
    - Random Forest (20%)
    - XGBoost (25%)
    - LSTM (20%)
    - Logistic Regression (15%)
    - SVM (20%)
    """
    
    def __init__(self, models: Optional[List[BaseMLModel]] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            models: List of trained models. If None, creates default models.
        """
        if models is None:
            self.models = self._create_default_models()
        else:
            self.models = models
        
        # Normalize weights to sum to 1.0
        self._normalize_weights()
        
        self.is_trained = False
        
    def _create_default_models(self) -> List[BaseMLModel]:
        """Create default ensemble models, skipping those that can't be loaded."""
        models = []

        # Try to create each model, skip if dependency missing
        try:
            models.append(RandomForestModel())  # 20%
            logger.info("  ✅ RandomForest loaded")
        except ImportError as e:
            logger.warning(f"  ⚠️ RandomForest skipped: {e}")

        try:
            models.append(XGBoostModel())       # 25%
            logger.info("  ✅ XGBoost loaded")
        except ImportError as e:
            logger.warning(f"  ⚠️ XGBoost skipped: {e}")

        try:
            models.append(LSTMModel())          # 20%
            logger.info("  ✅ LSTM loaded")
        except ImportError as e:
            logger.warning(f"  ⚠️ LSTM skipped (TensorFlow not installed): {e}")

        try:
            models.append(LogisticModel())      # 15%
            logger.info("  ✅ Logistic loaded")
        except ImportError as e:
            logger.warning(f"  ⚠️ Logistic skipped: {e}")

        try:
            models.append(SVMModel())            # 20%
            logger.info("  ✅ SVM loaded")
        except ImportError as e:
            logger.warning(f"  ⚠️ SVM skipped: {e}")

        if not models:
            raise ImportError("No ML models could be loaded. Install at least scikit-learn.")

        logger.info(f"  Ensemble using {len(models)} models")
        return models
    
    def _normalize_weights(self):
        """Normalize model weights to sum to 1.0."""
        total_weight = sum(model.config.weight for model in self.models)
        
        if total_weight > 0:
            for model in self.models:
                model.config.weight = model.config.weight / total_weight
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Dict[str, float]]:
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of training metrics for each model
        """
        all_metrics = {}
        
        logger.info(f"Training ensemble with {len(self.models)} models...")
        
        for i, model in enumerate(self.models):
            model_name = model.config.model_name
            logger.info(f"Training model {i+1}/{len(self.models)}: {model_name}")
            
            try:
                metrics = model.train(X_train, y_train, X_val, y_val)
                all_metrics[model_name] = metrics
                
                # Log validation accuracy if available
                if 'val_accuracy' in metrics:
                    logger.info(f"{model_name} validation accuracy: {metrics['val_accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                # Continue training other models
                all_metrics[model_name] = {'error': str(e)}
        
        self.is_trained = all(model.is_trained for model in self.models)
        
        if self.is_trained:
            logger.info("✅ All models trained successfully")
        else:
            logger.warning("⚠️ Some models failed to train")
        
        return all_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions using weighted voting.
        
        Args:
            X: Features to predict on
            
        Returns:
            Binary predictions (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble probability estimates using weighted averaging.
        
        Args:
            X: Features to predict on
            
        Returns:
            Probability estimates for positive class (0-1)
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
        
        weighted_predictions = np.zeros(len(X))
        total_weight = 0.0
        
        for model in self.models:
            if not model.is_trained:
                continue
            
            try:
                # Get model predictions
                model_proba = model.predict_proba(X)
                
                # Add weighted prediction
                weight = model.config.weight
                weighted_predictions += weight * model_proba
                total_weight += weight
                
            except Exception as e:
                logger.warning(f"Model {model.config.model_name} prediction failed: {str(e)}")
                continue
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_predictions /= total_weight
        
        return weighted_predictions
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get predictions with confidence levels.
        
        Args:
            X: Features to predict on
            
        Returns:
            Tuple of (predictions, probabilities, confidence_labels)
        """
        proba = self.predict_proba(X)
        pred = (proba > 0.5).astype(int)
        
        # Determine confidence levels
        confidence = []
        for p in proba:
            if p >= 0.70:
                confidence.append('VERY_HIGH')
            elif p >= 0.60:
                confidence.append('HIGH')
            elif p >= 0.50:
                confidence.append('MODERATE')
            elif p >= 0.40:
                confidence.append('LOW')
            else:
                confidence.append('VERY_LOW')
        
        return pred, proba, confidence
    
    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Dictionary mapping model names to their predictions
        """
        predictions = {}
        
        for model in self.models:
            if model.is_trained:
                try:
                    predictions[model.config.model_name] = model.predict_proba(X)
                except Exception as e:
                    logger.warning(f"Failed to get predictions from {model.config.model_name}: {str(e)}")
        
        return predictions
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate ensemble performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        pred = self.predict(X_test)
        proba = self.predict_proba(X_test)
        
        metrics = {
            'ensemble_accuracy': accuracy_score(y_test, pred),
            'ensemble_precision': precision_score(y_test, pred, zero_division=0),
            'ensemble_recall': recall_score(y_test, pred, zero_division=0),
            'ensemble_f1_score': f1_score(y_test, pred, zero_division=0),
            'ensemble_roc_auc': roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        return metrics
    
    def save(self, path: str):
        """Save the ensemble."""
        import joblib
        import os
        
        if not self.is_trained:
            raise ValueError("Cannot save untrained ensemble")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save each model
        base_path = path.replace('.pkl', '')
        for i, model in enumerate(self.models):
            model_path = f"{base_path}_{model.config.model_name}.pkl"
            model.save(model_path)
        
        # Save ensemble metadata
        ensemble_data = {
            'model_names': [m.config.model_name for m in self.models],
            'model_weights': [m.config.weight for m in self.models]
        }
        
        joblib.dump(ensemble_data, path)
        
        logger.info(f"✅ Ensemble saved to {path}")
    
    def load(self, path: str):
        """Load a saved ensemble."""
        import joblib
        
        # Load ensemble metadata
        ensemble_data = joblib.load(path)
        
        # Load each model
        base_path = path.replace('.pkl', '')
        models = []
        
        for model_name in ensemble_data['model_names']:
            model_path = f"{base_path}_{model_name}.pkl"
            
            # Create appropriate model instance
            if model_name == 'random_forest':
                model = RandomForestModel()
            elif model_name == 'xgboost':
                model = XGBoostModel()
            elif model_name == 'lstm':
                model = LSTMModel()
            elif model_name == 'logistic':
                model = LogisticModel()
            elif model_name == 'svm':
                model = SVMModel()
            else:
                logger.warning(f"Unknown model type: {model_name}")
                continue
            
            try:
                model.load(model_path)
                models.append(model)
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {str(e)}")
        
        self.models = models
        self._normalize_weights()
        self.is_trained = all(model.is_trained for model in self.models)
        
        logger.info(f"✅ Ensemble loaded from {path}")
