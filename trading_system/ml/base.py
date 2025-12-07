"""
Base classes for ML models in the trading system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class MLModelConfig:
    """Configuration for ML models."""
    model_name: str
    model_type: str  # 'classifier' or 'regressor'
    hyperparameters: Dict[str, Any]
    weight: float  # Weight in ensemble (0-1)
    retrain_frequency: str = "weekly"  # How often to retrain
    min_accuracy: float = 0.55  # Minimum acceptable accuracy


class BaseMLModel(ABC):
    """
    Base class for all ML models in the ensemble.
    
    Each model must implement:
    - train(): Train the model on historical data
    - predict(): Make predictions on new data
    - predict_proba(): Get probability estimates
    - evaluate(): Evaluate model performance
    """
    
    def __init__(self, config: MLModelConfig):
        """
        Initialize the ML model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        self.training_metrics = {}
        
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions (0 or 1)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability estimates.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of probabilities for positive class
        """
        pass
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        return metrics
    
    def save(self, path: str):
        """Save the trained model."""
        import joblib
        
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'config': self.config,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, path)
        
    def load(self, path: str):
        """Load a trained model."""
        import joblib
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.training_metrics = model_data.get('training_metrics', {})
        self.is_trained = True
        
    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for prediction (scaling, etc.).
        
        Args:
            X: Raw features
            
        Returns:
            Prepared feature array
        """
        if self.feature_columns is not None:
            # Ensure columns match training
            X = X[self.feature_columns]
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
            
        return X_scaled
