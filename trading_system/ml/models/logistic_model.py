"""
Logistic Regression Model for ensemble.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..base import BaseMLModel, MLModelConfig


class LogisticModel(BaseMLModel):
    """
    Logistic Regression model.
    
    Strengths:
    - Fast training and prediction
    - Interpretable (linear relationships)
    - Good baseline model
    - Regularization prevents overfitting
    """
    
    def __init__(self, config: Optional[MLModelConfig] = None):
        """Initialize Logistic Regression model."""
        if config is None:
            config = MLModelConfig(
                model_name="logistic",
                model_type="classifier",
                weight=0.15,
                hyperparameters={
                    'C': 1.0,  # Inverse regularization strength
                    'penalty': 'l2',
                    'solver': 'lbfgs',
                    'max_iter': 1000,
                    'random_state': 42,
                    'n_jobs': -1,
                    'class_weight': 'balanced'
                }
            )
        
        super().__init__(config)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """Train the Logistic Regression model."""
        
        # Store feature columns
        self.feature_columns = X_train.columns.tolist()
        
        # Logistic Regression requires scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize and train model
        self.model = LogisticRegression(**self.config.hyperparameters)
        self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Evaluate on training data
        train_metrics = self.evaluate(X_train, y_train)
        train_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
        
        # Evaluate on validation data if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
            train_metrics.update(val_metrics)
        
        self.training_metrics = train_metrics
        
        return train_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_prepared = self._prepare_features(X)
        return self.model.predict(X_prepared)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability estimates for positive class."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_prepared = self._prepare_features(X)
        # Return probability of positive class (index 1)
        return self.model.predict_proba(X_prepared)[:, 1]
    
    def get_coefficients(self) -> pd.Series:
        """Get model coefficients (feature weights)."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        coefficients = pd.Series(
            self.model.coef_[0],
            index=self.feature_columns
        ).sort_values(ascending=False, key=abs)
        
        return coefficients
