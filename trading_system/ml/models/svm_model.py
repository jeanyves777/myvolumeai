"""
Support Vector Machine Model for ensemble.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from ..base import BaseMLModel, MLModelConfig

# Lazy sklearn imports
_sklearn_loaded = False
SVC = None
StandardScaler = None

def _load_sklearn():
    """Lazy load sklearn to avoid slow import at module load."""
    global _sklearn_loaded, SVC, StandardScaler
    if not _sklearn_loaded:
        print(">>> [svm_model] Loading sklearn...", flush=True)
        from sklearn.svm import SVC as _SVC
        from sklearn.preprocessing import StandardScaler as SS
        SVC = _SVC
        StandardScaler = SS
        _sklearn_loaded = True
        print(">>> [svm_model] sklearn loaded OK", flush=True)


class SVMModel(BaseMLModel):
    """
    Support Vector Machine model.
    
    Strengths:
    - Finds optimal decision boundaries
    - Good for binary classification
    - Kernel trick for non-linear patterns
    - Robust to outliers
    """
    
    def __init__(self, config: Optional[MLModelConfig] = None):
        """Initialize SVM model."""
        if config is None:
            config = MLModelConfig(
                model_name="svm",
                model_type="classifier",
                weight=0.20,
                hyperparameters={
                    'C': 1.0,  # Regularization parameter
                    'kernel': 'rbf',  # Radial basis function
                    'gamma': 'scale',
                    'probability': True,  # Enable probability estimates
                    'random_state': 42,
                    'class_weight': 'balanced',
                    'max_iter': 1000
                }
            )
        
        super().__init__(config)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """Train the SVM model."""

        # Lazy load sklearn
        _load_sklearn()

        # Store feature columns
        self.feature_columns = X_train.columns.tolist()

        # SVM requires scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize and train model
        self.model = SVC(**self.config.hyperparameters)
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
    
    def get_support_vectors_count(self) -> int:
        """Get number of support vectors."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return len(self.model.support_vectors_)
