"""
Random Forest Model for ensemble.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from ..base import BaseMLModel, MLModelConfig

# Lazy sklearn imports
_sklearn_loaded = False
RandomForestClassifier = None
StandardScaler = None

def _load_sklearn():
    """Lazy load sklearn to avoid slow import at module load."""
    global _sklearn_loaded, RandomForestClassifier, StandardScaler
    if not _sklearn_loaded:
        print(">>> [random_forest_model] Loading sklearn...", flush=True)
        from sklearn.ensemble import RandomForestClassifier as RFC
        from sklearn.preprocessing import StandardScaler as SS
        RandomForestClassifier = RFC
        StandardScaler = SS
        _sklearn_loaded = True
        print(">>> [random_forest_model] sklearn loaded OK", flush=True)


class RandomForestModel(BaseMLModel):
    """
    Random Forest Classifier model.
    
    Strengths:
    - Handles non-linear patterns well
    - Provides feature importance
    - Resistant to overfitting
    - No need for feature scaling
    """
    
    def __init__(self, config: Optional[MLModelConfig] = None):
        """Initialize Random Forest model."""
        if config is None:
            config = MLModelConfig(
                model_name="random_forest",
                model_type="classifier",
                weight=0.20,
                hyperparameters={
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 10,
                    'min_samples_leaf': 4,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1,
                    'class_weight': 'balanced'
                }
            )
        
        super().__init__(config)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """Train the Random Forest model."""

        # Lazy load sklearn
        _load_sklearn()

        # Store feature columns
        self.feature_columns = X_train.columns.tolist()

        # Random Forest doesn't need scaling, but we'll do it for consistency
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize and train model
        self.model = RandomForestClassifier(**self.config.hyperparameters)
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
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_columns
        ).sort_values(ascending=False)
        
        return importance
