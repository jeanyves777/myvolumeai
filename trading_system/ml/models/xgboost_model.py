"""
XGBoost Model for ensemble.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from ..base import BaseMLModel, MLModelConfig

# Lazy imports
_libs_loaded = False
xgb = None
StandardScaler = None
XGBOOST_AVAILABLE = None

def _load_libs():
    """Lazy load xgboost and sklearn to avoid slow import at module load."""
    global _libs_loaded, xgb, StandardScaler, XGBOOST_AVAILABLE
    if not _libs_loaded:
        print(">>> [xgboost_model] Loading xgboost and sklearn...", flush=True)
        try:
            import xgboost as _xgb
            xgb = _xgb
            XGBOOST_AVAILABLE = True
        except ImportError:
            XGBOOST_AVAILABLE = False

        from sklearn.preprocessing import StandardScaler as SS
        StandardScaler = SS
        _libs_loaded = True
        print(">>> [xgboost_model] Libraries loaded OK", flush=True)


class XGBoostModel(BaseMLModel):
    """
    XGBoost Classifier model.
    
    Strengths:
    - Highest accuracy among tree models
    - Handles imbalanced data well
    - Built-in regularization
    - Fast training and prediction
    """
    
    def __init__(self, config: Optional[MLModelConfig] = None):
        """Initialize XGBoost model."""
        # Check availability lazily
        _load_libs()
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        if config is None:
            config = MLModelConfig(
                model_name="xgboost",
                model_type="classifier",
                weight=0.25,  # Highest weight - typically best performer
                hyperparameters={
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 3,
                    'gamma': 0.1,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': 42,
                    'n_jobs': -1,
                    'scale_pos_weight': 1.0,  # Adjust for class imbalance
                    'eval_metric': 'logloss'
                }
            )
        
        super().__init__(config)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """Train the XGBoost model."""
        
        # Store feature columns
        self.feature_columns = X_train.columns.tolist()
        
        # XGBoost benefits from scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Calculate scale_pos_weight for imbalanced data
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        # Update hyperparameters
        params = self.config.hyperparameters.copy()
        params['scale_pos_weight'] = scale_pos_weight
        
        # Initialize and train model
        self.model = xgb.XGBClassifier(**params)
        
        # Prepare eval set if validation data provided
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
        
        self.model.fit(
            X_train_scaled, 
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        
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
