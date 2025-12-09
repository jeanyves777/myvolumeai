"""
LSTM Neural Network Model for ensemble.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

# Lazy import - TensorFlow is heavy and slow to load
TENSORFLOW_AVAILABLE = None
tf = None
keras = None
layers = None

def _check_tensorflow():
    """Lazy check for TensorFlow availability."""
    global TENSORFLOW_AVAILABLE, tf, keras, layers
    if TENSORFLOW_AVAILABLE is None:
        try:
            import tensorflow as _tf
            tf = _tf
            keras = tf.keras
            layers = tf.keras.layers
            TENSORFLOW_AVAILABLE = True
        except ImportError:
            TENSORFLOW_AVAILABLE = False
    return TENSORFLOW_AVAILABLE

from sklearn.preprocessing import StandardScaler

from ..base import BaseMLModel, MLModelConfig


class LSTMModel(BaseMLModel):
    """
    LSTM Neural Network model.

    Strengths:
    - Learns temporal patterns and sequences
    - Captures time-series dependencies
    - Good for momentum and trend detection
    """

    def __init__(self, config: Optional[MLModelConfig] = None):
        """Initialize LSTM model."""
        if not _check_tensorflow():
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
        
        if config is None:
            config = MLModelConfig(
                model_name="lstm",
                model_type="classifier",
                weight=0.20,
                hyperparameters={
                    'sequence_length': 10,  # Look back 10 time steps
                    'lstm_units': 64,
                    'dropout': 0.3,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 50,
                    'patience': 10,
                    'random_state': 42
                }
            )
        
        super().__init__(config)
        self.sequence_length = config.hyperparameters['sequence_length']
        
    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Create sequences for LSTM input."""
        X_seq = []
        y_seq = []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        
        return X_seq
    
    def _build_model(self, input_shape):
        """Build LSTM model architecture."""
        params = self.config.hyperparameters
        
        model = keras.Sequential([
            layers.LSTM(
                params['lstm_units'],
                input_shape=input_shape,
                return_sequences=True
            ),
            layers.Dropout(params['dropout']),
            layers.LSTM(params['lstm_units'] // 2),
            layers.Dropout(params['dropout']),
            layers.Dense(32, activation='relu'),
            layers.Dropout(params['dropout']),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """Train the LSTM model."""
        
        # Store feature columns
        self.feature_columns = X_train.columns.tolist()
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train.values)
        
        # Build model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self._build_model(input_shape)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val.values)
            validation_data = (X_val_seq, y_val_seq)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=self.config.hyperparameters['patience'],
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train_seq, y_train_seq,
            batch_size=self.config.hyperparameters['batch_size'],
            epochs=self.config.hyperparameters['epochs'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_trained = True
        
        # Get metrics from last epoch
        train_metrics = {
            'train_accuracy': history.history['accuracy'][-1],
            'train_auc': history.history['auc'][-1] if 'auc' in history.history else 0.0
        }
        
        if validation_data is not None:
            train_metrics['val_accuracy'] = history.history['val_accuracy'][-1]
            train_metrics['val_auc'] = history.history['val_auc'][-1] if 'val_auc' in history.history else 0.0
        
        self.training_metrics = train_metrics
        
        return train_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability estimates for positive class."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_prepared = self._prepare_features(X)
        
        # Create sequences
        X_seq = self._create_sequences(X_prepared)
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0).flatten()
        
        # Pad beginning with 0.5 (neutral) for sequence length
        padded_predictions = np.concatenate([
            np.full(self.sequence_length, 0.5),
            predictions
        ])
        
        return padded_predictions[:len(X)]
    
    def save(self, path: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        import joblib
        
        # Save Keras model separately
        keras_path = path.replace('.pkl', '_keras.h5')
        self.model.save(keras_path)
        
        # Save other components
        model_data = {
            'keras_model_path': keras_path,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'sequence_length': self.sequence_length
        }
        
        joblib.dump(model_data, path)
    
    def load(self, path: str):
        """Load a trained model."""
        import joblib
        
        model_data = joblib.load(path)
        
        # Load Keras model
        keras_path = model_data['keras_model_path']
        self.model = keras.models.load_model(keras_path)
        
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.training_metrics = model_data.get('training_metrics', {})
        self.sequence_length = model_data.get('sequence_length', 10)
        self.is_trained = True
