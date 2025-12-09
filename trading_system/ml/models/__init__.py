"""
ML Models module.

ALL models are lazy-loaded to avoid slow sklearn/xgboost/TensorFlow imports.
"""
print(">>> [models/__init__.py] Loading models module (all lazy)...", flush=True)

# Lazy model classes - don't import until needed
_RandomForestModel = None
_XGBoostModel = None
_LogisticModel = None
_SVMModel = None

def _get_random_forest():
    global _RandomForestModel
    if _RandomForestModel is None:
        print(">>> [models/__init__.py] Importing RandomForestModel...", flush=True)
        from .random_forest_model import RandomForestModel
        _RandomForestModel = RandomForestModel
    return _RandomForestModel

def _get_xgboost():
    global _XGBoostModel
    if _XGBoostModel is None:
        print(">>> [models/__init__.py] Importing XGBoostModel...", flush=True)
        from .xgboost_model import XGBoostModel
        _XGBoostModel = XGBoostModel
    return _XGBoostModel

def _get_logistic():
    global _LogisticModel
    if _LogisticModel is None:
        print(">>> [models/__init__.py] Importing LogisticModel...", flush=True)
        from .logistic_model import LogisticModel
        _LogisticModel = LogisticModel
    return _LogisticModel

def _get_svm():
    global _SVMModel
    if _SVMModel is None:
        print(">>> [models/__init__.py] Importing SVMModel...", flush=True)
        from .svm_model import SVMModel
        _SVMModel = SVMModel
    return _SVMModel

def get_lstm_model():
    """Get LSTMModel class (lazy load to avoid TensorFlow startup delay)."""
    print(">>> [models/__init__.py] Importing LSTM (TensorFlow)...", flush=True)
    from .lstm_model import LSTMModel
    return LSTMModel

# For backward compatibility, expose classes that lazy-load on first access
class _LazyLoader:
    """Lazy loader that imports the real class on first use."""
    def __init__(self, loader_func):
        self._loader = loader_func
        self._class = None

    def __call__(self, *args, **kwargs):
        if self._class is None:
            self._class = self._loader()
        return self._class(*args, **kwargs)

    def __getattr__(self, name):
        if self._class is None:
            self._class = self._loader()
        return getattr(self._class, name)

# Create lazy wrappers
RandomForestModel = _LazyLoader(_get_random_forest)
XGBoostModel = _LazyLoader(_get_xgboost)
LogisticModel = _LazyLoader(_get_logistic)
SVMModel = _LazyLoader(_get_svm)

print(">>> [models/__init__.py] Module ready (sklearn/xgboost deferred)", flush=True)

__all__ = [
    'RandomForestModel',
    'XGBoostModel',
    'LogisticModel',
    'SVMModel',
    'get_lstm_model',
]
