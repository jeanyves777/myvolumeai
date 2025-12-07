"""
Model evaluation and analysis tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path

from ..ensemble import EnsemblePredictor

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluator for trained ML models.
    
    Provides:
    - Performance metrics
    - Confusion matrix analysis
    - Feature importance
    - Prediction distribution
    """
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate_ensemble(self, ensemble: EnsemblePredictor,
                         X_test: pd.DataFrame,
                         y_test: pd.Series) -> Dict:
        """
        Comprehensive evaluation of ensemble.
        
        Args:
            ensemble: Trained ensemble predictor
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Evaluating ensemble...")
        
        # Get predictions
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)
        _, _, confidence = ensemble.predict_with_confidence(X_test)
        
        # Basic metrics
        metrics = ensemble.evaluate(X_test, y_test)
        
        # Confusion matrix
        confusion = self._compute_confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = confusion
        
        # Confidence distribution
        confidence_dist = pd.Series(confidence).value_counts().to_dict()
        metrics['confidence_distribution'] = confidence_dist
        
        # Prediction threshold analysis
        threshold_metrics = self._analyze_thresholds(y_test, y_proba)
        metrics['threshold_analysis'] = threshold_metrics
        
        # Individual model contributions
        individual_preds = ensemble.get_individual_predictions(X_test)
        metrics['individual_models'] = {}
        
        for model_name, proba in individual_preds.items():
            pred = (proba > 0.5).astype(int)
            from sklearn.metrics import accuracy_score, f1_score
            metrics['individual_models'][model_name] = {
                'accuracy': accuracy_score(y_test, pred),
                'f1_score': f1_score(y_test, pred, zero_division=0)
            }
        
        self._log_evaluation_results(metrics)
        
        return metrics
    
    def _compute_confusion_matrix(self, y_true: pd.Series,
                                  y_pred: np.ndarray) -> Dict:
        """Compute confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total': int(tn + fp + fn + tp)
        }
    
    def _analyze_thresholds(self, y_true: pd.Series,
                           y_proba: np.ndarray) -> Dict:
        """Analyze performance at different probability thresholds."""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        thresholds = [0.4, 0.5, 0.6, 0.7]
        results = {}
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            results[f'threshold_{threshold}'] = {
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'positive_predictions': int(y_pred.sum())
            }
        
        return results
    
    def analyze_feature_importance(self, ensemble: EnsemblePredictor,
                                   top_n: int = 20) -> pd.DataFrame:
        """
        Analyze feature importance across models.
        
        Args:
            ensemble: Trained ensemble
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance scores
        """
        importance_scores = []
        
        for model in ensemble.models:
            if not model.is_trained:
                continue
            
            try:
                # Try to get feature importance (for tree-based models)
                if hasattr(model, 'get_feature_importance'):
                    importance = model.get_feature_importance()
                    importance_scores.append(importance.rename(model.config.model_name))
            except:
                pass
        
        if not importance_scores:
            logger.warning("No feature importance available")
            return pd.DataFrame()
        
        # Combine importance scores
        importance_df = pd.DataFrame(importance_scores).T
        importance_df['mean_importance'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('mean_importance', ascending=False)
        
        logger.info(f"\nTop {top_n} Most Important Features:")
        logger.info("-" * 60)
        for i, (feature, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
            logger.info(f"{i:2d}. {feature:30s} {row['mean_importance']:.4f}")
        
        return importance_df.head(top_n)
    
    def backtest_predictions(self, ensemble: EnsemblePredictor,
                            features: pd.DataFrame,
                            bars: pd.DataFrame,
                            initial_capital: float = 10000.0) -> Dict:
        """
        Simple backtest of ML predictions.
        
        Args:
            ensemble: Trained ensemble
            features: Feature DataFrame
            bars: Original OHLCV data
            initial_capital: Starting capital
            
        Returns:
            Dictionary of backtest results
        """
        logger.info("Running ML prediction backtest...")
        
        predictions, probabilities, confidence = ensemble.predict_with_confidence(features)
        
        # Simple strategy: Buy when probability > 0.6
        entry_threshold = 0.6
        
        capital = initial_capital
        position = None
        trades = []
        
        for i in range(len(bars)):
            if position is None:
                # Look for entry
                if probabilities[i] >= entry_threshold:
                    position = {
                        'entry_idx': i,
                        'entry_price': bars['close'].iloc[i],
                        'entry_prob': probabilities[i],
                        'shares': capital / bars['close'].iloc[i]
                    }
            else:
                # Look for exit (after 5 bars or probability drops)
                bars_held = i - position['entry_idx']
                if bars_held >= 5 or probabilities[i] < 0.45:
                    exit_price = bars['close'].iloc[i]
                    pnl = (exit_price - position['entry_price']) * position['shares']
                    pnl_pct = (exit_price / position['entry_price'] - 1) * 100
                    
                    capital += pnl
                    
                    trades.append({
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'bars_held': bars_held
                    })
                    
                    position = None
        
        # Calculate metrics
        if trades:
            trades_df = pd.DataFrame(trades)
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            results = {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_return': (capital / initial_capital - 1) * 100,
                'num_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
                'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
                'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
                'avg_bars_held': trades_df['bars_held'].mean()
            }
            
            if results['avg_loss'] != 0:
                results['profit_factor'] = abs(results['avg_win'] / results['avg_loss'])
            else:
                results['profit_factor'] = float('inf') if results['avg_win'] > 0 else 0
            
            logger.info("\nBacktest Results:")
            logger.info(f"  Total Return: {results['total_return']:.2f}%")
            logger.info(f"  Num Trades: {results['num_trades']}")
            logger.info(f"  Win Rate: {results['win_rate']:.1f}%")
            logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
            
            return results
        else:
            logger.warning("No trades executed in backtest")
            return {'error': 'No trades'}
    
    def _log_evaluation_results(self, metrics: Dict):
        """Log evaluation results."""
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        
        # Main metrics
        logger.info("\nEnsemble Metrics:")
        for key in ['ensemble_accuracy', 'ensemble_precision', 'ensemble_recall',
                    'ensemble_f1_score', 'ensemble_roc_auc']:
            if key in metrics:
                logger.info(f"  {key}: {metrics[key]:.4f}")
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            logger.info("\nConfusion Matrix:")
            logger.info(f"  True Positives:  {cm['true_positives']:5d}")
            logger.info(f"  False Positives: {cm['false_positives']:5d}")
            logger.info(f"  True Negatives:  {cm['true_negatives']:5d}")
            logger.info(f"  False Negatives: {cm['false_negatives']:5d}")
        
        # Confidence distribution
        if 'confidence_distribution' in metrics:
            logger.info("\nConfidence Distribution:")
            for conf, count in metrics['confidence_distribution'].items():
                logger.info(f"  {conf}: {count}")
        
        # Individual models
        if 'individual_models' in metrics:
            logger.info("\nIndividual Model Performance:")
            for model_name, model_metrics in metrics['individual_models'].items():
                logger.info(f"  {model_name}:")
                logger.info(f"    Accuracy: {model_metrics['accuracy']:.4f}")
                logger.info(f"    F1 Score: {model_metrics['f1_score']:.4f}")
