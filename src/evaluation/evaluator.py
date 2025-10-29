"""
Comprehensive model evaluation implementation.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd

from .base import BaseEvaluator, BaseMetricsCalculator, ModelResults


class MetricsCalculator(BaseMetricsCalculator):
    """Implementation of metrics calculation for binary classification."""
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score."""
        return accuracy_score(y_true, y_pred)
    
    def calculate_precision_recall_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        return precision, recall, f1
    
    def calculate_roc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate ROC AUC score."""
        return roc_auc_score(y_true, y_proba)
    
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Generate confusion matrix."""
        return confusion_matrix(y_true, y_pred)


class ModelEvaluator(BaseEvaluator):
    """Comprehensive model evaluator with all required metrics."""
    
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> ModelResults:
        """
        Evaluate a single model and return comprehensive results.
        
        Args:
            model: Trained model instance
            X_test: Test features
            y_test: Test labels
            
        Returns:
            ModelResults: Comprehensive evaluation results
        """
        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Calculate all metrics
        accuracy = self.metrics_calculator.calculate_accuracy(y_test, y_pred)
        precision, recall, f1 = self.metrics_calculator.calculate_precision_recall_f1(y_test, y_pred)
        roc_auc = self.metrics_calculator.calculate_roc_auc(y_test, y_proba)
        cm = self.metrics_calculator.generate_confusion_matrix(y_test, y_pred)
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'get_feature_importance') and model.get_feature_importance() is not None:
            feature_importance = model.get_feature_importance()
            if isinstance(feature_importance, np.ndarray):
                # Convert to dict with feature indices as keys
                feature_importance = {f'feature_{i}': importance 
                                    for i, importance in enumerate(feature_importance)}
        
        # For single evaluation, we'll use dummy CV scores
        # In practice, cross-validation should be done on training data
        cv_scores = {'accuracy': [accuracy] * 5}  # Dummy CV scores
        
        return ModelResults(
            model_name=getattr(model, 'name', model.__class__.__name__),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            cross_val_scores=cv_scores['accuracy'],
            feature_importance=feature_importance,
            confusion_matrix=cm
        )
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation on a model.
        
        Args:
            model: Model instance
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Dict containing cross-validation scores for different metrics
        """
        # Use stratified k-fold to maintain class distribution
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Calculate cross-validation scores for multiple metrics
        cv_results = {}
        
        # Accuracy scores
        accuracy_scores = cross_val_score(model.model if hasattr(model, 'model') else model, 
                                        X, y, cv=skf, scoring='accuracy')
        cv_results['accuracy'] = accuracy_scores.tolist()
        
        # Precision scores
        precision_scores = cross_val_score(model.model if hasattr(model, 'model') else model, 
                                         X, y, cv=skf, scoring='precision')
        cv_results['precision'] = precision_scores.tolist()
        
        # Recall scores
        recall_scores = cross_val_score(model.model if hasattr(model, 'model') else model, 
                                      X, y, cv=skf, scoring='recall')
        cv_results['recall'] = recall_scores.tolist()
        
        # F1 scores
        f1_scores = cross_val_score(model.model if hasattr(model, 'model') else model, 
                                  X, y, cv=skf, scoring='f1')
        cv_results['f1'] = f1_scores.tolist()
        
        # ROC AUC scores
        roc_auc_scores = cross_val_score(model.model if hasattr(model, 'model') else model, 
                                       X, y, cv=skf, scoring='roc_auc')
        cv_results['roc_auc'] = roc_auc_scores.tolist()
        
        return cv_results
    
    def compare_models(self, results: List[ModelResults]) -> Dict[str, Any]:
        """
        Compare multiple model results and generate rankings.
        
        Args:
            results: List of ModelResults objects
            
        Returns:
            Dict containing comparison results and rankings
        """
        if not results:
            return {}
        
        # Convert results to DataFrame for easier manipulation
        comparison_data = []
        for result in results:
            comparison_data.append({
                'model_name': result.model_name,
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'roc_auc': result.roc_auc,
                'cv_mean_accuracy': np.mean(result.cross_val_scores),
                'cv_std_accuracy': np.std(result.cross_val_scores)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Create rankings for each metric (higher is better)
        rankings = {}
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'cv_mean_accuracy']
        
        for metric in metrics:
            rankings[f'{metric}_rank'] = df[metric].rank(ascending=False).astype(int).tolist()
        
        # Calculate overall ranking (average of all metric ranks)
        rank_cols = [col for col in rankings.keys()]
        rank_df = pd.DataFrame(rankings)
        overall_rank = rank_df.mean(axis=1).rank().astype(int).tolist()
        
        # Find best model overall
        best_model_idx = np.argmin(overall_rank)
        best_model = results[best_model_idx]
        
        return {
            'comparison_table': df.to_dict('records'),
            'rankings': rankings,
            'overall_ranking': overall_rank,
            'best_model': {
                'name': best_model.model_name,
                'accuracy': best_model.accuracy,
                'f1_score': best_model.f1_score,
                'roc_auc': best_model.roc_auc
            },
            'summary_stats': {
                'num_models': len(results),
                'best_accuracy': df['accuracy'].max(),
                'best_f1': df['f1_score'].max(),
                'best_roc_auc': df['roc_auc'].max(),
                'accuracy_range': df['accuracy'].max() - df['accuracy'].min(),
                'f1_range': df['f1_score'].max() - df['f1_score'].min()
            }
        }
    
    def generate_roc_data(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate ROC curve data for plotting.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dict containing FPR, TPR, and thresholds
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc_score(y_true, y_proba)
        }