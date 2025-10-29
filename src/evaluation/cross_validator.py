"""
Cross-validation and performance assessment implementation.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.metrics import make_scorer
from scipy import stats
import pandas as pd

from .base import ModelResults


class CrossValidator:
    """
    Advanced cross-validation implementation with statistical significance testing.
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize CrossValidator.
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv_strategy = StratifiedKFold(
            n_splits=cv_folds, 
            shuffle=True, 
            random_state=random_state
        )
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation on a single model.
        
        Args:
            model: Model instance to validate
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dict containing detailed cross-validation results
        """
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            model.model if hasattr(model, 'model') else model,
            X, y,
            cv=self.cv_strategy,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Calculate statistics for each metric
        results = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results[metric] = {
                'test_scores': test_scores.tolist(),
                'train_scores': train_scores.tolist(),
                'test_mean': np.mean(test_scores),
                'test_std': np.std(test_scores),
                'train_mean': np.mean(train_scores),
                'train_std': np.std(train_scores),
                'test_ci_lower': np.percentile(test_scores, 2.5),
                'test_ci_upper': np.percentile(test_scores, 97.5),
                'overfitting_score': np.mean(train_scores) - np.mean(test_scores)
            }
        
        # Add overall assessment
        results['overall'] = {
            'model_name': getattr(model, 'name', model.__class__.__name__),
            'cv_folds': self.cv_folds,
            'fit_time_mean': np.mean(cv_results['fit_time']),
            'score_time_mean': np.mean(cv_results['score_time']),
            'stability_score': self._calculate_stability_score(results)
        }
        
        return results
    
    def cross_validate_multiple_models(self, models: List, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform cross-validation on multiple models and compare results.
        
        Args:
            models: List of model instances
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dict containing results for all models and comparisons
        """
        all_results = {}
        
        # Cross-validate each model
        for model in models:
            model_name = getattr(model, 'name', model.__class__.__name__)
            all_results[model_name] = self.cross_validate_model(model, X, y)
        
        # Perform statistical comparisons
        comparisons = self._perform_statistical_comparisons(all_results)
        
        return {
            'individual_results': all_results,
            'statistical_comparisons': comparisons,
            'summary': self._generate_summary(all_results)
        }
    
    def _calculate_stability_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate a stability score based on standard deviations across metrics.
        Lower values indicate more stable performance.
        """
        stds = []
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric in results:
                stds.append(results[metric]['test_std'])
        
        return np.mean(stds) if stds else 0.0
    
    def _perform_statistical_comparisons(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical significance tests between models.
        """
        model_names = list(all_results.keys())
        comparisons = {}
        
        # Compare each pair of models
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                comparison_key = f"{model1}_vs_{model2}"
                comparisons[comparison_key] = {}
                
                # Compare each metric
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                    if metric in all_results[model1] and metric in all_results[model2]:
                        scores1 = all_results[model1][metric]['test_scores']
                        scores2 = all_results[model2][metric]['test_scores']
                        
                        # Perform paired t-test
                        t_stat, p_value = stats.ttest_rel(scores1, scores2)
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                        cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0
                        
                        comparisons[comparison_key][metric] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'cohens_d': cohens_d,
                            'effect_size': self._interpret_effect_size(abs(cohens_d)),
                            'better_model': model1 if np.mean(scores1) > np.mean(scores2) else model2,
                            'mean_difference': np.mean(scores1) - np.mean(scores2)
                        }
        
        return comparisons
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _generate_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all models."""
        summary = {
            'num_models': len(all_results),
            'cv_folds': self.cv_folds,
            'metrics_summary': {}
        }
        
        # Summarize each metric across all models
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            metric_values = []
            stability_values = []
            
            for model_results in all_results.values():
                if metric in model_results:
                    metric_values.append(model_results[metric]['test_mean'])
                    stability_values.append(model_results[metric]['test_std'])
            
            if metric_values:
                summary['metrics_summary'][metric] = {
                    'best_score': max(metric_values),
                    'worst_score': min(metric_values),
                    'mean_score': np.mean(metric_values),
                    'score_range': max(metric_values) - min(metric_values),
                    'most_stable_std': min(stability_values),
                    'least_stable_std': max(stability_values)
                }
        
        # Find best model for each metric
        best_models = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            best_score = -1
            best_model = None
            
            for model_name, model_results in all_results.items():
                if metric in model_results:
                    score = model_results[metric]['test_mean']
                    if score > best_score:
                        best_score = score
                        best_model = model_name
            
            if best_model:
                best_models[metric] = {
                    'model': best_model,
                    'score': best_score
                }
        
        summary['best_models_by_metric'] = best_models
        
        return summary


class PerformanceAssessment:
    """
    Advanced performance assessment with statistical analysis.
    """
    
    def __init__(self):
        self.cross_validator = CrossValidator()
    
    def assess_model_performance(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive performance assessment for a single model.
        
        Args:
            model: Model instance
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dict containing comprehensive performance assessment
        """
        # Perform cross-validation
        cv_results = self.cross_validator.cross_validate_model(model, X, y)
        
        # Additional assessments
        assessment = {
            'cross_validation': cv_results,
            'performance_grade': self._grade_performance(cv_results),
            'recommendations': self._generate_recommendations(cv_results),
            'risk_assessment': self._assess_risks(cv_results)
        }
        
        return assessment
    
    def _grade_performance(self, cv_results: Dict[str, Any]) -> Dict[str, str]:
        """Grade model performance on different aspects."""
        grades = {}
        
        # Grade accuracy
        acc_mean = cv_results['accuracy']['test_mean']
        if acc_mean >= 0.9:
            grades['accuracy'] = 'A'
        elif acc_mean >= 0.8:
            grades['accuracy'] = 'B'
        elif acc_mean >= 0.7:
            grades['accuracy'] = 'C'
        else:
            grades['accuracy'] = 'D'
        
        # Grade stability (based on standard deviation)
        stability_score = cv_results['overall']['stability_score']
        if stability_score <= 0.02:
            grades['stability'] = 'A'
        elif stability_score <= 0.05:
            grades['stability'] = 'B'
        elif stability_score <= 0.1:
            grades['stability'] = 'C'
        else:
            grades['stability'] = 'D'
        
        # Grade overfitting (difference between train and test)
        overfitting = cv_results['accuracy']['overfitting_score']
        if overfitting <= 0.05:
            grades['overfitting'] = 'A'
        elif overfitting <= 0.1:
            grades['overfitting'] = 'B'
        elif overfitting <= 0.2:
            grades['overfitting'] = 'C'
        else:
            grades['overfitting'] = 'D'
        
        return grades
    
    def _generate_recommendations(self, cv_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance."""
        recommendations = []
        
        # Check for overfitting
        overfitting = cv_results['accuracy']['overfitting_score']
        if overfitting > 0.1:
            recommendations.append(
                "High overfitting detected. Consider regularization or reducing model complexity."
            )
        
        # Check for instability
        stability_score = cv_results['overall']['stability_score']
        if stability_score > 0.05:
            recommendations.append(
                "Model shows high variance across folds. Consider ensemble methods or more data."
            )
        
        # Check for low performance
        acc_mean = cv_results['accuracy']['test_mean']
        if acc_mean < 0.7:
            recommendations.append(
                "Low accuracy detected. Consider feature engineering or different algorithms."
            )
        
        # Check for class imbalance issues
        precision = cv_results['precision']['test_mean']
        recall = cv_results['recall']['test_mean']
        if abs(precision - recall) > 0.2:
            recommendations.append(
                "Large precision-recall gap suggests class imbalance. Consider resampling techniques."
            )
        
        if not recommendations:
            recommendations.append("Model performance looks good across all metrics.")
        
        return recommendations
    
    def _assess_risks(self, cv_results: Dict[str, Any]) -> Dict[str, str]:
        """Assess deployment risks based on performance."""
        risks = {}
        
        # Stability risk
        stability_score = cv_results['overall']['stability_score']
        if stability_score > 0.1:
            risks['stability'] = 'HIGH - Model performance varies significantly across folds'
        elif stability_score > 0.05:
            risks['stability'] = 'MEDIUM - Some variation in performance across folds'
        else:
            risks['stability'] = 'LOW - Consistent performance across folds'
        
        # Overfitting risk
        overfitting = cv_results['accuracy']['overfitting_score']
        if overfitting > 0.2:
            risks['overfitting'] = 'HIGH - Significant gap between training and validation performance'
        elif overfitting > 0.1:
            risks['overfitting'] = 'MEDIUM - Some overfitting detected'
        else:
            risks['overfitting'] = 'LOW - Good generalization'
        
        # Performance risk
        acc_mean = cv_results['accuracy']['test_mean']
        if acc_mean < 0.6:
            risks['performance'] = 'HIGH - Low accuracy may lead to poor business outcomes'
        elif acc_mean < 0.8:
            risks['performance'] = 'MEDIUM - Moderate accuracy, monitor closely'
        else:
            risks['performance'] = 'LOW - Good accuracy for deployment'
        
        return risks