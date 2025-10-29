"""
Evaluation module for model assessment and comparison.
"""
from .base import BaseEvaluator, BaseMetricsCalculator, ModelResults, PredictionResult
from .evaluator import ModelEvaluator, MetricsCalculator
from .cross_validator import CrossValidator, PerformanceAssessment
from .visualizer import EvaluationVisualizer

__all__ = [
    'BaseEvaluator',
    'BaseMetricsCalculator', 
    'ModelResults',
    'PredictionResult',
    'ModelEvaluator',
    'MetricsCalculator',
    'CrossValidator',
    'PerformanceAssessment',
    'EvaluationVisualizer'
]