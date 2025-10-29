"""
Base classes and interfaces for model evaluation components.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelResults:
    """Data class for storing model evaluation results."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cross_val_scores: List[float]
    feature_importance: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'cross_val_mean': np.mean(self.cross_val_scores),
            'cross_val_std': np.std(self.cross_val_scores),
            'feature_importance': self.feature_importance,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None
        }


@dataclass
class PredictionResult:
    """Data class for storing prediction results."""
    prediction: bool
    probability: float
    confidence_score: float
    feature_contributions: Optional[Dict[str, float]] = None


class BaseEvaluator(ABC):
    """Abstract base class for model evaluation."""
    
    @abstractmethod
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> ModelResults:
        """Evaluate a single model and return results."""
        pass
    
    @abstractmethod
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation on a model."""
        pass
    
    @abstractmethod
    def compare_models(self, results: List[ModelResults]) -> Dict[str, Any]:
        """Compare multiple model results."""
        pass


class BaseMetricsCalculator(ABC):
    """Abstract base class for calculating evaluation metrics."""
    
    @abstractmethod
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score."""
        pass
    
    @abstractmethod
    def calculate_precision_recall_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        pass
    
    @abstractmethod
    def calculate_roc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate ROC AUC score."""
        pass
    
    @abstractmethod
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Generate confusion matrix."""
        pass