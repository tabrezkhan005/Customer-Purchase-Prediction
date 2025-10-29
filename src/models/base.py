"""
Base classes and interfaces for machine learning models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import pickle
import os


class BaseModel(ABC):
    """Abstract base class for all machine learning models."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.is_trained = False
        self.hyperparameters = kwargs
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model with given training data."""
        pass
    
    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions on test data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores if available."""
        pass
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return current hyperparameters."""
        return self.hyperparameters.copy()
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'name': self.name,
                'hyperparameters': self.hyperparameters,
                'is_trained': self.is_trained
            }, f)
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.name = data['name']
            self.hyperparameters = data['hyperparameters']
            self.is_trained = data['is_trained']
            
            # For models with best_model attribute, restore it
            if hasattr(self, 'best_model') and hasattr(self.model, 'best_estimator_'):
                self.best_model = self.model.best_estimator_
                self.best_params = self.model.best_params_
                self.cv_results = self.model.cv_results_


class ModelFactory:
    """Factory class for creating different types of models."""
    
    _model_registry = {}
    
    @classmethod
    def register_model(cls, model_type: str, model_class):
        """Register a new model type."""
        cls._model_registry[model_type] = model_class
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """Create a model instance of the specified type."""
        if model_type not in cls._model_registry:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._model_registry[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model types."""
        return list(cls._model_registry.keys())