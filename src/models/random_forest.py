"""
Random Forest model implementation with hyperparameter tuning.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, Optional
from .base import BaseModel, ModelFactory


class RandomForestModel(BaseModel):
    """Random Forest model with hyperparameter optimization."""
    
    def __init__(self, name: str = "RandomForest", **kwargs):
        super().__init__(name, **kwargs)
        
        # Default hyperparameters for grid search
        self.param_grid = kwargs.get('param_grid', {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        })
        
        # Cross-validation settings
        self.cv_folds = kwargs.get('cv_folds', 5)
        self.scoring = kwargs.get('scoring', 'roc_auc')
        self.random_state = kwargs.get('random_state', 42)
        
        # Best model after grid search
        self.best_model = None
        self.best_params = None
        self.cv_results = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the random forest model with hyperparameter tuning."""
        # Create base random forest model
        base_model = RandomForestClassifier(
            random_state=self.random_state,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1  # Use all available cores
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        
        # Store results
        self.model = grid_search
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = grid_search.cv_results_
        self.is_trained = True
        
        # Update hyperparameters with best found parameters
        self.hyperparameters.update(self.best_params)
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make binary predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.best_model.predict(X_test)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.best_model.predict_proba(X_test)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from the random forest model."""
        if not self.is_trained:
            return None
        
        return self.best_model.feature_importances_
    
    def get_tree_count(self) -> Optional[int]:
        """Get the number of trees in the forest."""
        if not self.is_trained:
            return None
        
        return self.best_model.n_estimators
    
    def get_oob_score(self) -> Optional[float]:
        """Get out-of-bag score if available."""
        if not self.is_trained:
            return None
        
        # Check if OOB score was calculated (requires bootstrap=True and oob_score=True)
        if hasattr(self.best_model, 'oob_score_'):
            return self.best_model.oob_score_
        return None
    
    def get_cv_results(self) -> Optional[Dict[str, Any]]:
        """Get detailed cross-validation results."""
        return self.cv_results
    
    def get_feature_importance_ranking(self, feature_names: Optional[list] = None) -> list:
        """Get feature importance ranking with optional feature names."""
        if not self.is_trained:
            return []
        
        importance_scores = self.get_feature_importance()
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
        
        # Create list of (feature_name, importance_score) tuples
        feature_importance_pairs = list(zip(feature_names, importance_scores))
        
        # Sort by importance score in descending order
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance_pairs


# Register the model with the factory
ModelFactory.register_model('random_forest', RandomForestModel)