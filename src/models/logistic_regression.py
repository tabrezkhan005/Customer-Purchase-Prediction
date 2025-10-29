"""
Logistic Regression model implementation with hyperparameter tuning.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, Optional
from .base import BaseModel, ModelFactory


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model with hyperparameter optimization."""
    
    def __init__(self, name: str = "LogisticRegression", **kwargs):
        super().__init__(name, **kwargs)
        
        # Default hyperparameters for grid search
        self.param_grid = kwargs.get('param_grid', {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga', 'lbfgs'],
            'max_iter': [1000, 2000, 3000]
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
        """Train the logistic regression model with hyperparameter tuning."""
        # Create base logistic regression model
        base_model = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Adjust parameter grid based on solver constraints
        adjusted_param_grid = self._adjust_param_grid()
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=adjusted_param_grid,
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
        """Get feature importance from logistic regression coefficients."""
        if not self.is_trained:
            return None
        
        # Return absolute values of coefficients as feature importance
        coefficients = self.best_model.coef_[0]
        return np.abs(coefficients)
    
    def get_coefficients(self) -> Optional[np.ndarray]:
        """Get raw coefficients from the logistic regression model."""
        if not self.is_trained:
            return None
        
        return self.best_model.coef_[0]
    
    def get_intercept(self) -> Optional[float]:
        """Get the intercept from the logistic regression model."""
        if not self.is_trained:
            return None
        
        return self.best_model.intercept_[0]
    
    def get_cv_results(self) -> Optional[Dict[str, Any]]:
        """Get detailed cross-validation results."""
        return self.cv_results
    
    def _adjust_param_grid(self) -> list:
        """Adjust parameter grid to handle solver-penalty constraints."""
        adjusted_grids = []
        
        for solver in self.param_grid.get('solver', ['liblinear']):
            grid = {
                'C': self.param_grid.get('C', [1.0]),
                'solver': [solver],
                'max_iter': self.param_grid.get('max_iter', [1000])
            }
            
            # Adjust penalties based on solver constraints
            if solver == 'liblinear':
                grid['penalty'] = ['l1', 'l2']
            elif solver == 'saga':
                grid['penalty'] = ['l1', 'l2', 'elasticnet']
                if 'elasticnet' in grid['penalty']:
                    grid['l1_ratio'] = [0.1, 0.5, 0.7, 0.9]
            elif solver == 'lbfgs':
                grid['penalty'] = ['l2']
            else:
                grid['penalty'] = ['l2']
            
            adjusted_grids.append(grid)
        
        return adjusted_grids


# Register the model with the factory
ModelFactory.register_model('logistic_regression', LogisticRegressionModel)