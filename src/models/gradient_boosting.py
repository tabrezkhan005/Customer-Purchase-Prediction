"""
Gradient Boosting model implementation using XGBoost with hyperparameter tuning.
"""
import numpy as np
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, Optional
from .base import BaseModel, ModelFactory

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    # Fallback to sklearn's GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingClassifier


class GradientBoostingModel(BaseModel):
    """Gradient Boosting model with hyperparameter optimization using XGBoost or sklearn fallback."""
    
    def __init__(self, name: str = "GradientBoosting", **kwargs):
        super().__init__(name, **kwargs)
        
        # Check if XGBoost is available
        self.use_xgboost = XGBOOST_AVAILABLE and kwargs.get('use_xgboost', True)
        
        # Default hyperparameters for grid search
        if self.use_xgboost:
            self.param_grid = kwargs.get('param_grid', {
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            })
        else:
            self.param_grid = kwargs.get('param_grid', {
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            })
        
        # Cross-validation settings
        self.cv_folds = kwargs.get('cv_folds', 5)
        self.scoring = kwargs.get('scoring', 'roc_auc')
        self.random_state = kwargs.get('random_state', 42)
        
        # Early stopping parameters
        self.early_stopping_rounds = kwargs.get('early_stopping_rounds', 10)
        self.validation_fraction = kwargs.get('validation_fraction', 0.1)
        
        # Best model after grid search
        self.best_model = None
        self.best_params = None
        self.cv_results = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the gradient boosting model with hyperparameter tuning."""
        if self.use_xgboost:
            self._train_xgboost(X_train, y_train)
        else:
            self._train_sklearn(X_train, y_train)
    
    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train using XGBoost."""
        # Create base XGBoost model
        base_model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1
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
        
        print(f"Using XGBoost")
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    def _train_sklearn(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train using sklearn's GradientBoostingClassifier."""
        # Create base sklearn gradient boosting model
        base_model = GradientBoostingClassifier(
            random_state=self.random_state,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.early_stopping_rounds
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
        
        print(f"Using sklearn GradientBoostingClassifier (XGBoost not available)")
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
        """Get feature importance from the gradient boosting model."""
        if not self.is_trained:
            return None
        
        return self.best_model.feature_importances_
    
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model implementation being used."""
        return {
            'framework': 'XGBoost' if self.use_xgboost else 'sklearn',
            'xgboost_available': XGBOOST_AVAILABLE,
            'model_type': type(self.best_model).__name__ if self.is_trained else None
        }


# Register the model with the factory
ModelFactory.register_model('gradient_boosting', GradientBoostingModel)