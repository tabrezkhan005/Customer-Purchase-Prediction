"""
Model training orchestrator for coordinating the ML training process.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Any, List, Tuple, Optional, Union
import time
import warnings
from .base import BaseModel, ModelFactory


class ModelTrainer:
    """Orchestrates the model training process with train-test split and cross-validation."""
    
    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        
        # Train-test split configuration
        self.test_size = kwargs.get('test_size', 0.2)
        self.stratify = kwargs.get('stratify', True)
        
        # Cross-validation configuration
        self.cv_folds = kwargs.get('cv_folds', 5)
        self.cv_scoring = kwargs.get('cv_scoring', 'roc_auc')
        
        # Training results storage
        self.trained_models = {}
        self.training_results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Feature names for interpretability
        self.feature_names = None
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data by splitting into train and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional list of feature names
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.feature_names = feature_names
        
        # Perform stratified train-test split
        stratify_param = y if self.stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        print(f"Data split completed:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Test set: {self.X_test.shape[0]} samples")
        print(f"  Features: {self.X_train.shape[1]}")
        
        # Print class distribution
        train_pos = np.sum(self.y_train)
        test_pos = np.sum(self.y_test)
        print(f"  Training set positive class: {train_pos}/{len(self.y_train)} ({train_pos/len(self.y_train)*100:.1f}%)")
        print(f"  Test set positive class: {test_pos}/{len(self.y_test)} ({test_pos/len(self.y_test)*100:.1f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, model_type: str, model_name: Optional[str] = None, **model_kwargs) -> BaseModel:
        """
        Train a single model.
        
        Args:
            model_type: Type of model to create (registered in ModelFactory)
            model_name: Optional custom name for the model
            **model_kwargs: Additional arguments for model creation
            
        Returns:
            Trained model instance
        """
        if self.X_train is None:
            raise ValueError("Data must be prepared first using prepare_data()")
        
        # Create model instance
        if model_name is None:
            model_name = f"{model_type}_{int(time.time())}"
        
        model = ModelFactory.create_model(
            model_type, 
            name=model_name,
            random_state=self.random_state,
            **model_kwargs
        )
        
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        # Train the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.train(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        
        # Store trained model
        self.trained_models[model_name] = model
        
        # Store training results
        self.training_results[model_name] = {
            'model_type': model_type,
            'training_time': training_time,
            'hyperparameters': model.get_hyperparameters()
        }
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        return model
    
    def train_multiple_models(self, model_configs: List[Dict[str, Any]]) -> Dict[str, BaseModel]:
        """
        Train multiple models with different configurations.
        
        Args:
            model_configs: List of dictionaries with model configuration
                          Each dict should have 'model_type' and optionally 'model_name' and other kwargs
                          
        Returns:
            Dictionary of trained models
        """
        trained_models = {}
        
        for config in model_configs:
            model_type = config.pop('model_type')
            model_name = config.pop('model_name', None)
            
            try:
                model = self.train_model(model_type, model_name, **config)
                trained_models[model.name] = model
            except Exception as e:
                print(f"Failed to train {model_type}: {str(e)}")
                continue
        
        return trained_models
    
    def cross_validate_model(self, model: BaseModel, scoring: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform cross-validation on a trained model.
        
        Args:
            model: Trained model to cross-validate
            scoring: Scoring metric (defaults to self.cv_scoring)
            
        Returns:
            Dictionary with cross-validation results
        """
        if self.X_train is None:
            raise ValueError("Data must be prepared first using prepare_data()")
        
        scoring = scoring or self.cv_scoring
        
        # Create stratified k-fold cross-validator
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        print(f"\nPerforming {self.cv_folds}-fold cross-validation for {model.name}...")
        
        # Get the underlying sklearn model for cross-validation
        sklearn_model = model.best_model if hasattr(model, 'best_model') else model.model
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            sklearn_model, self.X_train, self.y_train,
            cv=cv, scoring=scoring, n_jobs=-1
        )
        
        cv_results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'scoring_metric': scoring
        }
        
        # Update training results
        if model.name in self.training_results:
            self.training_results[model.name].update(cv_results)
        
        print(f"Cross-validation {scoring}: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']*2:.4f})")
        
        return cv_results
    
    def cross_validate_all_models(self, scoring: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Perform cross-validation on all trained models.
        
        Args:
            scoring: Scoring metric (defaults to self.cv_scoring)
            
        Returns:
            Dictionary with cross-validation results for all models
        """
        all_cv_results = {}
        
        for model_name, model in self.trained_models.items():
            try:
                cv_results = self.cross_validate_model(model, scoring)
                all_cv_results[model_name] = cv_results
            except Exception as e:
                print(f"Cross-validation failed for {model_name}: {str(e)}")
                continue
        
        return all_cv_results
    
    def evaluate_on_test_set(self, model: BaseModel) -> Dict[str, Any]:
        """
        Evaluate a model on the test set.
        
        Args:
            model: Trained model to evaluate
            
        Returns:
            Dictionary with test set evaluation results
        """
        if self.X_test is None:
            raise ValueError("Data must be prepared first using prepare_data()")
        
        print(f"\nEvaluating {model.name} on test set...")
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # Generate classification report
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        test_results = {
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'classification_report': report,
            'confusion_matrix': cm,
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score']
        }
        
        # Update training results
        if model.name in self.training_results:
            self.training_results[model.name]['test_results'] = test_results
        
        print(f"Test set accuracy: {test_results['accuracy']:.4f}")
        print(f"Test set precision: {test_results['precision']:.4f}")
        print(f"Test set recall: {test_results['recall']:.4f}")
        print(f"Test set F1-score: {test_results['f1_score']:.4f}")
        
        return test_results
    
    def get_training_summary(self) -> pd.DataFrame:
        """
        Get a summary of all training results.
        
        Returns:
            DataFrame with training summary
        """
        if not self.training_results:
            return pd.DataFrame()
        
        summary_data = []
        
        for model_name, results in self.training_results.items():
            row = {
                'model_name': model_name,
                'model_type': results.get('model_type', 'Unknown'),
                'training_time': results.get('training_time', 0)
            }
            
            # Add cross-validation results if available
            if 'cv_mean' in results:
                row['cv_score_mean'] = results['cv_mean']
                row['cv_score_std'] = results['cv_std']
            
            # Add test results if available
            if 'test_results' in results:
                test_results = results['test_results']
                row['test_accuracy'] = test_results['accuracy']
                row['test_precision'] = test_results['precision']
                row['test_recall'] = test_results['recall']
                row['test_f1_score'] = test_results['f1_score']
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def get_best_model(self, metric: str = 'cv_score_mean') -> Optional[BaseModel]:
        """
        Get the best performing model based on a specified metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Best performing model or None if no models trained
        """
        if not self.training_results:
            return None
        
        best_score = -float('inf')
        best_model_name = None
        
        for model_name, results in self.training_results.items():
            if metric in results:
                score = results[metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        return self.trained_models.get(best_model_name)
    
    def save_all_models(self, directory: str) -> None:
        """
        Save all trained models to a directory.
        
        Args:
            directory: Directory to save models
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            filepath = os.path.join(directory, f"{model_name}.pkl")
            try:
                model.save_model(filepath)
                print(f"Saved {model_name} to {filepath}")
            except Exception as e:
                print(f"Failed to save {model_name}: {str(e)}")