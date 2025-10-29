"""
Main ML pipeline orchestrator for customer purchase prediction.
"""
import os
import logging
import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config.settings import ConfigManager, MLPipelineConfig
from src.models.base import BaseModel, ModelFactory
from src.data.base import DataPipeline
from src.data.loader import DataLoader
from src.data.preprocessing import PreprocessingPipeline
from src.evaluation.base import ModelResults
from src.evaluation.evaluator import ModelEvaluator
from src.visualization.base import VisualizationManager
from src.analysis.feature_importance import FeatureImportanceAnalyzer
from src.analysis.business_insights import BusinessInsightGenerator

# Import model implementations to register them
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.gradient_boosting import GradientBoostingModel


class MLPipeline:
    """Main orchestrator for the ML pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the ML pipeline with configuration."""
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        # Initialize components
        self.data_pipeline = None
        self.models = {}
        self.results = {}
        self.visualization_manager = VisualizationManager(
            output_dir=os.path.join(self.config.output.base_output_dir, self.config.output.plots_dir)
        )
        
        # Set up logging
        self._setup_logging()
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        # Create output directories
        self._create_output_directories()
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_dir = Path(self.config.output.base_output_dir) / self.config.output.logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO if self.config.verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("ML Pipeline initialized")
    
    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)
        
        # Set seeds for ML libraries
        try:
            import sklearn
            sklearn.utils.check_random_state(self.config.random_seed)
        except ImportError:
            pass
            
        try:
            import xgboost as xgb
            # XGBoost uses numpy random state
        except ImportError:
            pass
        
        self.logger.info(f"Random seed set to {self.config.random_seed}")
    
    def _create_output_directories(self) -> None:
        """Create necessary output directories."""
        base_dir = Path(self.config.output.base_output_dir)
        
        directories = [
            base_dir / self.config.output.models_dir,
            base_dir / self.config.output.plots_dir,
            base_dir / self.config.output.reports_dir,
            base_dir / self.config.output.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Output directories created")
    
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load the dataset."""
        data_path = filepath or self.config.data.dataset_path
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        
        self.logger.info(f"Loading data from {data_path}")
        
        # Use the DataLoader component
        data_loader = DataLoader()
        data = data_loader.load_data(data_path)
        
        # Validate the loaded data
        if not data_loader.validate_data(data):
            raise ValueError("Data validation failed")
        
        self.logger.info(f"Successfully loaded {len(data)} records with {len(data.columns)} features")
        return data
    
    def setup_data_pipeline(self, processors: List[Tuple[str, Any]]) -> None:
        """Set up the data preprocessing pipeline."""
        self.data_pipeline = DataPipeline(processors)
        self.logger.info("Data pipeline configured")
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess the data using the configured pipeline."""
        self.logger.info("Preprocessing data")
        
        # Initialize data preprocessor with configuration
        preprocessor = PreprocessingPipeline(
            numeric_imputation=self.config.data.missing_value_strategy,
            categorical_imputation='most_frequent',
            encoding_strategy=self.config.data.categorical_encoding,
            scaling_method=self.config.data.numeric_scaling
        )
        
        # Separate features and target
        if 'Revenue' not in data.columns:
            raise ValueError("Target column 'Revenue' not found in data")
        
        # Fit and transform the data
        processed_data = preprocessor.fit_transform(data)
        
        # Extract features and target
        feature_columns = [col for col in processed_data.columns if col != 'Revenue']
        
        # Ensure all feature columns are numeric
        feature_data = processed_data[feature_columns]
        
        # Convert any remaining non-numeric columns to numeric
        for col in feature_data.columns:
            if not pd.api.types.is_numeric_dtype(feature_data[col]):
                # Try to convert to numeric, if fails use label encoding
                try:
                    feature_data[col] = pd.to_numeric(feature_data[col])
                except (ValueError, TypeError):
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    feature_data[col] = le.fit_transform(feature_data[col].astype(str))
        
        X = feature_data.values.astype(np.float64)
        y = processed_data['Revenue'].astype(int).values
        
        # Get feature names
        feature_names = feature_columns
        
        self.logger.info(f"Data preprocessed: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names
    
    def register_model(self, model_type: str, model_class) -> None:
        """Register a new model type."""
        ModelFactory.register_model(model_type, model_class)
        self.logger.info(f"Model type '{model_type}' registered")
    
    def create_model(self, model_type: str, **kwargs) -> BaseModel:
        """Create a model instance."""
        model = ModelFactory.create_model(model_type, **kwargs)
        self.models[model.name] = model
        self.logger.info(f"Model '{model.name}' created")
        return model
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        self.logger.info(f"Training model '{model_name}'")
        model = self.models[model_name]
        model.train(X_train, y_train)
        
        # Save model if configured
        if self.config.output.save_models:
            model_path = Path(self.config.output.base_output_dir) / self.config.output.models_dir / f"{model_name}.pkl"
            model.save_model(str(model_path))
            self.logger.info(f"Model '{model_name}' saved to {model_path}")
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train all registered models."""
        self.logger.info("Training all models")
        
        for model_name in self.models:
            self.train_model(model_name, X_train, y_train)
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray, 
                      X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None) -> ModelResults:
        """Evaluate a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        self.logger.info(f"Evaluating model '{model_name}'")
        
        # Use the ModelEvaluator component
        evaluator = ModelEvaluator()
        
        model = self.models[model_name]
        results = evaluator.evaluate_model(model, X_test, y_test)
        
        # Perform cross-validation if training data is provided
        if X_train is not None and y_train is not None:
            cv_results = evaluator.cross_validate(
                model, X_train, y_train, cv=self.config.models.cv_folds
            )
            results.cross_val_scores = cv_results.get('test_score', [])
        
        return results
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray,
                           X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None) -> Dict[str, ModelResults]:
        """Evaluate all trained models."""
        self.logger.info("Evaluating all models")
        
        results = {}
        for model_name in self.models:
            results[model_name] = self.evaluate_model(model_name, X_test, y_test, X_train, y_train)
        
        self.results = results
        return results
    
    def generate_report(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a comprehensive report of results."""
        self.logger.info("Generating comprehensive report")
        
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_all_models first.")
        
        # Create comprehensive report
        report = {
            'pipeline_config': {
                'random_seed': self.config.random_seed,
                'test_size': self.config.data.test_size,
                'cv_folds': self.config.models.cv_folds,
                'preprocessing': {
                    'missing_value_strategy': self.config.data.missing_value_strategy,
                    'categorical_encoding': self.config.data.categorical_encoding,
                    'numeric_scaling': self.config.data.numeric_scaling
                }
            },
            'model_results': {},
            'model_comparison': {},
            'feature_analysis': {}
        }
        
        # Add model results
        for model_name, results in self.results.items():
            report['model_results'][model_name] = results.to_dict()
        
        # Generate model comparison
        evaluator = ModelEvaluator()
        comparison = evaluator.compare_models(list(self.results.values()))
        report['model_comparison'] = comparison
        
        # Generate feature importance analysis if feature names are provided
        if feature_names:
            feature_analyzer = FeatureImportanceAnalyzer()
            business_insights = BusinessInsightGenerator()
            
            for model_name, model in self.models.items():
                if model.is_trained:
                    importance_scores = feature_analyzer.analyze_feature_importance(model, feature_names)
                    if importance_scores:
                        report['feature_analysis'][model_name] = {
                            'importance_scores': importance_scores,
                            'top_features': feature_analyzer.get_top_features(importance_scores, n=10),
                            'business_insights': business_insights.generate_insights(importance_scores)
                        }
        
        # Save report to file
        if self.config.output.save_reports:
            report_path = Path(self.config.output.base_output_dir) / self.config.output.reports_dir / "pipeline_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Report saved to {report_path}")
        
        return report
    
    def run_complete_pipeline(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete ML pipeline from start to finish."""
        self.logger.info("Starting complete ML pipeline execution")
        
        try:
            # Step 1: Load data
            self.logger.info("Step 1: Loading data")
            data = self.load_data(data_path)
            
            # Step 2: Preprocess data
            self.logger.info("Step 2: Preprocessing data")
            X, y, feature_names = self.preprocess_data(data)
            
            # Step 3: Split data
            self.logger.info("Step 3: Splitting data")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.data.test_size,
                random_state=self.config.random_seed,
                stratify=y
            )
            
            self.logger.info(f"Training set: {X_train.shape[0]} samples")
            self.logger.info(f"Test set: {X_test.shape[0]} samples")
            
            # Step 4: Create and register models
            self.logger.info("Step 4: Creating models")
            self._create_default_models()
            
            # Step 5: Train models
            self.logger.info("Step 5: Training models")
            self.train_all_models(X_train, y_train)
            
            # Step 6: Evaluate models
            self.logger.info("Step 6: Evaluating models")
            results = self.evaluate_all_models(X_test, y_test, X_train, y_train)
            
            # Step 7: Generate visualizations
            self.logger.info("Step 7: Generating visualizations")
            self._generate_visualizations(X_test, y_test, feature_names)
            
            # Step 8: Generate comprehensive report
            self.logger.info("Step 8: Generating report")
            report = self.generate_report(feature_names)
            
            self.logger.info("Complete ML pipeline execution finished successfully!")
            
            return {
                "status": "success",
                "data_shape": X.shape,
                "train_size": X_train.shape[0],
                "test_size": X_test.shape[0],
                "models_trained": list(self.models.keys()),
                "best_model": self._get_best_model_name(),
                "report": report
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise
    
    def _create_default_models(self) -> None:
        """Create default models with configuration parameters."""
        # Logistic Regression
        lr_params = self.config.models.logistic_regression.copy()
        self.create_model('logistic_regression', 
                         name='LogisticRegression', 
                         param_grid=lr_params,
                         cv_folds=self.config.models.cv_folds,
                         scoring=self.config.models.cv_scoring,
                         random_state=self.config.random_seed)
        
        # Random Forest
        rf_params = self.config.models.random_forest.copy()
        self.create_model('random_forest', 
                         name='RandomForest', 
                         param_grid=rf_params,
                         cv_folds=self.config.models.cv_folds,
                         scoring=self.config.models.cv_scoring,
                         random_state=self.config.random_seed)
        
        # Gradient Boosting
        gb_params = self.config.models.gradient_boosting.copy()
        self.create_model('gradient_boosting', 
                         name='GradientBoosting', 
                         param_grid=gb_params,
                         cv_folds=self.config.models.cv_folds,
                         scoring=self.config.models.cv_scoring,
                         random_state=self.config.random_seed)
    
    def _generate_visualizations(self, X_test: np.ndarray, y_test: np.ndarray, 
                               feature_names: Optional[List[str]] = None) -> None:
        """Generate visualizations for model results."""
        if not self.config.visualization.plot_model_comparison:
            return
        
        try:
            from src.evaluation.visualizer import EvaluationVisualizer
            from src.analysis.visualization import FeatureVisualizationGenerator
            
            # Create evaluation visualizer
            eval_visualizer = EvaluationVisualizer(
                output_dir=Path(self.config.output.base_output_dir) / self.config.output.plots_dir
            )
            
            # Generate model comparison plots
            eval_visualizer.plot_model_comparison(self.results)
            
            # Generate ROC curves and confusion matrices for each model
            for model_name, model in self.models.items():
                if model.is_trained:
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    eval_visualizer.plot_roc_curve(y_test, y_proba, model_name)
                    eval_visualizer.plot_confusion_matrix(y_test, y_pred, model_name)
            
            # Generate feature importance plots if feature names are available
            if feature_names:
                feature_viz = FeatureVisualizationGenerator(
                    output_dir=Path(self.config.output.base_output_dir) / self.config.output.plots_dir
                )
                
                for model_name, model in self.models.items():
                    if model.is_trained:
                        importance_scores = model.get_feature_importance()
                        if importance_scores is not None:
                            feature_importance_dict = dict(zip(feature_names, importance_scores))
                            feature_viz.plot_feature_importance(feature_importance_dict, model_name)
            
            self.logger.info("Visualizations generated successfully")
            
        except ImportError as e:
            self.logger.warning(f"Could not generate visualizations: {e}")
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
    
    def _get_best_model_name(self) -> Optional[str]:
        """Get the name of the best performing model based on ROC AUC."""
        if not self.results:
            return None
        
        best_model = None
        best_score = 0.0
        
        for model_name, results in self.results.items():
            if results.roc_auc > best_score:
                best_score = results.roc_auc
                best_model = model_name
        
        return best_model
    
    def get_model(self, model_name: str) -> BaseModel:
        """Get a trained model by name."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        return self.models[model_name]
    
    def get_results(self) -> Dict[str, ModelResults]:
        """Get evaluation results."""
        return self.results.copy()
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save current configuration."""
        self.config_manager.save_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        self.config = self.config_manager.load_config(config_path)