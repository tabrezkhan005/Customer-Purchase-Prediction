"""
Configuration management for the ML pipeline.
"""
import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class DataConfig:
    """Configuration for data processing."""
    dataset_path: str = "online_shoppers_intention.csv"
    test_size: float = 0.2
    random_state: int = 42
    validation_split: float = 0.2
    
    # Data preprocessing
    missing_value_strategy: str = "median"  # median, mean, mode, drop
    categorical_encoding: str = "label"  # label, onehot
    numeric_scaling: str = "standard"  # standard, minmax, robust
    
    # Feature selection
    feature_selection: bool = False
    max_features: Optional[int] = None


@dataclass
class ModelConfig:
    """Configuration for model training."""
    # Cross-validation
    cv_folds: int = 5
    cv_scoring: str = "roc_auc"
    
    # Hyperparameter tuning
    hyperparameter_tuning: bool = True
    tuning_method: str = "grid_search"  # grid_search, random_search
    n_iter_random: int = 100
    
    # Model-specific hyperparameters
    logistic_regression: Dict[str, Any] = None
    random_forest: Dict[str, Any] = None
    gradient_boosting: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.logistic_regression is None:
            self.logistic_regression = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000]
            }
        
        if self.random_forest is None:
            self.random_forest = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        if self.gradient_boosting is None:
            self.gradient_boosting = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0]
            }


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    metrics: list = None
    plot_roc_curves: bool = True
    plot_confusion_matrices: bool = True
    plot_feature_importance: bool = True
    save_predictions: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    figure_size: tuple = (10, 6)
    dpi: int = 300
    style: str = 'seaborn-v0_8'
    color_palette: str = 'viridis'
    save_format: str = 'png'
    
    # EDA plots
    plot_distributions: bool = True
    plot_correlations: bool = True
    plot_feature_relationships: bool = True
    
    # Model evaluation plots
    plot_model_comparison: bool = True
    top_features_to_plot: int = 10


@dataclass
class OutputConfig:
    """Configuration for outputs."""
    base_output_dir: str = "outputs"
    models_dir: str = "models"
    plots_dir: str = "plots"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    
    save_models: bool = True
    save_plots: bool = True
    save_reports: bool = True


@dataclass
class MLPipelineConfig:
    """Main configuration class combining all configs."""
    data: DataConfig = None
    models: ModelConfig = None
    evaluation: EvaluationConfig = None
    visualization: VisualizationConfig = None
    output: OutputConfig = None
    
    # Global settings
    random_seed: int = 42
    verbose: bool = True
    n_jobs: int = -1
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.models is None:
            self.models = ModelConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.output is None:
            self.output = OutputConfig()


class ConfigManager:
    """Manager for loading and saving configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config = MLPipelineConfig()
    
    def load_config(self, config_path: Optional[str] = None) -> MLPipelineConfig:
        """Load configuration from file."""
        path = config_path or self.config_path
        
        if not os.path.exists(path):
            print(f"Config file {path} not found. Using default configuration.")
            return self.config
        
        try:
            with open(path, 'r') as f:
                if path.endswith('.json'):
                    config_dict = json.load(f)
                elif path.endswith(('.yaml', '.yml')):
                    config_dict = yaml.safe_load(f)
                else:
                    raise ValueError("Config file must be JSON or YAML format")
            
            # Update config with loaded values
            self._update_config_from_dict(config_dict)
            print(f"Configuration loaded from {path}")
            
        except Exception as e:
            print(f"Error loading config from {path}: {e}")
            print("Using default configuration.")
        
        return self.config
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        path = config_path or self.config_path
        
        config_dict = asdict(self.config)
        
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                if path.endswith('.json'):
                    json.dump(config_dict, f, indent=2)
                elif path.endswith(('.yaml', '.yml')):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    raise ValueError("Config file must be JSON or YAML format")
            
            print(f"Configuration saved to {path}")
            
        except Exception as e:
            print(f"Error saving config to {path}: {e}")
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(self.config.data, key):
                    setattr(self.config.data, key, value)
        
        if 'models' in config_dict:
            for key, value in config_dict['models'].items():
                if hasattr(self.config.models, key):
                    setattr(self.config.models, key, value)
        
        if 'evaluation' in config_dict:
            for key, value in config_dict['evaluation'].items():
                if hasattr(self.config.evaluation, key):
                    setattr(self.config.evaluation, key, value)
        
        if 'visualization' in config_dict:
            for key, value in config_dict['visualization'].items():
                if hasattr(self.config.visualization, key):
                    setattr(self.config.visualization, key, value)
        
        if 'output' in config_dict:
            for key, value in config_dict['output'].items():
                if hasattr(self.config.output, key):
                    setattr(self.config.output, key, value)
        
        # Update global settings
        for key in ['random_seed', 'verbose', 'n_jobs']:
            if key in config_dict:
                setattr(self.config, key, config_dict[key])
    
    def get_config(self) -> MLPipelineConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with keyword arguments."""
        config_dict = kwargs
        self._update_config_from_dict(config_dict)