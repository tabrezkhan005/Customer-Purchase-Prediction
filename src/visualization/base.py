"""
Base classes and interfaces for visualization components.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class BaseVisualizer(ABC):
    """Abstract base class for visualization components."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), style: str = 'default'):
        self.figsize = figsize
        self.style = style
        plt.style.use(style)
    
    @abstractmethod
    def create_plot(self, data: Any, **kwargs) -> plt.Figure:
        """Create a plot from the given data."""
        pass
    
    def save_plot(self, fig: plt.Figure, filepath: str, dpi: int = 300) -> None:
        """Save plot to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


class BaseEDAVisualizer(BaseVisualizer):
    """Abstract base class for exploratory data analysis visualizations."""
    
    @abstractmethod
    def plot_distributions(self, data: pd.DataFrame, columns: List[str]) -> plt.Figure:
        """Plot feature distributions."""
        pass
    
    @abstractmethod
    def plot_correlation_matrix(self, data: pd.DataFrame) -> plt.Figure:
        """Plot correlation matrix heatmap."""
        pass
    
    @abstractmethod
    def plot_feature_relationships(self, data: pd.DataFrame, target_col: str) -> plt.Figure:
        """Plot relationships between features and target."""
        pass


class BaseModelVisualizer(BaseVisualizer):
    """Abstract base class for model evaluation visualizations."""
    
    @abstractmethod
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]) -> plt.Figure:
        """Plot confusion matrix."""
        pass
    
    @abstractmethod
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, model_name: str) -> plt.Figure:
        """Plot ROC curve."""
        pass
    
    @abstractmethod
    def plot_feature_importance(self, importance_scores: Dict[str, float], top_n: int = 10) -> plt.Figure:
        """Plot feature importance."""
        pass
    
    @abstractmethod
    def plot_model_comparison(self, results: List[Dict[str, Any]]) -> plt.Figure:
        """Plot model performance comparison."""
        pass


class VisualizationManager:
    """Manager class for coordinating different visualizers."""
    
    def __init__(self, output_dir: str = "outputs/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizers = {}
    
    def register_visualizer(self, name: str, visualizer: BaseVisualizer) -> None:
        """Register a visualizer."""
        self.visualizers[name] = visualizer
    
    def get_visualizer(self, name: str) -> BaseVisualizer:
        """Get a registered visualizer."""
        if name not in self.visualizers:
            raise ValueError(f"Visualizer {name} not found")
        return self.visualizers[name]
    
    def save_plot(self, fig: plt.Figure, filename: str, subfolder: str = "") -> str:
        """Save plot to the output directory."""
        if subfolder:
            save_path = self.output_dir / subfolder / filename
        else:
            save_path = self.output_dir / filename
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)