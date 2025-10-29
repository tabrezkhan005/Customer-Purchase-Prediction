"""
Feature importance visualization generator for creating charts and plots.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")


class FeatureVisualizationGenerator:
    """
    Generator for creating feature importance visualizations including bar charts,
    comparative plots, and correlation visualizations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'whitegrid'):
        """
        Initialize the visualization generator.
        
        Args:
            figsize: Default figure size for plots
            style: Seaborn style to use
        """
        self.figsize = figsize
        self.style = style
        sns.set_style(style)
        
    def plot_feature_importance_bar(self, 
                                  feature_rankings: List[Tuple[str, float]],
                                  title: str = "Feature Importance Rankings",
                                  top_k: int = 15,
                                  save_path: Optional[str] = None,
                                  show_values: bool = True) -> plt.Figure:
        """
        Create a horizontal bar chart of feature importance rankings.
        
        Args:
            feature_rankings: List of (feature_name, importance_score) tuples
            title: Title for the plot
            top_k: Number of top features to display
            save_path: Path to save the plot (optional)
            show_values: Whether to show values on bars
            
        Returns:
            matplotlib Figure object
        """
        logger.info(f"Creating feature importance bar chart for top {top_k} features")
        
        # Prepare data
        top_features = feature_rankings[:top_k]
        features, scores = zip(*top_features)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, scores, color=sns.color_palette("viridis", len(features)))
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Top feature at the top
        ax.set_xlabel('Importance Score')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels on bars if requested
        if show_values:
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax.text(bar.get_width() + max(scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        # Add grid for better readability
        ax.grid(axis='x', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            self._save_plot(fig, save_path)
        
        logger.info("Feature importance bar chart created successfully")
        return fig
    
    def plot_model_comparison(self, 
                            model_importance_dict: Dict[str, Dict[str, float]],
                            title: str = "Feature Importance Comparison Across Models",
                            top_k: int = 10,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comparative bar chart showing feature importance across multiple models.
        
        Args:
            model_importance_dict: Dict mapping model names to their feature importance dicts
            title: Title for the plot
            top_k: Number of top features to display
            save_path: Path to save the plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        logger.info(f"Creating model comparison chart for {len(model_importance_dict)} models")
        
        # Get all unique features and their average importance
        all_features = set()
        for importance_dict in model_importance_dict.values():
            all_features.update(importance_dict.keys())
        
        # Calculate average importance for each feature
        feature_avg_importance = {}
        for feature in all_features:
            scores = [model_dict.get(feature, 0.0) for model_dict in model_importance_dict.values()]
            feature_avg_importance[feature] = np.mean(scores)
        
        # Get top k features by average importance
        top_features = sorted(feature_avg_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_feature_names = [f[0] for f in top_features]
        
        # Prepare data for plotting
        comparison_data = {}
        for model_name, importance_dict in model_importance_dict.items():
            comparison_data[model_name] = [importance_dict.get(feature, 0.0) for feature in top_feature_names]
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame(comparison_data, index=top_feature_names)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] + 2))
        
        # Create grouped bar plot
        df.plot(kind='barh', ax=ax, width=0.8)
        
        # Customize plot
        ax.set_xlabel('Importance Score')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='x', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            self._save_plot(fig, save_path)
        
        logger.info("Model comparison chart created successfully")
        return fig
    
    def plot_feature_importance_heatmap(self, 
                                      model_importance_dict: Dict[str, Dict[str, float]],
                                      title: str = "Feature Importance Heatmap Across Models",
                                      top_k: int = 15,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a heatmap showing feature importance across models.
        
        Args:
            model_importance_dict: Dict mapping model names to their feature importance dicts
            title: Title for the plot
            top_k: Number of top features to display
            save_path: Path to save the plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        logger.info(f"Creating feature importance heatmap for {len(model_importance_dict)} models")
        
        # Get all unique features and their average importance
        all_features = set()
        for importance_dict in model_importance_dict.values():
            all_features.update(importance_dict.keys())
        
        # Calculate average importance for each feature
        feature_avg_importance = {}
        for feature in all_features:
            scores = [model_dict.get(feature, 0.0) for model_dict in model_importance_dict.values()]
            feature_avg_importance[feature] = np.mean(scores)
        
        # Get top k features by average importance
        top_features = sorted(feature_avg_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_feature_names = [f[0] for f in top_features]
        
        # Prepare data for heatmap
        heatmap_data = []
        model_names = list(model_importance_dict.keys())
        
        for feature in top_feature_names:
            row = [model_importance_dict[model].get(feature, 0.0) for model in model_names]
            heatmap_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(heatmap_data, index=top_feature_names, columns=model_names)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2), max(8, len(top_feature_names) * 0.5)))
        
        # Create heatmap
        sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, 
                   cbar_kws={'label': 'Importance Score'})
        
        # Customize plot
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Models')
        ax.set_ylabel('Features')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            self._save_plot(fig, save_path)
        
        logger.info("Feature importance heatmap created successfully")
        return fig
    
    def plot_feature_correlation_matrix(self, 
                                      data: pd.DataFrame,
                                      feature_rankings: List[Tuple[str, float]],
                                      title: str = "Feature Correlation Matrix (Top Important Features)",
                                      top_k: int = 15,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a correlation matrix for the most important features.
        
        Args:
            data: DataFrame containing the feature data
            feature_rankings: List of (feature_name, importance_score) tuples
            title: Title for the plot
            top_k: Number of top features to include
            save_path: Path to save the plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        logger.info(f"Creating correlation matrix for top {top_k} features")
        
        # Get top k features
        top_features = [feature for feature, _ in feature_rankings[:top_k]]
        
        # Filter data to include only top features that exist in the dataset
        available_features = [f for f in top_features if f in data.columns]
        
        if len(available_features) < 2:
            logger.warning("Not enough features available for correlation matrix")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'Not enough features available\nfor correlation analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return fig
        
        # Calculate correlation matrix
        correlation_matrix = data[available_features].corr()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(max(10, len(available_features) * 0.8), 
                                       max(8, len(available_features) * 0.8)))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, ax=ax,
                   square=True, cbar_kws={'label': 'Correlation Coefficient'})
        
        # Customize plot
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            self._save_plot(fig, save_path)
        
        logger.info("Feature correlation matrix created successfully")
        return fig
    
    def plot_feature_distribution_by_target(self, 
                                          data: pd.DataFrame,
                                          feature_name: str,
                                          target_column: str = 'Revenue',
                                          title: Optional[str] = None,
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature distribution split by target variable.
        
        Args:
            data: DataFrame containing the data
            feature_name: Name of the feature to plot
            target_column: Name of the target column
            title: Title for the plot (auto-generated if None)
            save_path: Path to save the plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        logger.info(f"Creating distribution plot for feature: {feature_name}")
        
        if feature_name not in data.columns:
            logger.error(f"Feature {feature_name} not found in data")
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, f'Feature "{feature_name}" not found in data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create distribution plot
        for target_value in data[target_column].unique():
            subset = data[data[target_column] == target_value][feature_name]
            ax.hist(subset, alpha=0.7, label=f'{target_column}={target_value}', bins=30)
        
        # Customize plot
        if title is None:
            title = f'Distribution of {feature_name} by {target_column}'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            self._save_plot(fig, save_path)
        
        logger.info(f"Distribution plot for {feature_name} created successfully")
        return fig
    
    def create_feature_importance_dashboard(self, 
                                          model_importance_dict: Dict[str, Dict[str, float]],
                                          data: Optional[pd.DataFrame] = None,
                                          save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create a comprehensive dashboard of feature importance visualizations.
        
        Args:
            model_importance_dict: Dict mapping model names to their feature importance dicts
            data: Optional DataFrame for correlation analysis
            save_dir: Directory to save all plots (optional)
            
        Returns:
            Dictionary mapping plot names to Figure objects
        """
        logger.info("Creating comprehensive feature importance dashboard")
        
        figures = {}
        
        # 1. Individual model importance plots
        for model_name, importance_dict in model_importance_dict.items():
            feature_rankings = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            save_path = f"{save_dir}/{model_name}_importance.png" if save_dir else None
            
            fig = self.plot_feature_importance_bar(
                feature_rankings, 
                title=f"Feature Importance - {model_name}",
                save_path=save_path
            )
            figures[f"{model_name}_importance"] = fig
        
        # 2. Model comparison plot
        save_path = f"{save_dir}/model_comparison.png" if save_dir else None
        fig = self.plot_model_comparison(model_importance_dict, save_path=save_path)
        figures["model_comparison"] = fig
        
        # 3. Feature importance heatmap
        save_path = f"{save_dir}/importance_heatmap.png" if save_dir else None
        fig = self.plot_feature_importance_heatmap(model_importance_dict, save_path=save_path)
        figures["importance_heatmap"] = fig
        
        # 4. Correlation matrix (if data provided)
        if data is not None:
            # Get overall top features across all models
            all_features = {}
            for importance_dict in model_importance_dict.values():
                for feature, score in importance_dict.items():
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(score)
            
            # Calculate average importance
            avg_importance = {feature: np.mean(scores) for feature, scores in all_features.items()}
            feature_rankings = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            save_path = f"{save_dir}/correlation_matrix.png" if save_dir else None
            fig = self.plot_feature_correlation_matrix(data, feature_rankings, save_path=save_path)
            figures["correlation_matrix"] = fig
        
        logger.info(f"Dashboard created with {len(figures)} visualizations")
        return figures
    
    def _save_plot(self, fig: plt.Figure, save_path: str) -> None:
        """
        Save a plot to the specified path.
        
        Args:
            fig: matplotlib Figure object
            save_path: Path to save the plot
        """
        try:
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save the plot
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Plot saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot to {save_path}: {str(e)}")
    
    def set_style(self, style: str) -> None:
        """
        Set the plotting style.
        
        Args:
            style: Seaborn style name
        """
        self.style = style
        sns.set_style(style)
        logger.info(f"Plot style set to: {style}")
    
    def set_figsize(self, figsize: Tuple[int, int]) -> None:
        """
        Set the default figure size.
        
        Args:
            figsize: Tuple of (width, height) in inches
        """
        self.figsize = figsize
        logger.info(f"Default figure size set to: {figsize}")
    
    def close_all_figures(self) -> None:
        """Close all matplotlib figures to free memory."""
        plt.close('all')
        logger.info("All matplotlib figures closed")