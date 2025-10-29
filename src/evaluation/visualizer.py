"""
Visualization generator for model evaluation results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

from ..visualization.base import BaseModelVisualizer
from .base import ModelResults


class EvaluationVisualizer(BaseModelVisualizer):
    """
    Comprehensive visualization generator for model evaluation results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), style: str = 'seaborn-v0_8'):
        """
        Initialize the evaluation visualizer.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style to use
        """
        super().__init__(figsize, style)
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str] = None) -> plt.Figure:
        """
        Plot confusion matrix with annotations.
        
        Args:
            cm: Confusion matrix
            class_names: Names of classes
            
        Returns:
            matplotlib Figure object
        """
        if class_names is None:
            class_names = ['No Purchase', 'Purchase']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm[i, j] / total * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                       ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      model_name: str = "Model") -> plt.Figure:
        """
        Plot ROC curve for a single model.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_multiple_roc_curves(self, roc_data: Dict[str, Dict[str, np.ndarray]]) -> plt.Figure:
        """
        Plot ROC curves for multiple models on the same plot.
        
        Args:
            roc_data: Dict with model names as keys and ROC data as values
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot each model's ROC curve
        for i, (model_name, data) in enumerate(roc_data.items()):
            color = self.colors[i % len(self.colors)]
            ax.plot(data['fpr'], data['tpr'], color=color, lw=2,
                   label=f'{model_name} (AUC = {data["auc"]:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', 
               label='Random Classifier (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance_scores: Dict[str, float], 
                              top_n: int = 10) -> plt.Figure:
        """
        Plot feature importance scores.
        
        Args:
            importance_scores: Dict mapping feature names to importance scores
            top_n: Number of top features to display
            
        Returns:
            matplotlib Figure object
        """
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        features, scores = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], max(6, len(features) * 0.4)))
        
        # Create horizontal bar plot
        colors = ['green' if score > 0 else 'red' for score in scores]
        bars = ax.barh(range(len(features)), scores, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + (0.01 if score > 0 else -0.01), i, f'{score:.3f}', 
                   va='center', ha='left' if score > 0 else 'right', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, results: List[ModelResults]) -> plt.Figure:
        """
        Plot comprehensive model performance comparison.
        
        Args:
            results: List of ModelResults objects
            
        Returns:
            matplotlib Figure object
        """
        if not results:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No results to display', ha='center', va='center')
            return fig
        
        # Extract data for plotting
        model_names = [result.model_name for result in results]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        data = []
        for result in results:
            data.append([
                result.accuracy, result.precision, result.recall, 
                result.f1_score, result.roc_auc
            ])
        
        df = pd.DataFrame(data, columns=metrics, index=model_names)
        
        # Create subplot for bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart comparison
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            offset = (i - 2) * width
            bars = ax1.bar(x + offset, df[metric], width, 
                          label=metric.replace('_', ' ').title(),
                          color=self.colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.1)
        
        # Heatmap
        sns.heatmap(df.T, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=ax2, cbar_kws={'label': 'Score'})
        ax2.set_title('Performance Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Models', fontsize=12)
        ax2.set_ylabel('Metrics', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_cross_validation_results(self, cv_results: Dict[str, Any]) -> plt.Figure:
        """
        Plot cross-validation results with error bars.
        
        Args:
            cv_results: Cross-validation results from CrossValidator
            
        Returns:
            matplotlib Figure object
        """
        if 'individual_results' not in cv_results:
            # Single model results
            return self._plot_single_model_cv(cv_results)
        else:
            # Multiple model results
            return self._plot_multiple_models_cv(cv_results['individual_results'])
    
    def _plot_single_model_cv(self, cv_results: Dict[str, Any]) -> plt.Figure:
        """Plot cross-validation results for a single model."""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot of CV scores
        cv_data = []
        metric_names = []
        for metric in metrics:
            if metric in cv_results:
                cv_data.append(cv_results[metric]['test_scores'])
                metric_names.append(metric.replace('_', ' ').title())
        
        ax1.boxplot(cv_data, tick_labels=metric_names)
        ax1.set_title('Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Mean scores with error bars
        means = [np.mean(scores) for scores in cv_data]
        stds = [np.std(scores) for scores in cv_data]
        
        bars = ax2.bar(metric_names, means, yerr=stds, capsize=5, 
                      color=self.colors[:len(means)], alpha=0.7)
        ax2.set_title('Mean CV Scores with Standard Deviation', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Score', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax2.text(bar.get_x() + bar.get_width()/2., mean + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def _plot_multiple_models_cv(self, individual_results: Dict[str, Any]) -> plt.Figure:
        """Plot cross-validation results for multiple models."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        model_names = list(individual_results.keys())
        
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            means = []
            stds = []
            
            for metric in metrics:
                if metric in individual_results[model_name]:
                    means.append(individual_results[model_name][metric]['test_mean'])
                    stds.append(individual_results[model_name][metric]['test_std'])
                else:
                    means.append(0)
                    stds.append(0)
            
            offset = (i - len(model_names)/2 + 0.5) * width
            bars = ax.bar(x + offset, means, width, yerr=stds, 
                         label=model_name, color=self.colors[i], 
                         alpha=0.8, capsize=3)
            
            # Add value labels
            for bar, mean in zip(bars, means):
                if mean > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Cross-Validation Results Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        return fig
    
    def plot_performance_summary(self, comparison_results: Dict[str, Any]) -> plt.Figure:
        """
        Create a comprehensive performance summary visualization.
        
        Args:
            comparison_results: Results from ModelEvaluator.compare_models()
            
        Returns:
            matplotlib Figure object
        """
        if not comparison_results or 'comparison_table' not in comparison_results:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No comparison data available', ha='center', va='center')
            return fig
        
        df = pd.DataFrame(comparison_results['comparison_table'])
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main performance comparison
        ax1 = fig.add_subplot(gs[0, :])
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        x = np.arange(len(df))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            offset = (i - 2) * width
            bars = ax1.bar(x + offset, df[metric], width, 
                          label=metric.replace('_', ' ').title(),
                          color=self.colors[i], alpha=0.8)
        
        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Comprehensive Model Performance Comparison', 
                     fontsize=16, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['model_name'], rotation=45, ha='right')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.1)
        
        # Cross-validation stability
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.bar(df['model_name'], df['cv_std_accuracy'], 
               color='orange', alpha=0.7)
        ax2.set_title('Model Stability (CV Std)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Standard Deviation', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Best model summary
        ax3 = fig.add_subplot(gs[1, 1])
        if 'best_model' in comparison_results:
            best_model = comparison_results['best_model']
            metrics_values = [best_model['accuracy'], best_model['f1_score'], best_model['roc_auc']]
            metric_names = ['Accuracy', 'F1-Score', 'ROC-AUC']
            
            bars = ax3.bar(metric_names, metrics_values, color='green', alpha=0.7)
            ax3.set_title(f'Best Model: {best_model["name"]}', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Score', fontsize=10)
            ax3.set_ylim(0, 1.1)
            
            for bar, value in zip(bars, metrics_values):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Summary statistics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('tight')
        ax4.axis('off')
        
        if 'summary_stats' in comparison_results:
            stats = comparison_results['summary_stats']
            table_data = [
                ['Number of Models', stats['num_models']],
                ['Best Accuracy', f"{stats['best_accuracy']:.3f}"],
                ['Best F1-Score', f"{stats['best_f1']:.3f}"],
                ['Best ROC-AUC', f"{stats['best_roc_auc']:.3f}"],
                ['Accuracy Range', f"{stats['accuracy_range']:.3f}"],
                ['F1 Range', f"{stats['f1_range']:.3f}"]
            ]
            
            table = ax4.table(cellText=table_data, 
                            colLabels=['Metric', 'Value'],
                            cellLoc='center', loc='center',
                            colWidths=[0.3, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
        
        return fig
    
    def create_plot(self, data: Any, plot_type: str = "comparison", **kwargs) -> plt.Figure:
        """
        Create a plot based on the specified type and data.
        
        Args:
            data: Data to plot
            plot_type: Type of plot to create
            **kwargs: Additional arguments for specific plot types
            
        Returns:
            matplotlib Figure object
        """
        if plot_type == "confusion_matrix":
            return self.plot_confusion_matrix(data, **kwargs)
        elif plot_type == "roc_curve":
            return self.plot_roc_curve(**data, **kwargs)
        elif plot_type == "feature_importance":
            return self.plot_feature_importance(data, **kwargs)
        elif plot_type == "comparison":
            return self.plot_model_comparison(data, **kwargs)
        elif plot_type == "cv_results":
            return self.plot_cross_validation_results(data, **kwargs)
        elif plot_type == "summary":
            return self.plot_performance_summary(data, **kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")