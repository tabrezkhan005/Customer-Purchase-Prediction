"""
Exploratory Data Analysis (EDA) generator for customer purchase prediction.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import warnings
import logging
from pathlib import Path

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EDAGenerator:
    """
    EDAGenerator class to create distribution plots, correlation matrices, 
    and feature relationship visualizations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), save_plots: bool = True):
        """
        Initialize EDAGenerator.
        
        Args:
            figsize: Default figure size for plots
            save_plots: Whether to save plots to files
        """
        self.figsize = figsize
        self.save_plots = save_plots
        self.plots_dir = Path("plots")
        self.eda_results = {}
        
        # Create plots directory if saving plots
        if self.save_plots:
            self.plots_dir.mkdir(exist_ok=True)
    
    def generate_complete_eda(self, data: pd.DataFrame, target_column: str = 'Revenue') -> Dict[str, Any]:
        """
        Generate complete EDA analysis including all visualizations and statistics.
        
        Args:
            data: DataFrame to analyze
            target_column: Name of target variable
            
        Returns:
            Dict containing all EDA results
        """
        logger.info("Starting complete EDA analysis...")
        
        self.eda_results = {
            'data_overview': self._generate_data_overview(data),
            'missing_values_analysis': self._analyze_missing_values(data),
            'target_analysis': self._analyze_target_variable(data, target_column),
            'numeric_analysis': self._analyze_numeric_features(data, target_column),
            'categorical_analysis': self._analyze_categorical_features(data, target_column),
            'correlation_analysis': self._analyze_correlations(data),
            'feature_relationships': self._analyze_feature_relationships(data, target_column)
        }
        
        logger.info("Complete EDA analysis finished")
        return self.eda_results
    
    def _generate_data_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic data overview and summary statistics."""
        logger.info("Generating data overview...")
        
        overview = {
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'data_types': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'summary_stats': data.describe(include='all').to_dict()
        }
        
        # Create data types visualization
        plt.figure(figsize=self.figsize)
        dtype_counts = data.dtypes.value_counts()
        plt.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Data Types')
        
        if self.save_plots:
            plt.savefig(self.plots_dir / 'data_types_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return overview
    
    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values patterns."""
        logger.info("Analyzing missing values...")
        
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        missing_analysis = {
            'total_missing': missing_counts.sum(),
            'missing_by_column': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict()
        }
        
        # Visualize missing values if any exist
        if missing_counts.sum() > 0:
            plt.figure(figsize=self.figsize)
            missing_data = missing_counts[missing_counts > 0].sort_values(ascending=True)
            
            plt.barh(range(len(missing_data)), missing_data.values)
            plt.yticks(range(len(missing_data)), missing_data.index)
            plt.xlabel('Number of Missing Values')
            plt.title('Missing Values by Column')
            plt.grid(axis='x', alpha=0.3)
            
            if self.save_plots:
                plt.savefig(self.plots_dir / 'missing_values.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            logger.info("No missing values found in the dataset")
        
        return missing_analysis
    
    def _analyze_target_variable(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze target variable distribution."""
        logger.info(f"Analyzing target variable: {target_column}")
        
        if target_column not in data.columns:
            logger.warning(f"Target column {target_column} not found in data")
            return {}
        
        target_counts = data[target_column].value_counts()
        target_percentages = (target_counts / len(data)) * 100
        
        target_analysis = {
            'value_counts': target_counts.to_dict(),
            'percentages': target_percentages.to_dict(),
            'class_balance_ratio': target_counts.min() / target_counts.max()
        }
        
        # Create target distribution visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        target_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])
        ax1.set_title(f'{target_column} Distribution (Counts)')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        ax2.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', 
                colors=['skyblue', 'lightcoral'])
        ax2.set_title(f'{target_column} Distribution (Percentages)')
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.plots_dir / f'{target_column}_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return target_analysis
    
    def _analyze_numeric_features(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze numeric features and their relationship with target."""
        logger.info("Analyzing numeric features...")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_columns:
            numeric_columns.remove(target_column)
        
        numeric_analysis = {
            'columns': numeric_columns,
            'summary_stats': data[numeric_columns].describe().to_dict(),
            'skewness': data[numeric_columns].skew().to_dict(),
            'kurtosis': data[numeric_columns].kurtosis().to_dict()
        }
        
        # Create distribution plots for numeric features
        if numeric_columns:
            n_cols = 4
            n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for i, col in enumerate(numeric_columns):
                if i < len(axes):
                    data[col].hist(bins=30, ax=axes[i], alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(numeric_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            if self.save_plots:
                plt.savefig(self.plots_dir / 'numeric_distributions.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Box plots by target variable
            if target_column in data.columns and len(numeric_columns) > 0:
                n_cols = 3
                n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
                axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
                
                for i, col in enumerate(numeric_columns):
                    if i < len(axes):
                        data.boxplot(column=col, by=target_column, ax=axes[i])
                        axes[i].set_title(f'{col} by {target_column}')
                        axes[i].set_xlabel(target_column)
                        axes[i].set_ylabel(col)
                
                # Hide unused subplots
                for i in range(len(numeric_columns), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                if self.save_plots:
                    plt.savefig(self.plots_dir / 'numeric_boxplots_by_target.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        return numeric_analysis
    
    def _analyze_categorical_features(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze categorical features and their relationship with target."""
        logger.info("Analyzing categorical features...")
        
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column in categorical_columns:
            categorical_columns.remove(target_column)
        
        categorical_analysis = {
            'columns': categorical_columns,
            'unique_counts': {col: data[col].nunique() for col in categorical_columns},
            'value_counts': {col: data[col].value_counts().to_dict() for col in categorical_columns}
        }
        
        # Create visualizations for categorical features
        if categorical_columns:
            # Value counts for each categorical feature
            n_cols = 2
            n_rows = (len(categorical_columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for i, col in enumerate(categorical_columns):
                if i < len(axes):
                    value_counts = data[col].value_counts().head(10)  # Top 10 categories
                    value_counts.plot(kind='bar', ax=axes[i], color='lightgreen')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Count')
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].grid(axis='y', alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(categorical_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            if self.save_plots:
                plt.savefig(self.plots_dir / 'categorical_distributions.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Stacked bar plots by target variable
            if target_column in data.columns:
                for col in categorical_columns:
                    plt.figure(figsize=(12, 6))
                    
                    # Create crosstab
                    crosstab = pd.crosstab(data[col], data[target_column], normalize='index') * 100
                    
                    crosstab.plot(kind='bar', stacked=True, ax=plt.gca(), 
                                color=['lightcoral', 'skyblue'])
                    plt.title(f'{col} vs {target_column} (Percentage)')
                    plt.xlabel(col)
                    plt.ylabel('Percentage')
                    plt.legend(title=target_column)
                    plt.xticks(rotation=45)
                    plt.grid(axis='y', alpha=0.3)
                    
                    if self.save_plots:
                        plt.savefig(self.plots_dir / f'{col}_vs_{target_column}.png', 
                                  dpi=300, bbox_inches='tight')
                    plt.show()
        
        return categorical_analysis
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric features."""
        logger.info("Analyzing feature correlations...")
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            logger.warning("No numeric columns found for correlation analysis")
            return {}
        
        correlation_matrix = numeric_data.corr()
        
        correlation_analysis = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': self._find_high_correlations(correlation_matrix)
        }
        
        # Create correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.plots_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_analysis
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find pairs of features with high correlation."""
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))
        
        return high_corr_pairs
    
    def _analyze_feature_relationships(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Analyze relationships between features and target variable."""
        logger.info("Analyzing feature relationships with target...")
        
        if target_column not in data.columns:
            logger.warning(f"Target column {target_column} not found")
            return {}
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_columns:
            numeric_columns.remove(target_column)
        
        relationships = {
            'target_correlations': {},
            'feature_importance_proxy': {}
        }
        
        # Calculate correlations with target (if target is numeric)
        if pd.api.types.is_numeric_dtype(data[target_column]):
            target_corr = data[numeric_columns + [target_column]].corr()[target_column].drop(target_column)
            relationships['target_correlations'] = target_corr.to_dict()
            
            # Plot correlations with target
            plt.figure(figsize=(12, 8))
            target_corr_sorted = target_corr.abs().sort_values(ascending=True)
            
            colors = ['red' if x < 0 else 'blue' for x in target_corr[target_corr_sorted.index]]
            plt.barh(range(len(target_corr_sorted)), target_corr_sorted.values, color=colors, alpha=0.7)
            plt.yticks(range(len(target_corr_sorted)), target_corr_sorted.index)
            plt.xlabel('Absolute Correlation with Target')
            plt.title(f'Feature Correlations with {target_column}')
            plt.grid(axis='x', alpha=0.3)
            
            if self.save_plots:
                plt.savefig(self.plots_dir / f'target_correlations.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # For binary target, calculate mean differences
        elif data[target_column].dtype == 'bool' or data[target_column].nunique() == 2:
            mean_differences = {}
            
            for col in numeric_columns:
                group_means = data.groupby(target_column)[col].mean()
                if len(group_means) == 2:
                    mean_diff = abs(group_means.iloc[1] - group_means.iloc[0])
                    mean_differences[col] = mean_diff
            
            relationships['feature_importance_proxy'] = mean_differences
            
            # Plot mean differences
            if mean_differences:
                plt.figure(figsize=(12, 8))
                sorted_diffs = dict(sorted(mean_differences.items(), key=lambda x: x[1], reverse=True))
                
                plt.barh(range(len(sorted_diffs)), list(sorted_diffs.values()), color='green', alpha=0.7)
                plt.yticks(range(len(sorted_diffs)), list(sorted_diffs.keys()))
                plt.xlabel('Mean Difference Between Classes')
                plt.title(f'Feature Discrimination Power for {target_column}')
                plt.grid(axis='x', alpha=0.3)
                
                if self.save_plots:
                    plt.savefig(self.plots_dir / f'feature_discrimination.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        return relationships
    
    def generate_summary_report(self) -> str:
        """Generate a text summary of EDA findings."""
        if not self.eda_results:
            return "No EDA results available. Run generate_complete_eda() first."
        
        report = []
        report.append("=" * 80)
        report.append("EXPLORATORY DATA ANALYSIS SUMMARY REPORT")
        report.append("=" * 80)
        
        # Data overview
        if 'data_overview' in self.eda_results:
            overview = self.eda_results['data_overview']
            report.append(f"\nDATA OVERVIEW:")
            report.append(f"Dataset Shape: {overview['shape']}")
            report.append(f"Total Features: {len(overview['columns'])}")
            report.append(f"Memory Usage: {overview['memory_usage'] / 1024**2:.2f} MB")
        
        # Missing values
        if 'missing_values_analysis' in self.eda_results:
            missing = self.eda_results['missing_values_analysis']
            report.append(f"\nMISSING VALUES:")
            report.append(f"Total Missing: {missing['total_missing']}")
            if missing['missing_by_column']:
                report.append("Columns with missing values:")
                for col, count in missing['missing_by_column'].items():
                    pct = missing['missing_percentages'][col]
                    report.append(f"  {col}: {count} ({pct:.1f}%)")
        
        # Target analysis
        if 'target_analysis' in self.eda_results:
            target = self.eda_results['target_analysis']
            report.append(f"\nTARGET VARIABLE ANALYSIS:")
            if 'value_counts' in target:
                for value, count in target['value_counts'].items():
                    pct = target['percentages'][value]
                    report.append(f"  {value}: {count:,} ({pct:.1f}%)")
                report.append(f"Class Balance Ratio: {target['class_balance_ratio']:.3f}")
        
        # High correlations
        if 'correlation_analysis' in self.eda_results:
            corr = self.eda_results['correlation_analysis']
            if 'high_correlations' in corr and corr['high_correlations']:
                report.append(f"\nHIGH CORRELATIONS (>0.7):")
                for feat1, feat2, corr_val in corr['high_correlations']:
                    report.append(f"  {feat1} - {feat2}: {corr_val:.3f}")
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def save_results(self, filepath: str = "eda_results.json") -> None:
        """Save EDA results to JSON file."""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            return obj
        
        # Deep convert the results
        json_results = json.loads(json.dumps(self.eda_results, default=convert_numpy))
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"EDA results saved to {filepath}")