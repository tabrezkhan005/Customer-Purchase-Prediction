"""
Feature importance analyzer for tree-based and linear model analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Analyzer for extracting and ranking feature importance from different model types.
    Supports tree-based models (Random Forest, XGBoost) and linear models (Logistic Regression).
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize the feature importance analyzer.
        
        Args:
            feature_names: List of feature names corresponding to model features
        """
        self.feature_names = feature_names
        self.importance_cache = {}
        
    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Set or update feature names.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        logger.info(f"Updated feature names: {len(feature_names)} features")
    
    def analyze_tree_importance(self, model, model_name: str = "tree_model") -> Dict[str, float]:
        """
        Extract feature importance from tree-based models.
        
        Args:
            model: Trained tree-based model (RandomForest, XGBoost, etc.)
            model_name: Name identifier for the model
            
        Returns:
            Dict mapping feature names to importance scores
        """
        logger.info(f"Analyzing tree-based feature importance for {model_name}")
        
        # Get feature importance from the model
        if hasattr(model, 'get_feature_importance'):
            importance_scores = model.get_feature_importance()
        elif hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        elif hasattr(model, 'best_model') and hasattr(model.best_model, 'feature_importances_'):
            importance_scores = model.best_model.feature_importances_
        else:
            raise ValueError(f"Model {model_name} does not support feature importance extraction")
        
        if importance_scores is None:
            raise ValueError(f"Model {model_name} returned None for feature importance")
        
        # Create feature importance dictionary
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
        else:
            feature_names = self.feature_names
            
        if len(feature_names) != len(importance_scores):
            raise ValueError(f"Feature names length ({len(feature_names)}) does not match "
                           f"importance scores length ({len(importance_scores)})")
        
        importance_dict = dict(zip(feature_names, importance_scores))
        
        # Cache the results
        self.importance_cache[f"{model_name}_tree"] = importance_dict
        
        logger.info(f"Extracted {len(importance_dict)} feature importance scores for {model_name}")
        return importance_dict
    
    def analyze_linear_coefficients(self, model, model_name: str = "linear_model") -> Dict[str, float]:
        """
        Extract feature importance from linear models using coefficient magnitudes.
        
        Args:
            model: Trained linear model (LogisticRegression, etc.)
            model_name: Name identifier for the model
            
        Returns:
            Dict mapping feature names to importance scores (absolute coefficient values)
        """
        logger.info(f"Analyzing linear model coefficients for {model_name}")
        
        # Get coefficients from the model
        coefficients = None
        if hasattr(model, 'get_coefficients'):
            coefficients = model.get_coefficients()
        elif hasattr(model, 'coef_'):
            coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        elif hasattr(model, 'best_model') and hasattr(model.best_model, 'coef_'):
            coefficients = model.best_model.coef_[0] if len(model.best_model.coef_.shape) > 1 else model.best_model.coef_
        else:
            raise ValueError(f"Model {model_name} does not support coefficient extraction")
        
        if coefficients is None:
            raise ValueError(f"Model {model_name} returned None for coefficients")
        
        # Use absolute values as importance scores
        importance_scores = np.abs(coefficients)
        
        # Create feature importance dictionary
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance_scores))]
        else:
            feature_names = self.feature_names
            
        if len(feature_names) != len(importance_scores):
            raise ValueError(f"Feature names length ({len(feature_names)}) does not match "
                           f"coefficient length ({len(importance_scores)})")
        
        importance_dict = dict(zip(feature_names, importance_scores))
        
        # Cache the results
        self.importance_cache[f"{model_name}_linear"] = importance_dict
        
        logger.info(f"Extracted {len(importance_dict)} coefficient importance scores for {model_name}")
        return importance_dict
    
    def get_raw_coefficients(self, model, model_name: str = "linear_model") -> Dict[str, float]:
        """
        Get raw coefficients (with signs) from linear models.
        
        Args:
            model: Trained linear model
            model_name: Name identifier for the model
            
        Returns:
            Dict mapping feature names to raw coefficient values
        """
        logger.info(f"Extracting raw coefficients for {model_name}")
        
        # Get coefficients from the model
        coefficients = None
        if hasattr(model, 'get_coefficients'):
            coefficients = model.get_coefficients()
        elif hasattr(model, 'coef_'):
            coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        elif hasattr(model, 'best_model') and hasattr(model.best_model, 'coef_'):
            coefficients = model.best_model.coef_[0] if len(model.best_model.coef_.shape) > 1 else model.best_model.coef_
        else:
            raise ValueError(f"Model {model_name} does not support coefficient extraction")
        
        if coefficients is None:
            raise ValueError(f"Model {model_name} returned None for coefficients")
        
        # Create coefficient dictionary
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefficients))]
        else:
            feature_names = self.feature_names
            
        if len(feature_names) != len(coefficients):
            raise ValueError(f"Feature names length ({len(feature_names)}) does not match "
                           f"coefficient length ({len(coefficients)})")
        
        coefficient_dict = dict(zip(feature_names, coefficients))
        
        logger.info(f"Extracted {len(coefficient_dict)} raw coefficients for {model_name}")
        return coefficient_dict
    
    def rank_features(self, importance_scores: Dict[str, float], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Rank features by their importance scores in descending order.
        
        Args:
            importance_scores: Dictionary mapping feature names to importance scores
            top_k: Number of top features to return (None for all features)
            
        Returns:
            List of (feature_name, importance_score) tuples sorted by importance
        """
        logger.info(f"Ranking {len(importance_scores)} features by importance")
        
        # Sort features by importance score in descending order
        ranked_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k features if specified
        if top_k is not None:
            ranked_features = ranked_features[:top_k]
            logger.info(f"Returning top {len(ranked_features)} features")
        
        return ranked_features
    
    def compare_feature_importance(self, 
                                 model_importance_dict: Dict[str, Dict[str, float]],
                                 top_k: int = 10) -> pd.DataFrame:
        """
        Compare feature importance across multiple models.
        
        Args:
            model_importance_dict: Dict mapping model names to their feature importance dicts
            top_k: Number of top features to include in comparison
            
        Returns:
            DataFrame with features as rows and models as columns
        """
        logger.info(f"Comparing feature importance across {len(model_importance_dict)} models")
        
        # Get all unique features
        all_features = set()
        for importance_dict in model_importance_dict.values():
            all_features.update(importance_dict.keys())
        
        # Create comparison DataFrame
        comparison_data = {}
        for model_name, importance_dict in model_importance_dict.items():
            comparison_data[model_name] = [importance_dict.get(feature, 0.0) for feature in all_features]
        
        comparison_df = pd.DataFrame(comparison_data, index=list(all_features))
        
        # Calculate average importance across models
        comparison_df['average_importance'] = comparison_df.mean(axis=1)
        
        # Sort by average importance and take top k
        comparison_df = comparison_df.sort_values('average_importance', ascending=False)
        if top_k is not None:
            comparison_df = comparison_df.head(top_k)
        
        logger.info(f"Created comparison DataFrame with {len(comparison_df)} features")
        return comparison_df
    
    def get_feature_importance_summary(self, 
                                     model_importance_dict: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Generate a summary of feature importance across models.
        
        Args:
            model_importance_dict: Dict mapping model names to their feature importance dicts
            
        Returns:
            Dictionary containing summary statistics
        """
        logger.info("Generating feature importance summary")
        
        # Create comparison DataFrame
        comparison_df = self.compare_feature_importance(model_importance_dict, top_k=None)
        
        # Calculate summary statistics
        summary = {
            'total_features': len(comparison_df),
            'models_analyzed': list(model_importance_dict.keys()),
            'top_10_features': comparison_df.head(10).index.tolist(),
            'feature_importance_stats': {
                'mean_importance': comparison_df['average_importance'].mean(),
                'std_importance': comparison_df['average_importance'].std(),
                'max_importance': comparison_df['average_importance'].max(),
                'min_importance': comparison_df['average_importance'].min()
            },
            'model_agreement': self._calculate_model_agreement(comparison_df)
        }
        
        logger.info(f"Generated summary for {summary['total_features']} features across {len(summary['models_analyzed'])} models")
        return summary
    
    def _calculate_model_agreement(self, comparison_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate agreement between models on feature importance rankings.
        
        Args:
            comparison_df: DataFrame with feature importance comparison
            
        Returns:
            Dictionary with agreement metrics
        """
        model_columns = [col for col in comparison_df.columns if col != 'average_importance']
        
        if len(model_columns) < 2:
            return {'agreement_score': 1.0, 'rank_correlation': 1.0}
        
        # Calculate rank correlations between models
        correlations = []
        for i in range(len(model_columns)):
            for j in range(i + 1, len(model_columns)):
                model1_ranks = comparison_df[model_columns[i]].rank(ascending=False)
                model2_ranks = comparison_df[model_columns[j]].rank(ascending=False)
                correlation = model1_ranks.corr(model2_ranks, method='spearman')
                correlations.append(correlation)
        
        avg_correlation = np.mean(correlations) if correlations else 1.0
        
        # Calculate top-k agreement (how many of top 10 features are shared)
        top_k = min(10, len(comparison_df))
        top_features_per_model = {}
        for model in model_columns:
            top_features = comparison_df.nlargest(top_k, model).index.tolist()
            top_features_per_model[model] = set(top_features)
        
        # Calculate intersection of top features
        if len(top_features_per_model) > 1:
            intersection = set.intersection(*top_features_per_model.values())
            agreement_score = len(intersection) / top_k
        else:
            agreement_score = 1.0
        
        return {
            'agreement_score': agreement_score,
            'rank_correlation': avg_correlation,
            'top_k_analyzed': top_k
        }
    
    def get_cached_importance(self, model_name: str, model_type: str = None) -> Optional[Dict[str, float]]:
        """
        Retrieve cached feature importance results.
        
        Args:
            model_name: Name of the model
            model_type: Type of model ('tree' or 'linear')
            
        Returns:
            Cached importance dictionary or None if not found
        """
        if model_type:
            cache_key = f"{model_name}_{model_type}"
        else:
            # Try to find any cached result for this model
            cache_key = None
            for key in self.importance_cache.keys():
                if key.startswith(model_name):
                    cache_key = key
                    break
        
        if cache_key and cache_key in self.importance_cache:
            logger.info(f"Retrieved cached importance for {cache_key}")
            return self.importance_cache[cache_key]
        
        logger.warning(f"No cached importance found for {model_name}")
        return None
    
    def clear_cache(self) -> None:
        """Clear the importance cache."""
        self.importance_cache.clear()
        logger.info("Cleared feature importance cache")