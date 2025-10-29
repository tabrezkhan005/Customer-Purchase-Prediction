"""
Business insight generator to translate technical findings into actionable business insights.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BusinessInsightGenerator:
    """
    Generator for translating technical feature importance findings into business insights
    and actionable recommendations for marketing and website optimization.
    """
    
    def __init__(self):
        """Initialize the business insight generator."""
        self.feature_business_mapping = self._create_feature_business_mapping()
        self.insight_templates = self._create_insight_templates()
        
    def _create_feature_business_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Create mapping between technical feature names and business interpretations.
        
        Returns:
            Dictionary mapping feature names to business context
        """
        return {
            # Page visit metrics
            'Administrative': {
                'category': 'Navigation Behavior',
                'description': 'Number of administrative pages visited',
                'business_meaning': 'Account management and administrative engagement'
            },
            'Administrative_Duration': {
                'category': 'Navigation Behavior', 
                'description': 'Time spent on administrative pages',
                'business_meaning': 'Depth of administrative engagement'
            },
            'Informational': {
                'category': 'Navigation Behavior',
                'description': 'Number of informational pages visited',
                'business_meaning': 'Information seeking and research behavior'
            },
            'Informational_Duration': {
                'category': 'Navigation Behavior',
                'description': 'Time spent on informational pages', 
                'business_meaning': 'Depth of information consumption'
            },
            'ProductRelated': {
                'category': 'Product Engagement',
                'description': 'Number of product-related pages visited',
                'business_meaning': 'Product browsing and shopping intent'
            },
            'ProductRelated_Duration': {
                'category': 'Product Engagement',
                'description': 'Time spent on product-related pages',
                'business_meaning': 'Product consideration and evaluation time'
            },
            
            # Engagement metrics
            'BounceRates': {
                'category': 'Engagement Quality',
                'description': 'Average bounce rate of visited pages',
                'business_meaning': 'Page relevance and user engagement quality'
            },
            'ExitRates': {
                'category': 'Engagement Quality', 
                'description': 'Average exit rate of visited pages',
                'business_meaning': 'Session termination patterns and page effectiveness'
            },
            'PageValues': {
                'category': 'Conversion Potential',
                'description': 'Average page value of visited pages',
                'business_meaning': 'Revenue potential and conversion pathway effectiveness'
            },
            
            # Temporal and contextual factors
            'SpecialDay': {
                'category': 'Temporal Context',
                'description': 'Closeness to special day (holiday)',
                'business_meaning': 'Seasonal shopping patterns and holiday influence'
            },
            'Month': {
                'category': 'Temporal Context',
                'description': 'Month of the visit',
                'business_meaning': 'Seasonal trends and monthly shopping patterns'
            },
            'Weekend': {
                'category': 'Temporal Context',
                'description': 'Whether visit occurred on weekend',
                'business_meaning': 'Weekend vs weekday shopping behavior'
            },
            
            # Technical and user characteristics
            'OperatingSystems': {
                'category': 'Technical Profile',
                'description': 'Operating system used',
                'business_meaning': 'User technical profile and device preferences'
            },
            'Browser': {
                'category': 'Technical Profile',
                'description': 'Browser used for the session',
                'business_meaning': 'Browser preferences and technical compatibility'
            },
            'Region': {
                'category': 'Geographic Profile',
                'description': 'Geographic region of the user',
                'business_meaning': 'Regional preferences and market segments'
            },
            'TrafficType': {
                'category': 'Acquisition Channel',
                'description': 'Traffic source type',
                'business_meaning': 'Marketing channel effectiveness and user acquisition'
            },
            'VisitorType': {
                'category': 'User Loyalty',
                'description': 'Type of visitor (new, returning, other)',
                'business_meaning': 'Customer loyalty and repeat engagement patterns'
            }
        }
    
    def _create_insight_templates(self) -> Dict[str, List[str]]:
        """
        Create templates for generating business insights based on feature importance.
        
        Returns:
            Dictionary of insight templates by category
        """
        return {
            'high_importance': [
                "The {feature_name} is a critical factor in purchase decisions, suggesting that {business_meaning}.",
                "Strong influence of {feature_name} indicates that optimizing {business_meaning} could significantly impact conversions.",
                "High importance of {feature_name} reveals that customers who {business_meaning} are more likely to purchase."
            ],
            'product_engagement': [
                "Product page engagement metrics are key drivers, indicating that improving product presentation and information quality is crucial.",
                "Time spent on product pages strongly predicts purchases, suggesting the need for compelling product content and user experience.",
                "Product browsing behavior is highly predictive, emphasizing the importance of product discovery and navigation optimization."
            ],
            'navigation_behavior': [
                "Administrative page engagement affects purchase likelihood, suggesting that account management features influence buying decisions.",
                "Information-seeking behavior impacts conversions, indicating the value of comprehensive product information and educational content.",
                "Navigation patterns reveal customer journey preferences that can be optimized for better conversion rates."
            ],
            'engagement_quality': [
                "Page engagement quality metrics are strong predictors, highlighting the importance of relevant, engaging content.",
                "Bounce and exit rates significantly influence purchases, emphasizing the need for compelling page experiences.",
                "User engagement depth correlates with purchase intent, suggesting focus on content quality and user experience."
            ],
            'temporal_patterns': [
                "Seasonal and temporal factors play important roles, indicating opportunities for targeted timing strategies.",
                "Holiday and special day proximity affects purchase behavior, suggesting seasonal marketing optimization opportunities.",
                "Time-based patterns reveal optimal periods for promotions and marketing campaigns."
            ],
            'user_segmentation': [
                "Visitor type and loyalty patterns are significant factors, indicating the value of personalized experiences for different user segments.",
                "Geographic and demographic factors influence purchases, suggesting opportunities for regional customization.",
                "User technical profiles correlate with buying behavior, indicating potential for device-specific optimizations."
            ]
        }
    
    def generate_insights(self, 
                         feature_rankings: List[Tuple[str, float]], 
                         top_k: int = 10) -> List[str]:
        """
        Generate actionable business insights from feature importance rankings.
        
        Args:
            feature_rankings: List of (feature_name, importance_score) tuples sorted by importance
            top_k: Number of top features to analyze for insights
            
        Returns:
            List of business insight strings
        """
        logger.info(f"Generating business insights for top {top_k} features")
        
        insights = []
        top_features = feature_rankings[:top_k]
        
        # Analyze top features by category
        category_importance = self._analyze_category_importance(top_features)
        
        # Generate category-level insights
        insights.extend(self._generate_category_insights(category_importance))
        
        # Generate specific feature insights
        insights.extend(self._generate_feature_specific_insights(top_features[:5]))
        
        # Generate actionable recommendations
        insights.extend(self._generate_actionable_recommendations(top_features, category_importance))
        
        logger.info(f"Generated {len(insights)} business insights")
        return insights
    
    def _analyze_category_importance(self, 
                                   feature_rankings: List[Tuple[str, float]]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze importance by business category.
        
        Args:
            feature_rankings: List of (feature_name, importance_score) tuples
            
        Returns:
            Dictionary with category analysis
        """
        category_scores = {}
        category_features = {}
        
        for feature_name, importance_score in feature_rankings:
            # Get business mapping for this feature
            feature_info = self.feature_business_mapping.get(feature_name, {
                'category': 'Other',
                'description': feature_name,
                'business_meaning': f'Technical metric: {feature_name}'
            })
            
            category = feature_info['category']
            
            if category not in category_scores:
                category_scores[category] = []
                category_features[category] = []
            
            category_scores[category].append(importance_score)
            category_features[category].append((feature_name, importance_score))
        
        # Calculate category statistics
        category_analysis = {}
        for category, scores in category_scores.items():
            category_analysis[category] = {
                'total_importance': sum(scores),
                'avg_importance': np.mean(scores),
                'max_importance': max(scores),
                'feature_count': len(scores),
                'features': category_features[category]
            }
        
        return category_analysis
    
    def _generate_category_insights(self, 
                                  category_importance: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate insights based on category-level importance.
        
        Args:
            category_importance: Dictionary with category analysis
            
        Returns:
            List of category-level insights
        """
        insights = []
        
        # Sort categories by total importance
        sorted_categories = sorted(category_importance.items(), 
                                 key=lambda x: x[1]['total_importance'], 
                                 reverse=True)
        
        # Generate insights for top categories
        for i, (category, analysis) in enumerate(sorted_categories[:3]):
            if category == 'Product Engagement':
                insights.append(
                    f"Product engagement is the #{i+1} most important factor, with {analysis['feature_count']} "
                    f"key metrics. Focus on improving product page experience, content quality, and user engagement."
                )
            elif category == 'Navigation Behavior':
                insights.append(
                    f"Navigation behavior ranks #{i+1} in importance, indicating that how users move through "
                    f"your site significantly impacts purchase decisions. Optimize site structure and user flows."
                )
            elif category == 'Engagement Quality':
                insights.append(
                    f"Engagement quality metrics rank #{i+1}, showing that page relevance and user experience "
                    f"are critical. Focus on reducing bounce rates and improving content engagement."
                )
            elif category == 'Temporal Context':
                insights.append(
                    f"Temporal factors rank #{i+1}, revealing significant seasonal and timing effects. "
                    f"Develop time-based marketing strategies and seasonal optimizations."
                )
            else:
                insights.append(
                    f"{category} factors rank #{i+1} in importance, contributing {analysis['total_importance']:.3f} "
                    f"to the overall predictive power through {analysis['feature_count']} key metrics."
                )
        
        return insights
    
    def _generate_feature_specific_insights(self, 
                                          top_features: List[Tuple[str, float]]) -> List[str]:
        """
        Generate insights for specific high-importance features.
        
        Args:
            top_features: List of top (feature_name, importance_score) tuples
            
        Returns:
            List of feature-specific insights
        """
        insights = []
        
        for i, (feature_name, importance_score) in enumerate(top_features):
            feature_info = self.feature_business_mapping.get(feature_name, {
                'category': 'Other',
                'description': feature_name,
                'business_meaning': f'Technical metric: {feature_name}'
            })
            
            insight = (
                f"#{i+1} most important factor: {feature_info['description']} "
                f"(importance: {importance_score:.3f}). This indicates that "
                f"{feature_info['business_meaning'].lower()} strongly influences purchase decisions."
            )
            insights.append(insight)
        
        return insights
    
    def _generate_actionable_recommendations(self, 
                                           feature_rankings: List[Tuple[str, float]],
                                           category_importance: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate actionable business recommendations based on feature importance.
        
        Args:
            feature_rankings: List of (feature_name, importance_score) tuples
            category_importance: Dictionary with category analysis
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Get top categories
        top_categories = sorted(category_importance.items(), 
                              key=lambda x: x[1]['total_importance'], 
                              reverse=True)[:3]
        
        for category, analysis in top_categories:
            if category == 'Product Engagement':
                recommendations.extend([
                    "RECOMMENDATION: Enhance product page content with detailed descriptions, high-quality images, and customer reviews.",
                    "RECOMMENDATION: Implement product recommendation engines to increase product page engagement.",
                    "RECOMMENDATION: A/B test product page layouts to optimize time spent and reduce exit rates."
                ])
            elif category == 'Navigation Behavior':
                recommendations.extend([
                    "RECOMMENDATION: Analyze and optimize user navigation paths to reduce friction in the customer journey.",
                    "RECOMMENDATION: Improve site search functionality and category navigation to help users find products faster.",
                    "RECOMMENDATION: Implement breadcrumbs and clear navigation aids to improve user experience."
                ])
            elif category == 'Engagement Quality':
                recommendations.extend([
                    "RECOMMENDATION: Audit pages with high bounce rates and improve content relevance and loading speed.",
                    "RECOMMENDATION: Implement exit-intent popups or offers to retain users who are about to leave.",
                    "RECOMMENDATION: Use heat mapping and user session recordings to identify engagement issues."
                ])
            elif category == 'Temporal Context':
                recommendations.extend([
                    "RECOMMENDATION: Develop seasonal marketing campaigns aligned with high-conversion periods.",
                    "RECOMMENDATION: Adjust inventory and promotions based on monthly and holiday shopping patterns.",
                    "RECOMMENDATION: Create targeted weekend vs weekday marketing strategies."
                ])
        
        # Add specific feature-based recommendations
        top_features = dict(feature_rankings[:5])
        
        if 'PageValues' in top_features:
            recommendations.append(
                "RECOMMENDATION: Focus marketing spend on high-value pages and optimize conversion paths from these pages."
            )
        
        if 'VisitorType' in top_features:
            recommendations.append(
                "RECOMMENDATION: Implement personalized experiences for returning vs new visitors to maximize conversion rates."
            )
        
        if any('Duration' in feature for feature, _ in feature_rankings[:10]):
            recommendations.append(
                "RECOMMENDATION: Optimize page loading speeds and content engagement to increase time spent on key pages."
            )
        
        return recommendations
    
    def create_executive_summary(self, 
                               feature_rankings: List[Tuple[str, float]],
                               model_performance: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Create an executive summary of key findings and recommendations.
        
        Args:
            feature_rankings: List of (feature_name, importance_score) tuples
            model_performance: Optional dictionary with model performance metrics
            
        Returns:
            Dictionary containing executive summary
        """
        logger.info("Creating executive summary")
        
        # Analyze categories
        category_importance = self._analyze_category_importance(feature_rankings[:15])
        
        # Get top insights
        key_insights = self.generate_insights(feature_rankings, top_k=10)
        
        # Create summary
        summary = {
            'key_findings': {
                'top_3_factors': [
                    {
                        'feature': feature_rankings[i][0],
                        'importance': feature_rankings[i][1],
                        'business_meaning': self.feature_business_mapping.get(
                            feature_rankings[i][0], {}
                        ).get('business_meaning', 'Technical factor')
                    }
                    for i in range(min(3, len(feature_rankings)))
                ],
                'dominant_categories': [
                    {
                        'category': category,
                        'importance': analysis['total_importance'],
                        'feature_count': analysis['feature_count']
                    }
                    for category, analysis in sorted(
                        category_importance.items(),
                        key=lambda x: x[1]['total_importance'],
                        reverse=True
                    )[:3]
                ]
            },
            'business_insights': key_insights[:5],  # Top 5 insights
            'recommendations': self._generate_actionable_recommendations(
                feature_rankings, category_importance
            )[:5],  # Top 5 recommendations
            'model_performance': model_performance or {},
            'analysis_scope': {
                'total_features_analyzed': len(feature_rankings),
                'categories_identified': len(category_importance),
                'insights_generated': len(key_insights)
            }
        }
        
        logger.info("Executive summary created successfully")
        return summary
    
    def get_feature_business_context(self, feature_name: str) -> Dict[str, str]:
        """
        Get business context for a specific feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary with business context information
        """
        return self.feature_business_mapping.get(feature_name, {
            'category': 'Other',
            'description': feature_name,
            'business_meaning': f'Technical metric: {feature_name}'
        })
    
    def add_custom_feature_mapping(self, 
                                 feature_name: str, 
                                 category: str, 
                                 description: str, 
                                 business_meaning: str) -> None:
        """
        Add custom business mapping for a feature.
        
        Args:
            feature_name: Name of the feature
            category: Business category
            description: Technical description
            business_meaning: Business interpretation
        """
        self.feature_business_mapping[feature_name] = {
            'category': category,
            'description': description,
            'business_meaning': business_meaning
        }
        logger.info(f"Added custom mapping for feature: {feature_name}")