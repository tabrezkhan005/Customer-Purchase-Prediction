"""
Data loading and validation module for customer purchase prediction.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from .base import BaseDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader(BaseDataLoader):
    """
    DataLoader class to read and validate the online shoppers intention CSV dataset.
    Handles data type validation, basic statistics generation, and error handling.
    """
    
    def __init__(self):
        """Initialize DataLoader with expected schema."""
        self.expected_columns = [
            'Administrative', 'Administrative_Duration', 'Informational', 
            'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
            'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month',
            'OperatingSystems', 'Browser', 'Region', 'TrafficType', 
            'VisitorType', 'Weekend', 'Revenue'
        ]
        
        self.expected_dtypes = {
            'Administrative': 'int64',
            'Administrative_Duration': 'float64',
            'Informational': 'int64',
            'Informational_Duration': 'float64',
            'ProductRelated': 'int64',
            'ProductRelated_Duration': 'float64',
            'BounceRates': 'float64',
            'ExitRates': 'float64',
            'PageValues': 'float64',
            'SpecialDay': 'float64',
            'Month': 'object',
            'OperatingSystems': 'int64',
            'Browser': 'int64',
            'Region': 'int64',
            'TrafficType': 'int64',
            'VisitorType': 'object',
            'Weekend': 'bool',
            'Revenue': 'bool'
        }
        
        self.numeric_columns = [
            'Administrative', 'Administrative_Duration', 'Informational',
            'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
            'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
            'OperatingSystems', 'Browser', 'Region', 'TrafficType'
        ]
        
        self.categorical_columns = ['Month', 'VisitorType']
        self.boolean_columns = ['Weekend', 'Revenue']
        
        self.data_stats = {}
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file with comprehensive error handling.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded and validated dataset
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
            Exception: For other loading errors
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Dataset file not found: {filepath}")
            
            # Check file extension
            if not filepath.lower().endswith('.csv'):
                raise ValueError(f"Invalid file format. Expected CSV file, got: {filepath}")
            
            logger.info(f"Loading data from {filepath}")
            
            # Load the CSV file
            data = pd.read_csv(filepath)
            
            # Log basic info about loaded data
            logger.info(f"Successfully loaded data with shape: {data.shape}")
            
            # Validate the loaded data
            if not self.validate_data(data):
                raise ValueError("Data validation failed")
            
            # Generate basic statistics
            self.data_stats = self._generate_basic_statistics(data)
            logger.info("Data validation and statistics generation completed")
            
            return data
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Empty CSV file: {filepath}")
            raise ValueError(f"The CSV file is empty: {filepath}")
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file: {e}")
            raise ValueError(f"Invalid CSV format in file {filepath}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading data: {e}")
            raise Exception(f"Failed to load data from {filepath}: {e}")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the loaded data against expected schema and constraints.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            # Check if DataFrame is empty
            if data.empty:
                logger.error("Dataset is empty")
                return False
            
            # Check expected columns
            missing_columns = set(self.expected_columns) - set(data.columns)
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            extra_columns = set(data.columns) - set(self.expected_columns)
            if extra_columns:
                logger.warning(f"Extra columns found (will be ignored): {extra_columns}")
            
            # Validate data types for key columns
            validation_errors = []
            
            # Check numeric columns
            for col in self.numeric_columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        validation_errors.append(f"Column {col} should be numeric")
            
            # Check categorical columns
            for col in self.categorical_columns:
                if col in data.columns:
                    if not pd.api.types.is_object_dtype(data[col]) and not pd.api.types.is_string_dtype(data[col]):
                        validation_errors.append(f"Column {col} should be categorical/string")
            
            # Check boolean columns
            for col in self.boolean_columns:
                if col in data.columns:
                    if not pd.api.types.is_bool_dtype(data[col]):
                        # Try to convert if it's string representation of boolean
                        if data[col].dtype == 'object':
                            unique_vals = data[col].dropna().unique()
                            bool_vals = {'TRUE', 'FALSE', 'True', 'False', 'true', 'false', '1', '0'}
                            if not set(str(v) for v in unique_vals).issubset(bool_vals):
                                validation_errors.append(f"Column {col} should be boolean")
                        else:
                            validation_errors.append(f"Column {col} should be boolean")
            
            # Check for reasonable data ranges
            if 'Administrative' in data.columns:
                if data['Administrative'].min() < 0:
                    validation_errors.append("Administrative pages should be non-negative")
            
            if 'BounceRates' in data.columns:
                if data['BounceRates'].min() < 0 or data['BounceRates'].max() > 1:
                    logger.warning("BounceRates values outside expected range [0,1]")
            
            if 'ExitRates' in data.columns:
                if data['ExitRates'].min() < 0 or data['ExitRates'].max() > 1:
                    logger.warning("ExitRates values outside expected range [0,1]")
            
            if validation_errors:
                logger.error(f"Data validation errors: {validation_errors}")
                return False
            
            logger.info("Data validation passed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during data validation: {e}")
            return False
    
    def _generate_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive basic statistics for the dataset.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dict containing various statistics
        """
        stats = {
            'shape': data.shape,
            'total_records': len(data),
            'total_features': len(data.columns),
            'missing_values': {},
            'data_types': {},
            'numeric_stats': {},
            'categorical_stats': {},
            'target_distribution': {}
        }
        
        # Missing values analysis
        missing_counts = data.isnull().sum()
        stats['missing_values'] = {
            'total_missing': missing_counts.sum(),
            'missing_by_column': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentage': (missing_counts / len(data) * 100).round(2).to_dict()
        }
        
        # Data types
        stats['data_types'] = data.dtypes.astype(str).to_dict()
        
        # Numeric statistics
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            stats['numeric_stats'] = {
                'summary': numeric_data.describe().to_dict(),
                'correlations': numeric_data.corr().to_dict() if len(numeric_data.columns) > 1 else {}
            }
        
        # Categorical statistics
        categorical_data = data.select_dtypes(include=['object', 'category'])
        if not categorical_data.empty:
            cat_stats = {}
            for col in categorical_data.columns:
                cat_stats[col] = {
                    'unique_count': data[col].nunique(),
                    'unique_values': data[col].unique().tolist()[:10],  # Limit to first 10
                    'value_counts': data[col].value_counts().head(10).to_dict()
                }
            stats['categorical_stats'] = cat_stats
        
        # Target variable distribution (Revenue)
        if 'Revenue' in data.columns:
            revenue_counts = data['Revenue'].value_counts()
            stats['target_distribution'] = {
                'counts': revenue_counts.to_dict(),
                'percentages': (revenue_counts / len(data) * 100).round(2).to_dict(),
                'class_balance_ratio': revenue_counts.min() / revenue_counts.max()
            }
        
        return stats
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get the generated data statistics.
        
        Returns:
            Dict containing data statistics
        """
        return self.data_stats
    
    def print_data_summary(self) -> None:
        """Print a formatted summary of the data statistics."""
        if not self.data_stats:
            print("No statistics available. Load data first.")
            return
        
        print("=" * 60)
        print("DATA SUMMARY REPORT")
        print("=" * 60)
        
        # Basic info
        print(f"Dataset Shape: {self.data_stats['shape']}")
        print(f"Total Records: {self.data_stats['total_records']:,}")
        print(f"Total Features: {self.data_stats['total_features']}")
        
        # Missing values
        print(f"\nMissing Values: {self.data_stats['missing_values']['total_missing']}")
        if self.data_stats['missing_values']['missing_by_column']:
            print("Missing by column:")
            for col, count in self.data_stats['missing_values']['missing_by_column'].items():
                pct = self.data_stats['missing_values']['missing_percentage'][col]
                print(f"  {col}: {count} ({pct}%)")
        
        # Target distribution
        if 'target_distribution' in self.data_stats and self.data_stats['target_distribution']:
            print(f"\nTarget Variable (Revenue) Distribution:")
            for value, count in self.data_stats['target_distribution']['counts'].items():
                pct = self.data_stats['target_distribution']['percentages'][value]
                print(f"  {value}: {count:,} ({pct}%)")
            print(f"Class Balance Ratio: {self.data_stats['target_distribution']['class_balance_ratio']:.3f}")
        
        # Categorical features summary
        if self.data_stats['categorical_stats']:
            print(f"\nCategorical Features:")
            for col, stats in self.data_stats['categorical_stats'].items():
                print(f"  {col}: {stats['unique_count']} unique values")
        
        print("=" * 60)