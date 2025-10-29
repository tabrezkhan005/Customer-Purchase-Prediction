"""
Data cleaning and preprocessing pipeline for customer purchase prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging
from .base import BaseDataProcessor, BaseFeatureEncoder, BaseFeatureScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner(BaseDataProcessor):
    """
    DataCleaner class to handle missing values with configurable imputation strategies.
    """
    
    def __init__(self, 
                 numeric_strategy: str = 'median',
                 categorical_strategy: str = 'most_frequent',
                 **kwargs):
        """
        Initialize DataCleaner with imputation strategies.
        
        Args:
            numeric_strategy: Strategy for numeric features ('mean', 'median', 'constant')
            categorical_strategy: Strategy for categorical features ('most_frequent', 'constant')
        """
        super().__init__(**kwargs)
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.boolean_columns = []
        
    def fit(self, data: pd.DataFrame) -> 'DataCleaner':
        """
        Fit the data cleaner to the training data.
        
        Args:
            data: Training DataFrame
            
        Returns:
            self: Fitted DataCleaner instance
        """
        logger.info("Fitting DataCleaner...")
        
        # Identify column types
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.boolean_columns = data.select_dtypes(include=['bool']).columns.tolist()
        
        # Remove target column if present
        if 'Revenue' in self.numeric_columns:
            self.numeric_columns.remove('Revenue')
        if 'Revenue' in self.boolean_columns:
            self.boolean_columns.remove('Revenue')
        
        logger.info(f"Identified {len(self.numeric_columns)} numeric, "
                   f"{len(self.categorical_columns)} categorical, "
                   f"{len(self.boolean_columns)} boolean columns")
        
        # Fit numeric imputer
        if self.numeric_columns:
            self.numeric_imputer = SimpleImputer(strategy=self.numeric_strategy)
            self.numeric_imputer.fit(data[self.numeric_columns])
        
        # Fit categorical imputer
        if self.categorical_columns:
            self.categorical_imputer = SimpleImputer(strategy=self.categorical_strategy)
            self.categorical_imputer.fit(data[self.categorical_columns])
        
        self.is_fitted = True
        logger.info("DataCleaner fitted successfully")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by handling missing values.
        
        Args:
            data: DataFrame to transform
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if not self.is_fitted:
            raise ValueError("DataCleaner must be fitted before transform")
        
        logger.info("Transforming data with DataCleaner...")
        data_cleaned = data.copy()
        
        # Handle numeric columns
        if self.numeric_columns and self.numeric_imputer:
            numeric_data = self.numeric_imputer.transform(data_cleaned[self.numeric_columns])
            data_cleaned[self.numeric_columns] = numeric_data
        
        # Handle categorical columns
        if self.categorical_columns and self.categorical_imputer:
            categorical_data = self.categorical_imputer.transform(data_cleaned[self.categorical_columns])
            data_cleaned[self.categorical_columns] = categorical_data
        
        # Handle boolean columns (fill with mode)
        for col in self.boolean_columns:
            if data_cleaned[col].isnull().any():
                mode_value = data_cleaned[col].mode().iloc[0] if not data_cleaned[col].mode().empty else False
                data_cleaned[col].fillna(mode_value, inplace=True)
        
        # Handle any remaining missing values in other columns
        for col in data_cleaned.columns:
            if data_cleaned[col].isnull().any():
                if pd.api.types.is_numeric_dtype(data_cleaned[col]):
                    data_cleaned[col].fillna(data_cleaned[col].median(), inplace=True)
                else:
                    data_cleaned[col].fillna(data_cleaned[col].mode().iloc[0] if not data_cleaned[col].mode().empty else 'Unknown', inplace=True)
        
        logger.info("Data cleaning completed")
        return data_cleaned


class FeatureEncoder(BaseFeatureEncoder):
    """
    FeatureEncoder for categorical variable encoding (label encoding, one-hot encoding).
    """
    
    def __init__(self, 
                 encoding_strategy: str = 'label',
                 handle_unknown: str = 'ignore',
                 **kwargs):
        """
        Initialize FeatureEncoder.
        
        Args:
            encoding_strategy: 'label' for label encoding, 'onehot' for one-hot encoding
            handle_unknown: How to handle unknown categories ('ignore', 'error')
        """
        super().__init__(**kwargs)
        self.encoding_strategy = encoding_strategy
        self.handle_unknown = handle_unknown
        self.encoders = {}
        self.categorical_columns = []
        self.encoded_columns = []
        
    def fit(self, data: pd.DataFrame) -> 'FeatureEncoder':
        """
        Fit the feature encoder to the training data.
        
        Args:
            data: Training DataFrame
            
        Returns:
            self: Fitted FeatureEncoder instance
        """
        logger.info(f"Fitting FeatureEncoder with {self.encoding_strategy} strategy...")
        
        # Identify categorical columns (excluding target)
        self.categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'Revenue' in self.categorical_columns:
            self.categorical_columns.remove('Revenue')
        
        logger.info(f"Found categorical columns: {self.categorical_columns}")
        
        # Fit encoders for each categorical column
        for col in self.categorical_columns:
            if self.encoding_strategy == 'label':
                encoder = LabelEncoder()
                encoder.fit(data[col].astype(str))
                self.encoders[col] = encoder
            elif self.encoding_strategy == 'onehot':
                encoder = OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False)
                encoder.fit(data[[col]])
                self.encoders[col] = encoder
                # Store column names for one-hot encoding
                feature_names = encoder.get_feature_names_out([col])
                self.encoded_columns.extend(feature_names)
        
        self.is_fitted = True
        logger.info("FeatureEncoder fitted successfully")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features using fitted encoders.
        
        Args:
            data: DataFrame to transform
            
        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        if not self.is_fitted:
            raise ValueError("FeatureEncoder must be fitted before transform")
        
        logger.info("Transforming categorical features...")
        data_encoded = data.copy()
        
        for col in self.categorical_columns:
            if col in data_encoded.columns:
                if self.encoding_strategy == 'label':
                    # Handle unknown categories for label encoding
                    try:
                        data_encoded[col] = self.encoders[col].transform(data_encoded[col].astype(str))
                    except ValueError as e:
                        if self.handle_unknown == 'ignore':
                            # Replace unknown categories with most frequent class
                            known_classes = set(self.encoders[col].classes_)
                            data_encoded[col] = data_encoded[col].astype(str).apply(
                                lambda x: x if x in known_classes else self.encoders[col].classes_[0]
                            )
                            data_encoded[col] = self.encoders[col].transform(data_encoded[col])
                        else:
                            raise e
                            
                elif self.encoding_strategy == 'onehot':
                    # One-hot encoding
                    encoded_features = self.encoders[col].transform(data_encoded[[col]])
                    feature_names = self.encoders[col].get_feature_names_out([col])
                    
                    # Add encoded features to dataframe
                    for i, feature_name in enumerate(feature_names):
                        data_encoded[feature_name] = encoded_features[:, i]
                    
                    # Drop original column
                    data_encoded.drop(columns=[col], inplace=True)
        
        logger.info("Feature encoding completed")
        return data_encoded
    
    def encode_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features (alias for transform).
        
        Args:
            data: DataFrame to encode
            
        Returns:
            pd.DataFrame: DataFrame with encoded features
        """
        return self.transform(data)


class FeatureScaler(BaseFeatureScaler):
    """
    FeatureScaler for numeric feature normalization (StandardScaler, MinMaxScaler).
    """
    
    def __init__(self, 
                 scaling_method: str = 'standard',
                 **kwargs):
        """
        Initialize FeatureScaler.
        
        Args:
            scaling_method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
        """
        super().__init__(**kwargs)
        self.scaling_method = scaling_method
        self.scaler = None
        self.numeric_columns = []
        
    def fit(self, data: pd.DataFrame) -> 'FeatureScaler':
        """
        Fit the feature scaler to the training data.
        
        Args:
            data: Training DataFrame
            
        Returns:
            self: Fitted FeatureScaler instance
        """
        logger.info(f"Fitting FeatureScaler with {self.scaling_method} method...")
        
        # Identify numeric columns (excluding target)
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'Revenue' in self.numeric_columns:
            self.numeric_columns.remove('Revenue')
        
        # Remove boolean columns that might be treated as numeric
        boolean_cols = data.select_dtypes(include=['bool']).columns.tolist()
        self.numeric_columns = [col for col in self.numeric_columns if col not in boolean_cols]
        
        logger.info(f"Found numeric columns for scaling: {self.numeric_columns}")
        
        # Initialize and fit scaler
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}. Supported methods: 'standard', 'minmax'")
        
        if self.numeric_columns:
            self.scaler.fit(data[self.numeric_columns])
        
        self.is_fitted = True
        logger.info("FeatureScaler fitted successfully")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numeric features using fitted scaler.
        
        Args:
            data: DataFrame to transform
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        if not self.is_fitted:
            raise ValueError("FeatureScaler must be fitted before transform")
        
        logger.info("Scaling numeric features...")
        data_scaled = data.copy()
        
        if self.numeric_columns and self.scaler:
            scaled_features = self.scaler.transform(data_scaled[self.numeric_columns])
            data_scaled[self.numeric_columns] = scaled_features
        
        logger.info("Feature scaling completed")
        return data_scaled
    
    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numeric features (alias for transform).
        
        Args:
            data: DataFrame to scale
            
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        return self.transform(data)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline that combines cleaning, encoding, and scaling.
    """
    
    def __init__(self,
                 numeric_imputation: str = 'median',
                 categorical_imputation: str = 'most_frequent',
                 encoding_strategy: str = 'label',
                 scaling_method: str = 'standard',
                 handle_unknown: str = 'ignore'):
        """
        Initialize preprocessing pipeline.
        
        Args:
            numeric_imputation: Strategy for numeric missing values
            categorical_imputation: Strategy for categorical missing values
            encoding_strategy: Categorical encoding method
            scaling_method: Numeric scaling method
            handle_unknown: How to handle unknown categories
        """
        self.cleaner = DataCleaner(
            numeric_strategy=numeric_imputation,
            categorical_strategy=categorical_imputation
        )
        self.encoder = FeatureEncoder(
            encoding_strategy=encoding_strategy,
            handle_unknown=handle_unknown
        )
        self.scaler = FeatureScaler(scaling_method=scaling_method)
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame) -> 'PreprocessingPipeline':
        """
        Fit the complete preprocessing pipeline.
        
        Args:
            data: Training DataFrame
            
        Returns:
            self: Fitted pipeline instance
        """
        logger.info("Fitting complete preprocessing pipeline...")
        
        # Step 1: Clean data
        self.cleaner.fit(data)
        cleaned_data = self.cleaner.transform(data)
        
        # Step 2: Encode categorical features
        self.encoder.fit(cleaned_data)
        encoded_data = self.encoder.transform(cleaned_data)
        
        # Step 3: Scale numeric features
        self.scaler.fit(encoded_data)
        
        self.is_fitted = True
        logger.info("Preprocessing pipeline fitted successfully")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data through the complete preprocessing pipeline.
        
        Args:
            data: DataFrame to transform
            
        Returns:
            pd.DataFrame: Fully preprocessed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        logger.info("Transforming data through preprocessing pipeline...")
        
        # Apply transformations in sequence
        processed_data = self.cleaner.transform(data)
        processed_data = self.encoder.transform(processed_data)
        processed_data = self.scaler.transform(processed_data)
        
        logger.info("Preprocessing pipeline transformation completed")
        return processed_data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data in one step.
        
        Args:
            data: DataFrame to fit and transform
            
        Returns:
            pd.DataFrame: Fully preprocessed DataFrame
        """
        return self.fit(data).transform(data)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features after preprocessing.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted to get feature names")
        
        # This is a simplified version - in practice, you'd track feature names through transformations
        feature_names = []
        
        # Add numeric features
        feature_names.extend(self.scaler.numeric_columns)
        
        # Add encoded categorical features
        if self.encoder.encoding_strategy == 'onehot':
            feature_names.extend(self.encoder.encoded_columns)
        else:
            feature_names.extend(self.encoder.categorical_columns)
        
        # Add boolean features
        feature_names.extend(self.cleaner.boolean_columns)
        
        return feature_names