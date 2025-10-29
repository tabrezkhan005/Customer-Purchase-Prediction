"""
Base classes and interfaces for data processing components.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np


class BaseDataProcessor(ABC):
    """Abstract base class for data processing components."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseDataProcessor':
        """Fit the processor to the data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data in one step."""
        return self.fit(data).transform(data)


class BaseDataLoader(ABC):
    """Abstract base class for data loading components."""
    
    @abstractmethod
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from file."""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the loaded data."""
        pass


class BaseFeatureEncoder(BaseDataProcessor):
    """Abstract base class for feature encoding."""
    
    @abstractmethod
    def encode_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        pass


class BaseFeatureScaler(BaseDataProcessor):
    """Abstract base class for feature scaling."""
    
    @abstractmethod
    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features."""
        pass


class DataPipeline:
    """Pipeline for chaining data processing steps."""
    
    def __init__(self, steps: list):
        """
        Initialize pipeline with list of (name, processor) tuples.
        
        Args:
            steps: List of (name, processor) tuples where processor 
                   implements BaseDataProcessor interface
        """
        self.steps = steps
        self.processors = {}
        
        for name, processor in steps:
            if not isinstance(processor, BaseDataProcessor):
                raise ValueError(f"Processor {name} must inherit from BaseDataProcessor")
            self.processors[name] = processor
    
    def fit(self, data: pd.DataFrame) -> 'DataPipeline':
        """Fit all processors in the pipeline."""
        current_data = data.copy()
        
        for name, processor in self.steps:
            processor.fit(current_data)
            current_data = processor.transform(current_data)
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data through all processors."""
        current_data = data.copy()
        
        for name, processor in self.steps:
            if not processor.is_fitted:
                raise ValueError(f"Processor {name} must be fitted before transform")
            current_data = processor.transform(current_data)
        
        return current_data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data in one step."""
        return self.fit(data).transform(data)
    
    def get_processor(self, name: str) -> BaseDataProcessor:
        """Get a specific processor by name."""
        if name not in self.processors:
            raise ValueError(f"Processor {name} not found in pipeline")
        return self.processors[name]