# In src/model.py
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
import logging
from pathlib import Path
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A comprehensive data preprocessing pipeline for house price prediction.
    Handles feature engineering, missing values, categorical encoding, and scaling.
    
    Args:
        target_col (str): Name of the target column. Defaults to 'SalePrice'.
        numerical_imputer_strategy (str): Strategy for numerical imputation. 
            Options: 'mean', 'median', 'most_frequent'. Defaults to 'median'.
        categorical_imputer_strategy (str): Strategy for categorical imputation.
            Options: 'most_frequent', 'constant'. Defaults to 'constant'.
        scaler_type (str): Type of scaler to use. 
            Options: 'robust', 'standard'. Defaults to 'robust'.
        cardinality_threshold (int): Threshold for high cardinality features. 
            Features with more unique values than this will use target encoding.
            Defaults to 10.
    """
    
    def __init__(
        self,
        target_col: str = 'SalePrice',
        numerical_imputer_strategy: str = 'median',
        categorical_imputer_strategy: str = 'constant',
        scaler_type: str = 'robust',
        cardinality_threshold: int = 10
    ):
        self.target_col = target_col
        self.numerical_imputer_strategy = numerical_imputer_strategy
        self.categorical_imputer_strategy = categorical_imputer_strategy
        self.scaler_type = scaler_type
        self.cardinality_threshold = cardinality_threshold
        
        # Initialize attributes
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.high_cardinality_cols: List[str] = []
        self.low_cardinality_cols: List[str] = []
        
        # Model components
        self.scaler_ = None
        self.num_imputer_ = None
        self.target_means_: Dict[str, Dict] = {}
        self.overall_mean_: Optional[float] = None
        self.current_year = datetime.now().year
        
        # Validate scaler type
        if self.scaler_type not in ['robust', 'standard']:
            raise ValueError("scaler_type must be either 'robust' or 'standard'")
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input DataFrame structure and types."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame")
            
        if self.target_col in X.columns and not pd.api.types.is_numeric_dtype(X[self.target_col]):
            raise ValueError(f"Target column '{self.target_col}' must be numeric")
    
    def _identify_columns(self, X: pd.DataFrame) -> None:
        """Identify and categorize columns by their data types."""
        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column if present
        if self.target_col in self.num_cols:
            self.num_cols.remove(self.target_col)
        
        # Categorize by cardinality
        self.low_cardinality_cols = [
            col for col in self.cat_cols 
            if X[col].nunique() <= self.cardinality_threshold
        ]
        self.high_cardinality_cols = [
            col for col in self.cat_cols 
            if col not in self.low_cardinality_cols
        ]
        
        logger.info(
            f"Identified {len(self.num_cols)} numerical, "
            f"{len(self.low_cardinality_cols)} low-cardinality categorical, and "
            f"{len(self.high_cardinality_cols)} high-cardinality categorical columns"
        )
    
    def _calculate_target_means(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Calculate target means for high cardinality features."""
        self.overall_mean_ = y.mean()
        for col in self.high_cardinality_cols:
            self.target_means_[col] = y.groupby(X[col]).mean().to_dict()
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor on the training data.
        
        Args:
            X: Training features
            y: Training target values (required for target encoding)
            
        Returns:
            self: Fitted preprocessor
        """
        self._validate_input(X)
        X = X.copy()
        
        # Identify column types
        self._identify_columns(X)
        
        # Calculate target means if target is provided
        if y is not None and len(self.high_cardinality_cols) > 0:
            if not isinstance(y, pd.Series):
                y = pd.Series(y)
            self._calculate_target_means(X, y)
            
        # Fit numerical imputer
        if self.num_cols:
            self.num_imputer_ = SimpleImputer(strategy=self.numerical_imputer_strategy)
            self.num_imputer_.fit(X[self.num_cols])
            
            # Fit scaler
            X_num = self.num_imputer_.transform(X[self.num_cols])
            self.scaler_ = (
                RobustScaler() if self.scaler_type == 'robust' 
                else StandardScaler()
            )
            self.scaler_.fit(X_num)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using the fitted preprocessor.
        
        Args:
            X: Input features to transform
            
        Returns:
            Transformed features as a pandas DataFrame
        """
        self._validate_input(X)
        X = X.copy()
        
        # Apply feature engineering
        X = self._feature_engineering(X)
        
        # Handle missing values and scale numerical features
        if self.num_cols:
            X_num = self.num_imputer_.transform(X[self.num_cols])
            X_num = self.scaler_.transform(X_num)
            X[self.num_cols] = X_num
            
        # Encode categorical features
        X = self._encode_categoricals(X)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit the preprocessor and transform the data."""
        return self.fit(X, y).transform(X)
    
    def _feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to create new features."""
        X = X.copy()
        
        # Age features
        if 'YearBuilt' in X.columns:
            X['HouseAge'] = self.current_year - X['YearBuilt']
        if 'YearRemodAdd' in X.columns:
            X['RemodAge'] = self.current_year - X['YearRemodAdd']
        if 'GarageYrBlt' in X.columns:
            X['GarageAge'] = self.current_year - X['GarageYrBlt'].fillna(self.current_year)
        
        # Bathroom features
        bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        if all(col in X.columns for col in bath_cols):
            X['TotalBath'] = (
                X['FullBath'] + 
                0.5 * X['HalfBath'] + 
                X['BsmtFullBath'] + 
                0.5 * X['BsmtHalfBath']
            )
        
        # Square footage
        sf_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
        if all(col in X.columns for col in sf_cols):
            X['TotalSF'] = X[sf_cols].sum(axis=1)
            
        return X
    
    def _encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        X = X.copy()
        
        # One-hot encode low cardinality features
        for col in self.low_cardinality_cols:
            if col in X.columns:
                dummies = pd.get_dummies(
                    X[col].astype(str), 
                    prefix=col, 
                    drop_first=True
                )
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
        
        # Target encoding for high cardinality features
        if hasattr(self, 'target_means_') and self.target_means_:
            for col in self.high_cardinality_cols:
                if col in X.columns:
                    X[col] = X[col].map(self.target_means_.get(col, {}))
                    X[col].fillna(self.overall_mean_, inplace=True)
                    # Rename to indicate it's been encoded
                    X.rename(columns={col: f"{col}_encoded"}, inplace=True)
        
        return X
    
    def get_feature_names(self) -> List[str]:
        """Get the names of the transformed features."""
        features = []
        
        # Numerical features
        features.extend(self.num_cols)
        
        # Categorical features
        # Note: For one-hot encoded features, we don't know the exact names
        # as they're created during transform. This is a limitation.
        features.extend([f"{col}_encoded" for col in self.high_cardinality_cols])
        
        return features
    
    def save(self, filepath: str) -> None:
        """Save the preprocessor to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a dictionary of attributes to save
        state = {
            'target_col': self.target_col,
            'numerical_imputer_strategy': self.numerical_imputer_strategy,
            'categorical_imputer_strategy': self.categorical_imputer_strategy,
            'scaler_type': self.scaler_type,
            'cardinality_threshold': self.cardinality_threshold,
            'num_cols': self.num_cols,
            'cat_cols': self.cat_cols,
            'high_cardinality_cols': self.high_cardinality_cols,
            'low_cardinality_cols': self.low_cardinality_cols,
            'overall_mean': self.overall_mean_,
            'target_means': self.target_means_,
            'current_year': self.current_year,
        }
        
        # Save the state
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """Load a preprocessor from disk."""
        path = Path(filepath)
        
        with open(path, 'r') as f:
            state = json.load(f)
        
        # Create a new instance
        preprocessor = cls(
            target_col=state['target_col'],
            numerical_imputer_strategy=state['numerical_imputer_strategy'],
            categorical_imputer_strategy=state['categorical_imputer_strategy'],
            scaler_type=state['scaler_type'],
            cardinality_threshold=state['cardinality_threshold']
        )
        
        # Set attributes
        preprocessor.num_cols = state['num_cols']
        preprocessor.cat_cols = state['cat_cols']
        preprocessor.high_cardinality_cols = state['high_cardinality_cols']
        preprocessor.low_cardinality_cols = state['low_cardinality_cols']
        preprocessor.overall_mean_ = state['overall_mean']
        preprocessor.target_means_ = state['target_means']
        preprocessor.current_year = state['current_year']
        
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor