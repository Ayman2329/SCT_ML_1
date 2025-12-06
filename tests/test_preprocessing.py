import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from src.preprocessing import DataPreprocessor

def test_feature_engineering():
    """Test feature engineering functionality."""
    # Create test data with house-related features
    data = {
        'YearBuilt': [2000, 1990, 2010, 1980, 2005],
        'YearRemodAdd': [2010, 1995, 2010, 2000, 2010],
        'GarageYrBlt': [2005, 1990, 2010, 2000, 2005],
        'FullBath': [2, 1, 3, 2, 2],
        'HalfBath': [1, 0, 1, 0, 1],
        'BsmtFullBath': [1, 0, 1, 1, 0],
        'BsmtHalfBath': [0, 0, 1, 0, 0],
        'TotalBsmtSF': [1000, 800, 1200, 900, 1100],
        '1stFlrSF': [1200, 900, 1300, 1000, 1150],
        '2ndFlrSF': [800, 600, 900, 0, 850],
        'price': [250000, 180000, 320000, 150000, 280000]
    }
    df = pd.DataFrame(data)
    
    preprocessor = DataPreprocessor(target_col='price')
    processed_df = preprocessor._feature_engineering(df)
    
    # Check if new features are created
    current_year = pd.Timestamp.now().year
    assert 'HouseAge' in processed_df.columns
    assert 'RemodAge' in processed_df.columns
    assert 'GarageAge' in processed_df.columns
    assert 'TotalBath' in processed_df.columns
    assert 'TotalSF' in processed_df.columns
    
    # Check calculations
    assert processed_df['HouseAge'].equals(current_year - df['YearBuilt'])
    assert processed_df['RemodAge'].equals(current_year - df['YearRemodAdd'])
    assert processed_df['GarageAge'].equals(current_year - df['GarageYrBlt'])
    assert processed_df['TotalBath'].equals(
        df['FullBath'] + 0.5 * df['HalfBath'] + 
        df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    )
    assert processed_df['TotalSF'].equals(
        df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    )
    
    # Check if original columns are still present
    assert 'YearBuilt' in processed_df.columns
    assert 'YearRemodAdd' in processed_df.columns
    assert 'GarageYrBlt' in processed_df.columns


def test_create_preprocessor():
    """Test the creation of the preprocessing pipeline."""
    preprocessor = DataPreprocessor(target_col='price')
    preprocessor.numerical_cols = ['num1', 'num2']
    preprocessor.categorical_cols = ['cat1', 'cat2']
    
    # Create preprocessor
    preprocessor._create_preprocessor()
    
    # Check if preprocessor is created
    assert hasattr(preprocessor, 'preprocessor')
    assert preprocessor.preprocessor is not None
    assert isinstance(preprocessor.preprocessor, ColumnTransformer)
    
    # Check if transformers are set up correctly
    transformers = preprocessor.preprocessor.transformers
    assert len(transformers) == 2  # numerical and categorical
    assert transformers[0][0] == 'num'
    assert transformers[1][0] == 'cat'