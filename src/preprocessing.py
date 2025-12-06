import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

class DataPreprocessor:
    """
    A class to handle preprocessing of the house price prediction dataset.
    """
    def __init__(self, target_col='price', test_size=0.2, random_state=42):
        """
        Initialize the preprocessor.
        
        Args:
            target_col (str): Name of the target column
            test_size (float): Proportion of the dataset to include in the test split
            random_state (int): Random seed for reproducibility
        """
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.numerical_cols = None
        self.categorical_cols = None
        self.preprocessor = None
        self.feature_names = None
        
    def _identify_columns(self, X):
        """Identify numerical and categorical columns."""
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column if present in numerical columns
        if self.target_col in self.numerical_cols:
            self.numerical_cols.remove(self.target_col)
            
        print(f"Numerical features: {self.numerical_cols}")
        print(f"Categorical features: {self.categorical_cols}")
    def _feature_engineering(self, X):
        X = X.copy()
        current_year = pd.Timestamp.now().year
        
        # Age features
        X['HouseAge'] = current_year - X['YearBuilt']
        if 'YearRemodAdd' in X.columns:
            X['RemodAge'] = current_year - X['YearRemodAdd']
        if 'GarageYrBlt' in X.columns:
            X['GarageAge'] = current_year - X['GarageYrBlt'].fillna(current_year)
        
        # Bathroom features
        bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        if all(col in X.columns for col in bath_cols):
            X['TotalBath'] = (X['FullBath'] + 0.5 * X['HalfBath'] + 
                            X['BsmtFullBath'] + 0.5 * X['BsmtHalfBath'])
        
        # Square footage
        sf_cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
        if all(col in X.columns for col in sf_cols):
            X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        
        # Quality features
        qual_cols = [col for col in X.columns if 'Qual' in col or 'Cond' in col]
        for col in qual_cols:
            if X[col].dtype == 'object':
                # Convert quality ratings to numerical (assuming they're like 'Ex', 'Gd', 'TA', etc.)
                quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
                X[col] = X[col].map(quality_map).fillna(0)
        
        return X

    def _create_preprocessor(self):
        """Create the preprocessing pipeline."""
        # Preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Bundle preprocessing for numerical and categorical data
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])
    
    def fit(self, X, y=None):
        """
        Fit the preprocessor on the training data.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable
            
        Returns:
            self: Fitted preprocessor
        """
        self._identify_columns(X)
        self._create_preprocessor()
        self.preprocessor.fit(X)
        
        # Get feature names after one-hot encoding
        if hasattr(self.preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
            cat_features = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_cols)
            self.feature_names = np.concatenate([self.numerical_cols, cat_features])
        else:
            self.feature_names = self.numerical_cols
            
        return self
    
    def transform(self, X):
        """
        Transform the input data using the fitted preprocessor.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Transformed features
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call 'fit' first.")
            
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X, y=None):
        """
        Fit the preprocessor and transform the data.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series, optional): Target variable
            
        Returns:
            np.ndarray: Transformed features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def save(self, filepath):
        """
        Save the preprocessor to disk.
        
        Args:
            filepath (str): Path to save the preprocessor
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a preprocessor from disk.
        
        Args:
            filepath (str): Path to the saved preprocessor
            
        Returns:
            DataPreprocessor: Loaded preprocessor
        """
        return joblib.load(filepath)


def handle_outliers(df, column, method='iqr', threshold=1.5):
    """
    Handle outliers in a numerical column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to handle outliers for
        method (str): Method to detect outliers ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection
        
    Returns:
        pd.DataFrame: Dataframe with outliers handled
    """
    df = df.copy()
    
    if method == 'iqr':
        # IQR method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Cap the outliers
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
    elif method == 'zscore':
        # Z-score method
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        df = df[(z_scores.abs() < threshold)]
    
    return df


def analyze_correlation(df, target_col, threshold=0.5):
    """
    Analyze correlation between features and target variable.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target column
        threshold (float): Correlation threshold to consider features as important
        
    Returns:
        pd.Series: Correlation values with target column
    """
    # Calculate correlation with target
    corr = df.corr()[target_col].sort_values(ascending=False)
    
    # Filter features with correlation above threshold
    important_features = corr[abs(corr) > threshold]
    
    print(f"\nFeatures with correlation > {threshold} with {target_col}:")
    print(important_features)
    
    return important_features