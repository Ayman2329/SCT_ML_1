import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        # Resolve the full path to handle relative paths
        full_path = Path(file_path).resolve()
        
        # Check if file exists
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
            
        # Load the dataset
        df = pd.read_csv(full_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def explore_data(df: pd.DataFrame) -> None:
    """
    Perform basic data exploration.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n=== Dataset Information ===")
    print(f"Shape: {df.shape}")
    
    print("\n=== First 5 Rows ===")
    display(df.head())
    
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    print("\n=== Basic Statistics ===")
    print(df.describe())

if __name__ == "__main__":
    # Example usage
    data_path = "../data/house_prices.csv"
    df = load_data(data_path)
    explore_data(df)
