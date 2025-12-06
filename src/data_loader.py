import pandas as pd

def load_data():
    # Mock data loading, replace with actual data loading
    data = pd.read_csv('data.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y