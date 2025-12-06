import pytest
from src.data_loader import load_data

def test_load_data():
    X, y = load_data()
    assert len(X) == len(y)