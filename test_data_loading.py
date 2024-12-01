# tests/test_data_loading.py
import pytest
import pandas as pd
from src.model import IrisModel

def test_data_loading():
    model = IrisModel('datasets/iris.csv')
    assert isinstance(model.data, pd.DataFrame)  # Ensure that data is loaded as a DataFrame
    assert not model.data.empty  # Ensure that the DataFrame is not empty