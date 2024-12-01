# tests/test_feature_target_separation.py
import pytest
from src.model import IrisModel

def test_feature_target_separation():
    model = IrisModel('datasets/iris.csv')
    X = model.data.drop('species', axis=1)
    y = model.data['species']
    
    assert X.shape[0] == y.shape[0]  # Ensure that the number of samples in features and target are the same
    assert 'species' not in X.columns  # Ensure that 'species' is not in features