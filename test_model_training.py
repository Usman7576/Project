# tests/test_model_training.py
import pytest
from src.model import IrisModel

def test_model_training():
    model = IrisModel('datasets/iris.csv')
    accuracy = model.train()
    assert accuracy is not None  # Ensure that the model returns an accuracy score