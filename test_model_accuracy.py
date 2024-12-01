# tests/test_model_accuracy.py
import pytest
from src.model import IrisModel

def test_model_accuracy():
    model = IrisModel('datasets/iris.csv')
    accuracy = model.train()
    assert accuracy > 0.90  # Check if the model accuracy is above 90%