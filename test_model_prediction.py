# tests/test_model_prediction.py
import pytest
from src.model import IrisModel

def test_model_prediction():
    model = IrisModel('datasets/iris.csv')
    model.train()
    
    # Check if the model can make predictions
    sample_data = [[5.1, 3.5, 1.4, 0.2]]  # Sample data point
    prediction = model.model.predict(sample_data)
    
    assert prediction is not None  # Ensure that a prediction is made
    assert len(prediction) == 1  # Ensure that only one prediction is returned