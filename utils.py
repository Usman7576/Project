# src/utils.py
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def load_data(file_path):
    """Load dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def evaluate_model(y_true, y_pred):
    """Evaluate model performance using classification report and confusion matrix."""
    report = classification_report(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return report, confusion

def preprocess_data(data):
    """Preprocess the data (if needed)."""
    # Example: Handle missing values, encode categorical variables, etc.
    # For the Iris dataset, we might not need much preprocessing
    return data.dropna()

def get_feature_target(data, target_column):
    """Separate features and target variable."""
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y