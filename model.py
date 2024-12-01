# src/model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class IrisModel:
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)
        self.model = RandomForestClassifier()

    def train(self):
        X = self.data.drop('species', axis=1)
        y = self.data['species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        return accuracy_score(y_test, predictions)

if __name__ == "__main__":
    iris_model = IrisModel('datasets/iris.csv')
    accuracy = iris_model.train()
    print(f"Model Accuracy: {accuracy:.2f}")