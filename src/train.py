import pickle
import json
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def train_model():
    print("Training model...")
    
    X, y = make_classification(
        n_samples=5000,
        n_features=10,
        n_informative=7,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = \
        train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    os.makedirs('models', exist_ok=True)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "model_type": "RandomForest",
        "n_estimators": 100,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }
    
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Model and metrics saved!")
    return accuracy, f1

if __name__ == "__main__":
    accuracy, f1 = train_model()