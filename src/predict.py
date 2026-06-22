import pickle
import numpy as np

def load_model(path='models/model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict(features, model=None):
    if model is None:
        model = load_model()
    
    X = np.array(features).reshape(1, -1)
    prediction = int(model.predict(X)[0])
    confidence = round(
        float(model.predict_proba(X)[0].max()), 3
    )
    
    tiers = {
        0: "Low Performer",
        1: "High Performer"
    }
    
    return {
        "prediction": prediction,
        "performance_tier": tiers[prediction],
        "confidence": confidence
    }

if __name__ == "__main__":
    result = predict(
        [1.2, 0.5, -0.3, 2.1, 0.8,
         -1.2, 0.4, 1.1, -0.7, 0.9]
    )
    print(f"Result: {result}")