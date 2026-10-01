import pickle
import numpy as np
from config import MODEL_DIR

def predict(sample_features):
    """sample_features: list of 4 numbers [sepal_len, sepal_wid, petal_len, petal_wid]"""
    model_path = f"{MODEL_DIR}iris_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict([sample_features])
    probability = model.predict_proba([sample_features])
    print(f"Predicted: {prediction[0]}")
    print(f"Confidence: {probability[0].max():.2%}")

if __name__ == "__main__":
    # Example: setosa
    predict([5.1, 3.5, 1.4, 0.2])   