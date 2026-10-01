from train_model import train
from predict import predict
#
from sklearn.datasets import load_iris
import pandas as pd
import os

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = [iris.target_names[t] for t in iris.target]
os.makedirs("data", exist_ok=True)
df.to_csv("data/iris.csv", index=False)   

if __name__ == "__main__":
    print("=== Training ===")
    train()
    print("\n=== Prediction ===")
    predict([5.1, 3.5, 1.4, 0.2])   