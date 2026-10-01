import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils import load_data
from config import MODEL_DIR, TEST_SIZE, RANDOM_STATE

def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(columns=["species"]), y,
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

   
    model_path = f"{MODEL_DIR}iris_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

   
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Model saved → {model_path}")

if __name__ == "__main__":
    train()   