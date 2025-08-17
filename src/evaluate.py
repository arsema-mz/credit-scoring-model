import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import yaml
import os

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    print("[INFO] Loading config...")
    config = load_config()
    data_path = config["data"]["processed"]

    print("[INFO] Loading dataset...")
    df = pd.read_csv(data_path)

    X = df.drop("default", axis=1)
    y = df["default"]

    # same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": "models/logreg_best.pkl",
        "Random Forest": "models/random_forest_best.pkl",
    }

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"[WARN] {name} not found at {path}, skipping...")
            continue

        print(f"\n[INFO] Evaluating {name}...")
        model = joblib.load(path)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        print(classification_report(y_test, y_pred))

        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
            print(f"AUC: {auc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()
