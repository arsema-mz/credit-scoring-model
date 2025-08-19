import pandas as pd
import joblib
import os

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def run_model(model, features):
    # Log the features being predicted
    print("Predicting with features:", features)

    preds = model.predict(features)  # Ensure features is a DataFrame or 2D array
    proba = model.predict_proba(features)[:, 1]

    print("Predictions:", preds)
    print("Probabilities:", proba)

    return preds, proba

def run_predictions(model_choice="both", features=None):
    # === Model Paths ===
    model_paths = {
        "logreg": "models/logreg_best.pkl",
        "random_forest": "models/random_forest_best.pkl"
    }

    models = {}

    # Load the models
    if model_choice in ["logreg", "both"]:
        models["logreg"] = load_model(model_paths["logreg"])

    if model_choice in ["random_forest", "both"]:
        models["random_forest"] = load_model(model_paths["random_forest"])

    # Run Predictions
    results = {}

    if "logreg" in models:
        preds, proba = run_model(models["logreg"], features)
        results["logreg_prediction"] = preds[0]  # Assuming single input
        results["logreg_risk_probability"] = proba[0]

    if "random_forest" in models:
        preds, proba = run_model(models["random_forest"], features)
        results["rf_prediction"] = preds[0]  # Assuming single input
        results["rf_risk_probability"] = proba[0]

    return results