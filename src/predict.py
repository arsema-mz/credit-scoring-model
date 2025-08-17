import pandas as pd
import joblib
import os
import argparse

# === Argument Parser ===
parser = argparse.ArgumentParser(description="Run predictions using trained credit risk models.")
parser.add_argument("--model", type=str, choices=["logreg", "random_forest", "both"], default="both",
                    help="Which model to use for prediction (default: both).")
parser.add_argument("--input", type=str, default="data/new/new_data.csv",
                    help="Path to new input CSV file.")
parser.add_argument("--output", type=str, default="data/predictions/predicted_risks.csv",
                    help="Path to save prediction output.")
args = parser.parse_args()

# === Model Paths ===
model_paths = {
    "logreg": "models/logreg_best.pkl",
    "random_forest": "models/random_forest_best.pkl"
}

# === Load Data ===
if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input file not found: {args.input}")

new_data = pd.read_csv(args.input)

# Drop unnecessary IDs
drop_cols = [
    'customerid', 'transactionid', 'accountid',
    'batchid', 'subscriptionid', 'transactionstarttime'
]
new_data = new_data.drop(columns=[c for c in drop_cols if c in new_data.columns], errors='ignore')

# Convert booleans to integers
bool_cols = new_data.select_dtypes(include='bool').columns
new_data[bool_cols] = new_data[bool_cols].astype(int)

# === Run Predictions ===
results = new_data.copy()

def run_model(model_name, model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    preds = model.predict(new_data)
    proba = model.predict_proba(new_data)[:, 1]
    return preds, proba

if args.model in ["logreg", "both"]:
    preds, proba = run_model("logreg", model_paths["logreg"])
    results["logreg_prediction"] = preds
    results["logreg_risk_probability"] = proba

if args.model in ["random_forest", "both"]:
    preds, proba = run_model("random_forest", model_paths["random_forest"])
    results["rf_prediction"] = preds
    results["rf_risk_probability"] = proba

# === Save Predictions ===
os.makedirs(os.path.dirname(args.output), exist_ok=True)
results.to_csv(args.output, index=False)
print(f"[✓] Predictions saved to: {args.output}")





# python src/predict.py --model both
# python src/predict.py --model logreg
# python src/predict.py --model random_forest
