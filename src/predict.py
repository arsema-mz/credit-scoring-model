import pandas as pd
import joblib
import os

# === Load the trained model ===
model_path = "models/random_forest_best.pkl"  
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = joblib.load(model_path)

new_data = pd.read_csv("data/new/new_data.csv")  # example new file path

# Drop or preprocess any unnecessary columns
drop_cols = ['customerid', 'transactionid', 'accountid', 'batchid', 'subscriptionid', 'transactionstarttime']
new_data = new_data.drop(columns=[col for col in drop_cols if col in new_data.columns], errors='ignore')

# Convert bools to int (if needed)
bool_cols = new_data.select_dtypes(include='bool').columns
new_data[bool_cols] = new_data[bool_cols].astype(int)

# Make prediction
predictions = model.predict(new_data)
proba = model.predict_proba(new_data)[:, 1]  # probability of class 1

# Save or print
output = pd.DataFrame({
    'prediction': predictions,
    'risk_probability': proba
})

output.to_csv("data/processed/predicted_risks.csv", index=False)
print("[✓] Predictions saved to: data/predictions/predicted_risks.csv")
