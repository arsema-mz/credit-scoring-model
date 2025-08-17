import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ======================
# 1. Load Processed Data
# ======================
print("[INFO] Loading dataset...")
df = pd.read_csv("data/processed/processed.csv")

# ======================
# 2. Drop Unnecessary Columns
# ======================
drop_cols = [
    "is_high_risk",
    "transactionid",
    "batchid",
    "accountid",
    "subscriptionid",
    "customerid",
    "transactionstarttime"
]
drop_cols = [col for col in drop_cols if col in df.columns]  # only drop if exists

X = df.drop(columns=drop_cols)
y = df["is_high_risk"]

# ======================
# 3. Handle Column Types
# ======================
# Convert boolean to int
bool_cols = X.select_dtypes(include="bool").columns
if len(bool_cols) > 0:
    X[bool_cols] = X[bool_cols].astype(int)
    print(f"[INFO] Converted boolean columns to int: {list(bool_cols)}")

# Drop object/string columns
obj_cols = X.select_dtypes(include="object").columns
if len(obj_cols) > 0:
    print(f"[INFO] Dropping object/string columns: {list(obj_cols)}")
    X = X.drop(columns=obj_cols)

# ======================
# 4. Train-Test Split
# ======================
print("[INFO] Splitting data into train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 5. Hyperparameter Tuning
# ======================
print("[INFO] Running GridSearchCV for Logistic Regression...")
logreg_params = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["liblinear"]
}
logreg = LogisticRegression(max_iter=1000)
logreg_grid = GridSearchCV(logreg, logreg_params, cv=5, scoring="f1", n_jobs=-1)
logreg_grid.fit(X_train, y_train)
print(f"[✓] Best Logistic Regression Params: {logreg_grid.best_params_}")

print("[INFO] Running GridSearchCV for Random Forest...")
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring="f1", n_jobs=-1)
rf_grid.fit(X_train, y_train)
print(f"[✓] Best Random Forest Params: {rf_grid.best_params_}")

# ======================
# 6. Save Best Models
# ======================
os.makedirs("models", exist_ok=True)
joblib.dump(logreg_grid.best_estimator_, "models/logreg_best.pkl")
joblib.dump(rf_grid.best_estimator_, "models/random_forest_best.pkl")

print("[✓] Models saved to:")
print("   → models/logreg_best.pkl")
print("   → models/random_forest_best.pkl")
