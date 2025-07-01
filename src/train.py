import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('data/processed/processed.csv')

if 'customerid' in df.columns:
    df = df.drop(columns=['customerid'])
drop_cols = [
    'is_high_risk',
    'transactionid',
    'batchid',
    'accountid',
    'subscriptionid',
    'customerid',
    'transactionstarttime'
]
drop_cols = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=drop_cols)
y = df['is_high_risk']

bool_cols = X.select_dtypes(include='bool').columns
X[bool_cols] = X[bool_cols].astype(int)

obj_cols = X.select_dtypes(include='object').columns
if len(obj_cols) > 0:
    print("Dropping object/string columns:", list(obj_cols))
    X = X.drop(columns=obj_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logreg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(random_state=42)

logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)

print("[✓] Logistic Regression and Random Forest trained successfully.")


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# === Tune Logistic Regression ===
logreg_params = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

logreg = LogisticRegression(max_iter=1000)
logreg_grid = GridSearchCV(logreg, logreg_params, cv=5, scoring='f1', n_jobs=-1)
logreg_grid.fit(X_train, y_train)

print("[✓] Best Logistic Regression Params:", logreg_grid.best_params_)

# === Tune Random Forest ===
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print("[✓] Best Random Forest Params:", rf_grid.best_params_)


import joblib
import os

os.makedirs("models", exist_ok=True)

# Save best tuned models
joblib.dump(logreg_grid.best_estimator_, "models/logreg_best.pkl")
joblib.dump(rf_grid.best_estimator_, "models/random_forest_best.pkl")

print("[✓] Models saved to models/logreg_best.pkl and models/random_forest_best.pkl")
