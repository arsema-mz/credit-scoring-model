# app/main.py
from fastapi import FastAPI
from app.pydantic_models import CreditRequest, CreditResponse
import joblib
import pandas as pd

# Load best models
logreg_model = joblib.load("models/logreg_best.pkl")
rf_model = joblib.load("models/random_forest_best.pkl")

app = FastAPI(title="Credit Risk Probability API", version="1.0")


@app.get("/")
def root():
    return {"message": "Welcome to the Credit Risk Probability API 🚀"}


@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest, model: str = "logreg"):
    """
    Predict credit risk probability using specified model.
    Args:
        request: JSON payload with borrower features.
        model: "logreg" or "rf" (default = "logreg").
    """
    # Convert request to DataFrame for model
    X = pd.DataFrame([request.dict()])

    # Choose model
    if model == "rf":
        chosen_model = rf_model
        model_name = "Random Forest"
    else:
        chosen_model = logreg_model
        model_name = "Logistic Regression"

    # Predict probability
    prob_default = float(chosen_model.predict_proba(X)[0][1])
    prediction = "High Risk" if prob_default > 0.5 else "Low Risk"

    return CreditResponse(
        model_used=model_name,
        probability_of_default=prob_default,
        prediction=prediction,
        details={"threshold": 0.5}
    )
