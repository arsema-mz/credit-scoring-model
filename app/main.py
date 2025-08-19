from fastapi import FastAPI
from src.api.pydantic_models import BaseModel
from src.predict import load_model, run_predictions
import numpy as np
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import Optional

# Initialize FastAPI app
app = FastAPI(title="Credit Scoring API")

# Load trained model at startup
model = load_model("models/logreg_best.pkl")

@app.get("/")
def home():
    return {"message": "Credit Scoring API is running!"}

# Define the request model
class CreditRequest(BaseModel):
    countrycode: str
    providerid: str
    productid: str
    channelid: str
    amount: float
    value: float
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int
    total_amount: float
    average_amount: float
    transaction_count: int
    amount_std: float
    productcategory_data_bundles: int
    productcategory_financial_services: int
    productcategory_movies: int
    productcategory_other: int
    productcategory_ticket: int
    productcategory_transport: int
    productcategory_tv: int
    productcategory_utility_bill: int
    fraudresult: Optional[int] = None  # Add this field
    pricingstrategy: Optional[str] = None  # Add this field

# Define the response model
class CreditResponse(BaseModel):
    probability_of_default: float
    prediction: str

# Load your model (update with your model's path)
model = joblib.load("models/logreg_best.pkl")



@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    # Prepare features for prediction
    features = pd.DataFrame([{
        "countrycode": request.countrycode,
        "providerid": request.providerid,
        "productid": request.productid,
        "channelid": request.channelid,
        "amount": request.amount,
        "value": request.value,
        "transaction_hour": request.transaction_hour,
        "transaction_day": request.transaction_day,
        "transaction_month": request.transaction_month,
        "transaction_year": request.transaction_year,
        "total_amount": request.total_amount,
        "average_amount": request.average_amount,
        "transaction_count": request.transaction_count,
        "amount_std": request.amount_std,
        "fraudresult": request.fraudresult,  # Include this field
        "pricingstrategy": request.pricingstrategy,  # Include this field
        "productcategory_data_bundles": request.productcategory_data_bundles,
        "productcategory_financial_services": request.productcategory_financial_services,
        "productcategory_movies": request.productcategory_movies,
        "productcategory_other": request.productcategory_other,
        "productcategory_ticket": request.productcategory_ticket,
        "productcategory_transport": request.productcategory_transport,
        "productcategory_tv": request.productcategory_tv,
        "productcategory_utility_bill": request.productcategory_utility_bill,
    }])

    # Ensure the order of features matches the training data
    feature_order = [
        "countrycode",
        "providerid",
        "productid",
        "channelid",
        "amount",
        "value",
        "pricingstrategy",
        "fraudresult",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
        "total_amount",
        "average_amount",
        "transaction_count",
        "amount_std",
        "productcategory_data_bundles",
        "productcategory_financial_services",
        "productcategory_movies",
        "productcategory_other",
        "productcategory_ticket",
        "productcategory_transport",
        "productcategory_tv",
        "productcategory_utility_bill",
    ]

    # Reorder the DataFrame columns to match the training data
    features = features[feature_order]

    # Make prediction
    results = run_predictions(model_choice="both", features=features)

    return {
        "model_used": "logreg",  # Example: specify which model was used
        "probability_of_default": results.get("logreg_risk_probability", 0),
        "prediction": "default" if results.get("logreg_prediction", 0) == 1 else "no default",
        "details": results  # Include more details if needed
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)