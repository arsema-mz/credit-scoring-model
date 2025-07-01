from fastapi import FastAPI
from .pydantic_models import CustomerFeatures, PredictionResponse
import numpy as np
import pandas as pd
import os
import joblib


app = FastAPI()


MODEL_PATH = "models/random_forest_best.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model not found!")

model = joblib.load(MODEL_PATH)
@app.get("/")
def read_root():
    return {"message": "Credit Risk API is running!"}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(features: CustomerFeatures):
    data = pd.DataFrame([features.dict()])
    proba = model.predict_proba(data)[0][1]
    pred = int(proba >= 0.5)
    return PredictionResponse(risk_probability=proba, prediction=pred)

