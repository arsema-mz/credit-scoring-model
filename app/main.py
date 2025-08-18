from fastapi import FastAPI
from src.api.pydantic_models import CreditRequest, CreditResponse
from src.data_processing import process_input
from src.predict import load_model, make_prediction

# Initialize FastAPI app
app = FastAPI(title="Credit Scoring API")

# Load trained model at startup
model = load_model("models/final_model.pkl")

@app.get("/")
def home():
    return {"message": "Credit Scoring API is running!"}

@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    # Step 1: process input using your pipeline
    features = process_input(request.dict())

    # Step 2: make prediction
    probability, label = make_prediction(model, features)

    return CreditResponse(
        probability=probability,
        label=label
    )
