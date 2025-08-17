from pydantic import BaseModel
from typing import Dict


class CreditRequest(BaseModel):
    """
    Schema for incoming credit risk prediction requests.
    Adjust fields to match your model features.
    """
    age: int
    income: float
    loan_amount: float
    credit_history: int
    employment_status: str
    transaction_count: int


class CreditResponse(BaseModel):
    """
    Schema for model prediction response.
    """
    model_used: str
    probability_of_default: float
    prediction: str
    details: Dict[str, float]
