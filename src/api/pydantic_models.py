from pydantic import BaseModel, Field
from typing import Dict, Optional

class CreditRequest(BaseModel):
    """
    Schema for incoming credit risk prediction requests.
    Adjust fields to match your model features.
    """
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
    fraudresult: Optional[int] = Field(None)
    pricingstrategy: Optional[str] = Field(None)
    productcategory_data_bundles: int
    productcategory_financial_services: int
    productcategory_movies: int
    productcategory_other: int
    productcategory_ticket: int
    productcategory_transport: int
    productcategory_tv: int
    productcategory_utility_bill: int

class CreditResponse(BaseModel):
    """
    Schema for model prediction response.
    """
    model_used: str
    probability_of_default: float
    prediction: str
    details: Dict[str, float]