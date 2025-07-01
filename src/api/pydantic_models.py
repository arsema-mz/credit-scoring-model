from pydantic import BaseModel
from typing import List

class CustomerFeatures(BaseModel):
    countrycode: float
    providerid: float
    productid: float
    channelid: float
    amount: float
    value: float
    pricingstrategy: float
    fraudresult: float
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int
    total_amount: float
    average_amount: float
    transaction_count: float
    amount_std: float
    productcategory_data_bundles: int
    productcategory_financial_services: int
    productcategory_movies: int
    productcategory_other: int
    productcategory_ticket: int
    productcategory_transport: int
    productcategory_tv: int
    productcategory_utility_bill: int

class PredictionResponse(BaseModel):
    risk_probability: float
    prediction: int
