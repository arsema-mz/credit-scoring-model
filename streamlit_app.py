import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Credit Scoring Dashboard", layout="centered")

st.title("💳 Credit Risk Scoring Dashboard")
st.write("Enter applicant details and get default probability.")

# --- Input Form ---
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Monthly Income ($)", min_value=0, value=3000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=15000)
    credit_history = st.number_input("Credit History Length (years)", min_value=0, value=5)
    employment_status = st.selectbox("Employment Status", ("employed", "unemployed", "self-employed"))
    transaction_count = st.number_input("Transaction Count", min_value=0, value=10)

    submitted = st.form_submit_button("Predict")

# --- Call API when form is submitted ---
if submitted:
    payload = {
    "countrycode": "US",
    "providerid": "123",
    "productid": "456",
    "channelid": "789",
    "amount": 15000,
    "value": 3000,
    "transaction_hour": 12,
    "transaction_day": 15,
    "transaction_month": 8,
    "transaction_year": 2025,
    "total_amount": 3000,
    "average_amount": 500,
    "transaction_count": 10,
    "amount_std": 100,
    "fraudresult": 0,
    "pricingstrategy": "standard",
    "productcategory_data_bundles": 0,
    "productcategory_financial_services": 0,
    "productcategory_movies": 0,
    "productcategory_other": 0,
    "productcategory_ticket": 0,
    "productcategory_transport": 0,
    "productcategory_tv": 0,
    "productcategory_utility_bill": 0
}

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"📊 Default Probability: **{result['probability_of_default']:.2f}**")
            st.write(f"Prediction: **{result['prediction']}**")
        else:
            st.error("⚠️ Prediction failed. Check the API.")
    except Exception as e:
        st.error(f"❌ Error: {e}")