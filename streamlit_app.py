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
    employment_length = st.number_input("Employment Length (years)", min_value=0, value=2)
    credit_history_length = st.number_input("Credit History Length (years)", min_value=0, value=5)
    outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0, value=5000)

    submitted = st.form_submit_button("Predict")

# --- Call API when form is submitted ---
if submitted:
    payload = {
        "age": age,
        "income": income,
        "employment_length": employment_length,
        "credit_history_length": credit_history_length,
        "outstanding_debt": outstanding_debt
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"📊 Default Probability: **{result['default_probability']:.2f}**")
        else:
            st.error("⚠️ Prediction failed. Check the API.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
