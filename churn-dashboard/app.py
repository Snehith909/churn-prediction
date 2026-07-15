"""Streamlit UI for the churn prediction API. Talks to the FastAPI backend
deployed as a separate Render service — set API_URL as an env var, don't hardcode it.
"""
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Churn Predictor", page_icon="📊")
st.title("📊 Customer Churn Predictor")
st.caption("Bank customer churn model — served via FastAPI on Render")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 650)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 35)
        tenure = st.slider("Tenure (years)", 0, 15, 3)

    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
        num_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Active Member?", ["Yes", "No"])
        salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0, step=1000.0)

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    payload = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()

        prob = result["churn_probability"]
        will_churn = result["will_churn"]

        st.divider()
        if will_churn:
            st.error(f"⚠️ High churn risk — probability: **{prob:.1%}**")
        else:
            st.success(f"✅ Low churn risk — probability: **{prob:.1%}**")

        st.progress(prob)

    except requests.exceptions.ConnectionError:
        st.error(f"Couldn't reach the API at `{API_URL}`. Is the backend service running?")
    except requests.exceptions.Timeout:
        st.error("Request timed out — the backend may be cold-starting (free tier). Try again in a moment.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.divider()
st.caption(f"API endpoint: `{API_URL}`")
