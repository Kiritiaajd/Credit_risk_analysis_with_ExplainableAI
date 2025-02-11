# Streamlit or Flask dashboard application

import streamlit as st
import requests
import json

st.title("Credit Risk Analysis Dashboard")

# User input form
st.sidebar.header("Input Features")
person_age = st.sidebar.number_input("Person Age", min_value=18, max_value=100, value=30)
person_income = st.sidebar.number_input("Person Income", min_value=0, value=50000)
person_emp_length = st.sidebar.number_input("Employment Length", min_value=0, max_value=50, value=5)
loan_amnt = st.sidebar.number_input("Loan Amount", min_value=100, max_value=1000000, value=15000)
loan_int_rate = st.sidebar.slider("Loan Interest Rate", min_value=0.0, max_value=100.0, value=12.5)
loan_status = st.sidebar.selectbox("Loan Status", [0, 1])
loan_percent_income = st.sidebar.slider("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.3)
cb_person_default_on_file = st.sidebar.selectbox("Default on File", ["Y", "N"])
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length", min_value=0, max_value=50, value=10)

# One-hot encoded categorical inputs
home_ownership_options = ["OTHER", "OWN", "RENT"]
person_home_ownership = st.sidebar.radio("Home Ownership", home_ownership_options)
loan_intent_options = ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
loan_intent = st.sidebar.radio("Loan Intent", loan_intent_options)

# Convert categorical variables into one-hot encoding
home_ownership_dict = {f"person_home_ownership_{opt}": int(person_home_ownership == opt) for opt in home_ownership_options}
loan_intent_dict = {f"loan_intent_{opt}": int(loan_intent == opt) for opt in loan_intent_options}

# Create input dictionary
input_data = {
    "person_age": person_age,
    "person_income": person_income,
    "person_emp_length": person_emp_length,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_status": loan_status,
    "loan_percent_income": loan_percent_income,
    "cb_person_default_on_file": cb_person_default_on_file,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    **home_ownership_dict,
    **loan_intent_dict
}

# Make prediction request
if st.button("Predict Credit Risk"):
    response = requests.post("http://127.0.0.1:5000/predict", data=json.dumps(input_data), headers={"Content-Type": "application/json"})
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
        st.json(result["shap_values"])
    else:
        st.error("Error: " + response.json().get("error", "Unknown error"))