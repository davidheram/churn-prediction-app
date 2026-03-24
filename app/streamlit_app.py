import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

import streamlit as st
import pandas as pd
import joblib
from src.utils.explain import generate_explanation


model = joblib.load("models/churn_model.pkl")

st.title("Chun Prediction App")

st.write("Ingresa los datos del cliente")

#inputs
tenure = st.slider("Tenure (meses)", 0,72,12)
monthly_charges = st.number_input("Monthly Charges", 0.0,200.0,50.0)
total_charges = st.number_input("Total Charges", 0.0,10000.0, 500.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", 
     "Mailed check",
     "Bank transfer (automatic)",
     "Credit card (automatic)"],)

#cargar dataset limpio 
base_df = pd.read_csv("data/processed/churn_clean.csv")

#tomar un cliente real como base
input_data = base_df[base_df["Churn"]==1].drop("Churn", axis=1).iloc[[0]].copy()

input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges
input_data["TotalCharges"] = total_charges
input_data["Contract"] = contract
input_data["InternetService"] = internet_service
input_data["PaymentMethod"] = payment_method

#prediction

if st.button("Predecir Churn"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.write(f"Probabilidad de churn: {proba:.2f}")

    if prediction ==1: 
            st.error(f"el cliente probablemente se irá")

    else:
          st.success("el cliente probablemente se quedará")

    with st.spinner("Generando explicacion..."):
          explanation = generate_explanation(input_data, proba)

    st.subheader("Explicación")
    st.write(explanation )                    

