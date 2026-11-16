import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="EV Population Predictor", page_icon="⚡", layout="centered")

st.title("⚡ Electric Vehicle Population Predictor")
st.write("This web application uses a trained **Random Forest Machine Learning Model** to predict whether an Electric Vehicle is a **Battery Electric Vehicle (BEV)** or a **Plug-in Hybrid Electric Vehicle (PHEV)**.")

@st.cache_resource
def load_model():
    return joblib.load('ev_model_pipeline.pkl')

model = load_model()

st.sidebar.header("Vehicle Specifications Input")

model_year = st.sidebar.slider("Model Year", 2010, 2026, 2023)
electric_range = st.sidebar.slider("Electric Range (miles)", 0, 350, 75)
base_msrp = st.sidebar.number_input("Base MSRP ($)", 0, 100000, 0)
postal_code = st.sidebar.number_input("Postal Code", 90000, 99999, 98101)
make = st.sidebar.selectbox("Vehicle Make", ["TESLA", "NISSAN", "CHEVROLET", "FORD", "BMW", "AUDI", "OTHER"])
cafv_eligibility = st.sidebar.selectbox(
    "CAFV Eligibility", 
    ["Clean Alternative Fuel Vehicle Eligible", "Eligibility unknown as battery range has not been researched"]
)

input_data = pd.DataFrame({
    'Model Year': [model_year],
    'Electric Range': [electric_range],
    'Base MSRP': [base_msrp],
    'Postal Code': [postal_code],
    'County': ['King'],
    'City': ['Seattle'],
    'State': ['WA'],
    'Make': [make],
    'Model': ['OTHER'],
    'Clean Alternative Fuel Vehicle (CAFV) Eligibility': [cafv_eligibility],
    'Legislative District': [36],
    'Electric Utility': ['CITY OF SEATTLE - (WA)|CITY OF TACOMA - (WA)'],
    '2020 Census Tract': [53033008400]
})

st.subheader("User Input Parameters")
st.write(input_data[['Model Year', 'Electric Range', 'Base MSRP', 'Make', 'Clean Alternative Fuel Vehicle (CAFV) Eligibility']])

if st.button("Predict Electric Vehicle Type"):
    prediction = model.predict(input_data)
    st.subheader("Prediction Result")
    if prediction[0] == "Battery Electric Vehicle (BEV)":
        st.success(f"The predicted vehicle type is: **{prediction[0]}** 🔋 (100% Electric)")
    else:
        st.warning(f"The predicted vehicle type is: **{prediction[0]}** ⛽🔌 (Hybrid)")
