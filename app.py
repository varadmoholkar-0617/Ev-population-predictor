import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="EV Population Predictor", page_icon="⚡", layout="centered")

st.title("⚡ Electric Vehicle Population Predictor")
st.write("This web application uses a trained **Random Forest Machine Learning Model** to predict whether an Electric Vehicle is a **Battery Electric Vehicle (BEV)** or a **Plug-in Hybrid Electric Vehicle (PHEV)**.")

# Train model directly on load so there are zero pickle/version errors!
@st.cache_resource
def get_trained_model():
    # Load sample / online dataset or build a lightweight robust pipeline
    # For speed and stability on cloud, let's build the pipeline directly:
    from sklearn.datasets import make_classification
    
    # We load your dataset or fallback to a trained pipeline matching your schema
    url = "https://raw.githubusercontent.com/varadmoholkar-0617/Ev-population-predictor/main/Electric_Vehicle_Population_Data%20(1).csv"
    try:
        df = pd.read_csv('Electric_Vehicle_Population_Data (1).csv')
    except:
        df = pd.read_csv('https://raw.githubusercontent.com/varadmoholkar-0617/Ev-population-predictor/main/Electric_Vehicle_Population_Data%20(1).csv')
        
    df = df.dropna().drop_duplicates()
    target_column = 'Electric Vehicle Type'
    drop_cols = ['VIN (1-10)', 'DOL Vehicle ID', 'Vehicle Location']
    df_model = df.drop(columns=[col for col in drop_cols if col in df.columns])
    df_model = df_model.dropna(subset=[target_column])

    X = df_model.drop(columns=[target_column])
    y = df_model[target_column]

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1))
    ])
    
    # Train on a manageable subset for instant cloud boot
    X_train, _, y_train, _ = train_test_split(X.head(5000), y.head(5000), test_size=0.1, random_state=42)
    rf_pipeline.fit(X_train, y_train)
    return rf_pipeline

with st.spinner("Training model pipeline... Please wait a moment."):
    model = get_trained_model()

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
