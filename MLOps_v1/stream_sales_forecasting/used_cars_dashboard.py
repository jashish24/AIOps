import streamlit as st
import requests

st.title("Used Car Price Predictor")

col1, col2 = st.columns(2)

with col1:
    brand = st.text_input("Brand", "Toyota")
    model = st.text_input("Model", "Camry")
    model_year = st.number_input("Model Year", 2000, 2025, 2020)
    milage = st.number_input("Mileage", 0, 500000, 35000)
    fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid", "Unknown"])

with col2:
    transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
    accident = st.selectbox("Accident History", ["None reported", "At least 1 accident or damage reported"])
    clean_title = st.selectbox("Clean Title", ["Yes", "No"])
    horsepower = st.number_input("Horsepower (HP)", 50, 1000, 203)
    engine_size = st.number_input("Engine Size (L)", 0.5, 10.0, 2.5)

if st.button("Predict Price"):
    params = {
        "brand": brand, "model": model, "model_year": model_year,
        "milage": milage, "fuel_type": fuel_type, "transmission": transmission,
        "accident": accident, "clean_title": clean_title,
        "horsepower": horsepower, "engine_size": engine_size
    }
    response = requests.get("http://127.0.0.1:8000/predict/used-cars", params=params)
    result = response.json()
    st.success(f"Estimated Price: ${result['predicted_price']:,.2f}")
