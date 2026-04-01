import streamlit as st
import requests

st.title("Sales Forecast AI Agent")

marketing = st.slider("Marketing Spend", 1000, 5000)
season = st.selectbox("Season", [1, 2, 3])
region = st.selectbox("Region", [1, 2, 3, 4])

url = f"http://127.0.0.1:8000/predict?marketing_spend={marketing}&season={season}&region={region}"
response = requests.get(url)
result = response.json()
st.success(f"Predicted Sales: {int(result['predicted_sales'])}")
