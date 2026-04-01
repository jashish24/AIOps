from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("sales_model.pkl")
used_cars_model = joblib.load("../models/used_cars_catboost_v2.pkl")

@app.get("/predict")

def predict(marketing_spend:int, season:int, region:int):

    df = pd.DataFrame([{
        "marketing_spend": marketing_spend,
        "season": season,
        "region": region
    }])

    pred = model.predict(df)

    return {"predicted_sales": float(pred[0])}


@app.get("/predict/used-cars")
def predict_used_cars(
    brand: str, model: str, model_year: int, milage: float,
    fuel_type: str, transmission: str, accident: str, clean_title: str,
    horsepower: float, engine_size: float
):
    CURRENT_YEAR = 2026
    car_age = CURRENT_YEAR - model_year
    milage_log = np.log1p(milage)
    milage_per_year = milage / (car_age + 1)
    brand_model = f"{brand}_{model}"

    df = pd.DataFrame([{
        "brand": brand, "model": model, "model_year": model_year,
        "milage": milage, "fuel_type": fuel_type, "transmission": transmission,
        "accident": accident, "clean_title": clean_title,
        "horsepower": horsepower, "engine_size": engine_size,
        "car_age": car_age, "milage_log": milage_log,
        "milage_per_year": milage_per_year, "brand_model": brand_model
    }])

    pred = used_cars_model.predict(df)
    return {"predicted_price": round(float(np.expm1(pred[0])), 2)}
