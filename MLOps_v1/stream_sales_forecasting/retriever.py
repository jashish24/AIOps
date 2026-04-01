from fastapi import FastAPI
import requests

app = FastAPI()

PREDICT_URL = "http://127.0.0.1:8000/predict"

@app.get("/retrieve")
def retrieve(marketing_spend: int, season: int, region: int):
    response = requests.get(PREDICT_URL, params={
        "marketing_spend": marketing_spend,
        "season": season,
        "region": region
    })
    return response.json()
