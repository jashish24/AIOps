import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("sales_data.csv")

X = data[["marketing_spend","season","region"]]
y = data["sales"]

model = LinearRegression()

model.fit(X,y)

joblib.dump(model,"sales_model.pkl")

print("Model saved successfully")
