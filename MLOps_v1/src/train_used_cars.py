import numpy as np
import pandas as pd
import joblib
import mlflow
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# Load Data
# =========================
df = pd.read_csv("../data/used_cars.csv")

# =========================
# Cleaning
# =========================
df["price"] = df["price"].str.replace(r"[$,]", "", regex=True).astype(float)

df["milage"] = (
    df["milage"]
    .str.replace(" mi.", "", regex=False)
    .str.replace(",", "")
    .astype(float)
)

# Remove extreme outliers
df = df[df["price"] < 200000]

# =========================
# Missing Values
# =========================
df["fuel_type"] = df["fuel_type"].fillna("Unknown")
df["accident"] = df["accident"].fillna("None reported")
df["clean_title"] = df["clean_title"].fillna("No")

# =========================
# Engine Feature Extraction (IMPORTANT)
# =========================
df["horsepower"] = df["engine"].str.extract(r"(\d+\.?\d*)\s*HP", expand=False)
df["engine_size"] = df["engine"].str.extract(r"(\d+\.?\d*)L", expand=False)

df["horsepower"] = df["horsepower"].astype(float)
df["engine_size"] = df["engine_size"].astype(float)

# Drop raw engine column
df.drop(columns=["engine"], inplace=True)

# =========================
# Feature Engineering
# =========================
CURRENT_YEAR = 2026

df["car_age"] = CURRENT_YEAR - df["model_year"]
df["milage_log"] = np.log1p(df["milage"])
df["milage_per_year"] = df["milage"] / (df["car_age"] + 1)
df["brand_model"] = df["brand"] + "_" + df["model"]



# =========================
# Features
# =========================
cat_cols = [
    "brand", "model", "brand_model",
    "fuel_type", "transmission",
    "accident", "clean_title"
]

# Optionally drop weak noisy cols
df.drop(columns=["ext_col", "int_col"], inplace=True, errors="ignore")

X = df.drop("price", axis=1)
y = np.log1p(df["price"])

# =========================
# Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# MLflow
# =========================
mlflow.set_experiment("used_cars_price_prediction")

with mlflow.start_run():

    model = CatBoostRegressor(
        iterations=4233,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5,
        subsample=0.8,
        eval_metric="R2",
        random_seed=42,
        verbose=200
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    # =========================
    # Predictions
    # =========================
    preds = np.expm1(model.predict(X_test))
    actual = np.expm1(y_test)

    mae = mean_absolute_error(actual, preds)
    r2 = r2_score(actual, preds)

    # =========================
    # Logging
    # =========================
    mlflow.log_param("model", "CatBoost")
    mlflow.log_param("iterations", 5000)
    mlflow.log_param("learning_rate", 0.02)
    mlflow.log_param("depth", 8)
    mlflow.log_param("l2_leaf_reg", 5)
    mlflow.log_param("subsample", 0.8)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # =========================
    # Save Model
    # =========================
    joblib.dump(model, "../models/used_cars_catboost_v2.pkl")

    print(f"MAE: {mae:.2f}, R2: {r2:.4f}")
