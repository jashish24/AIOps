import pandas as pd
import joblib
import mlflow

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mlflow.set_experiment("mlflow_evaluate")
df = pd.read_csv("data/processed.csv")

X = df.drop("target", axis=1)
y = df["target"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load("models/model.pkl")

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Accuracy:", float(acc))

mlflow.log_metric("accuracy", float(acc))
