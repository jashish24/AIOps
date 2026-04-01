import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("iris_training")
df = pd.read_csv("data/processed.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")

    mlflow.sklearn.log_model(model, "model")

print("Training completed")
