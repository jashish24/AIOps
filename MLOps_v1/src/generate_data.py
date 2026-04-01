import pandas as pd
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
df = data.frame

df.to_csv("data/raw.csv", index=False)
print("Dataset generated")
