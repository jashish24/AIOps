import pandas as pd

df = pd.read_csv("data/raw.csv")

df = df.sample(frac=0.7, random_state=42)

df.to_csv("data/processed.csv", index=False)

print("Preprocessing completed")
