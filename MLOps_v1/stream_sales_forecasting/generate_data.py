import pandas as pd
import numpy as np

np.random.seed(42)

data = pd.DataFrame({
    "marketing_spend": np.random.randint(1000,5000,200),
    "season": np.random.randint(1,4,200),
    "region": np.random.randint(1,5,200),
    "sales": np.random.randint(20000,80000,200)
})

data.to_csv("sales_data.csv",index=False)

print("Dataset created")
