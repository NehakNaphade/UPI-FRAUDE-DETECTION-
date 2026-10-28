
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 5000

amount = np.round(rng.lognormal(mean=6.0, sigma=1.2, size=n), 2)
amount = np.clip(amount, 1, 200000)

hour = rng.integers(0, 24, size=n)
transactions_last_1h = rng.poisson(1.5, size=n)
new_payee = rng.binomial(1, 0.25, size=n)
distance_km = np.round(rng.exponential(8, size=n), 2)

# Synthetic fraud labels for demonstration only
risk_score = (
    -4.0
    + 0.000012 * amount
    + 0.55 * (hour <= 5)
    + 0.75 * (transactions_last_1h >= 5)
    + 1.0 * new_payee
    + 0.035 * distance_km
)

probability = 1 / (1 + np.exp(-risk_score))
is_fraud = rng.binomial(1, probability)

df = pd.DataFrame({
    "amount": amount,
    "hour": hour,
    "transactions_last_1h": transactions_last_1h,
    "new_payee": new_payee,
    "distance_km": distance_km,
    "is_fraud": is_fraud
})

df.to_csv("upi_transactions.csv", index=False)

print("Dataset created successfully!")
print("Total transactions:", len(df))
print("\nFraud label counts:")
print(df["is_fraud"].value_counts())
print("\nFirst 5 rows:")
print(df.head())
