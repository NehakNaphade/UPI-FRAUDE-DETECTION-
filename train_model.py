
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load dataset
df = pd.read_csv("upi_transactions.csv")

features = [
    "amount",
    "hour",
    "transactions_last_1h",
    "new_payee",
    "distance_km"
]

X = df[features]
y = df["is_fraud"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train the model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate the model on unseen test data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nModel Training Completed!")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

if len(set(y_test)) == 2:
    print("ROC-AUC:", round(
        roc_auc_score(y_test, probabilities), 4
    ))

# Save trained model and feature names
os.makedirs("models", exist_ok=True)

joblib.dump(
    {"model": model, "features": features},
    "models/upi_fraud_model.joblib"
)

print("\nModel saved at models/upi_fraud_model.joblib")
