"""Reusable inference logic — used by the API and any batch scoring scripts."""
import joblib
import pandas as pd

from src import config
from src.feature_engineering import add_engineered_features
from src.preprocessing import drop_unused_columns

_model = None  # loaded once, reused across requests


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(config.MODEL_PATH)
    return _model


def predict_single(record: dict) -> dict:
    """
    Run inference on a single customer record (as a dict matching raw schema).
    Returns churn probability + predicted label.
    """
    model = load_model()
    df = pd.DataFrame([record])
    df = add_engineered_features(df)
    df = drop_unused_columns(df)

    proba = model.predict_proba(df)[0, 1]
    label = int(proba >= 0.5)

    return {
        "churn_probability": round(float(proba), 4),
        "will_churn": bool(label),
    }


if __name__ == "__main__":
    sample = {
        "RowNumber": 1, "CustomerId": 1, "Surname": "Test",
        "CreditScore": 600, "Geography": "France", "Gender": "Male",
        "Age": 40, "Tenure": 3, "Balance": 60000, "NumOfProducts": 2,
        "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 50000,
    }
    print(predict_single(sample))
