"""Centralized configuration — paths, feature lists, hyperparameters."""
import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", ROOT_DIR / "data" / "raw" / "Churn_Modelling.csv"))
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "model.pkl"))
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Server
PORT = int(os.getenv("PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Columns
TARGET_COL = "Exited"
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]  # non-predictive identifiers

# Raw columns present in the source CSV (used for schema validation on ingestion)
RAW_CATEGORICAL_COLS = ["Geography", "Gender"]
RAW_NUMERIC_COLS = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

# Final columns fed into the sklearn preprocessing pipeline, AFTER feature_engineering.py
# adds derived columns (BalanceSalaryRatio, ProductsPerTenure, IsZeroBalance, AgeGroup)
CATEGORICAL_COLS = RAW_CATEGORICAL_COLS + ["AgeGroup"]
NUMERIC_COLS = RAW_NUMERIC_COLS + ["BalanceSalaryRatio", "ProductsPerTenure", "IsZeroBalance"]

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model hyperparameters (RandomForest baseline)
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "class_weight": "balanced",  # handle 20/80 imbalance
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
