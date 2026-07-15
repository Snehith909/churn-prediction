"""Train the churn model end-to-end and persist model + metadata."""
import json
import logging
from datetime import datetime, timezone

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from src import config
from src.data_ingestion import load_raw_data
from src.feature_engineering import add_engineered_features
from src.preprocessing import drop_unused_columns, build_preprocessing_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train():
    # 1. Load + validate raw data
    df = load_raw_data()

    # 2. Feature engineering (before split — these are row-wise, not leakage-prone)
    df = add_engineered_features(df)
    df = drop_unused_columns(df)

    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]

    # 3. Split BEFORE fitting any scaler/encoder to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # 4. Build full pipeline: preprocessing + model, fit on train only
    preprocessor = build_preprocessing_pipeline()
    model = RandomForestClassifier(**config.MODEL_PARAMS)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    logger.info(f"Test metrics: {metrics}")

    # 6. Persist model
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    logger.info(f"Model saved to {config.MODEL_PATH}")

    # 7. Persist metadata — this is what most student projects skip
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "RandomForestClassifier",
        "hyperparameters": config.MODEL_PARAMS,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": metrics,
        "feature_columns": list(X.columns),
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {config.METADATA_PATH}")

    return pipeline, metrics


if __name__ == "__main__":
    train()
