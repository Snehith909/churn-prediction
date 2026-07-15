"""Load and validate raw data before it enters the pipeline."""
import pandas as pd
import logging

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw churn CSV and run basic schema/quality checks."""
    path = path or config.RAW_DATA_PATH
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path}")

    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Fail fast if the incoming data doesn't match what the pipeline expects."""
    expected_cols = set(
        config.DROP_COLS + config.RAW_CATEGORICAL_COLS + config.RAW_NUMERIC_COLS + [config.TARGET_COL]
    )
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Nulls found:\n{null_counts[null_counts > 0]}")

    if not df[config.TARGET_COL].isin([0, 1]).all():
        raise ValueError("Target column contains values outside {0, 1}")

    logger.info("Schema validation passed")


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
