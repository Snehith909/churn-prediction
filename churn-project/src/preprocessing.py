"""Clean and encode raw data into a model-ready format."""
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src import config


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier columns that carry no predictive signal."""
    return df.drop(columns=config.DROP_COLS, errors="ignore")


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Returns a sklearn ColumnTransformer that scales numeric features
    and one-hot encodes categoricals. Fit this on TRAIN data only,
    then reuse (via .transform) on test/inference data to avoid leakage.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.NUMERIC_COLS),
            ("cat", categorical_transformer, config.CATEGORICAL_COLS),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extract readable feature names after fitting, useful for feature importance."""
    num_features = config.NUMERIC_COLS
    cat_features = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(config.CATEGORICAL_COLS)
    )
    return num_features + cat_features
