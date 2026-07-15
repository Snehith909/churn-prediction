import pandas as pd
import pytest

from src.feature_engineering import add_engineered_features
from src.preprocessing import drop_unused_columns, build_preprocessing_pipeline
from src import config


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "RowNumber": 1, "CustomerId": 100, "Surname": "Smith",
            "CreditScore": 650, "Geography": "France", "Gender": "Male",
            "Age": 35, "Tenure": 5, "Balance": 50000, "NumOfProducts": 2,
            "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 60000,
            "Exited": 0,
        },
        {
            "RowNumber": 2, "CustomerId": 101, "Surname": "Jones",
            "CreditScore": 500, "Geography": "Spain", "Gender": "Female",
            "Age": 55, "Tenure": 0, "Balance": 0, "NumOfProducts": 1,
            "HasCrCard": 0, "IsActiveMember": 0, "EstimatedSalary": 30000,
            "Exited": 1,
        },
    ])


def test_drop_unused_columns_removes_identifiers(sample_df):
    result = drop_unused_columns(sample_df)
    for col in config.DROP_COLS:
        assert col not in result.columns


def test_engineered_features_are_added(sample_df):
    result = add_engineered_features(sample_df)
    assert "BalanceSalaryRatio" in result.columns
    assert "ProductsPerTenure" in result.columns
    assert "IsZeroBalance" in result.columns
    assert "AgeGroup" in result.columns


def test_zero_balance_flag_correct(sample_df):
    result = add_engineered_features(sample_df)
    # second row has Balance == 0
    assert result.loc[1, "IsZeroBalance"] == 1
    assert result.loc[0, "IsZeroBalance"] == 0


def test_no_nulls_after_feature_engineering(sample_df):
    result = add_engineered_features(sample_df)
    assert result.isnull().sum().sum() == 0


def test_preprocessing_pipeline_fits_and_transforms(sample_df):
    df = add_engineered_features(sample_df)
    df = drop_unused_columns(df)
    X = df.drop(columns=[config.TARGET_COL])

    preprocessor = build_preprocessing_pipeline()
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0
