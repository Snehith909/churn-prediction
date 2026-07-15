"""Derive new features from raw columns to boost model signal."""
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-informed features for churn prediction.
    These are computed BEFORE the sklearn preprocessing pipeline runs.
    """
    df = df.copy()

    # Balance-to-salary ratio: captures financial exposure relative to income
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

    # Products per tenure year: engagement intensity
    df["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    # Zero balance flag: many churned users have $0 balance (dormant accounts)
    df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)

    # Age bucket: churn risk isn't linear with age
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0, 30, 40, 50, 60, 100],
        labels=["<30", "30-40", "40-50", "50-60", "60+"]
    ).astype(str)

    return df
