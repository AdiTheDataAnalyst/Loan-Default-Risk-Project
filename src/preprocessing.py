"""Load data, create features, split it, and build the preprocessing step."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLUMNS, ENGINEERED_COLUMNS, ID_COLUMNS, NUMERIC_COLUMNS,
    RANDOM_STATE, RAW_DIR, TARGET,
)

FEATURE_COLUMNS = NUMERIC_COLUMNS + ENGINEERED_COLUMNS + CATEGORICAL_COLUMNS


def load_raw(path=None) -> pd.DataFrame:
    """Read the original CSV. We never modify the raw file."""
    path = path or next(RAW_DIR.glob("*.csv"))
    return pd.read_csv(path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create two simple, explainable ratios. Each row uses only its own values,
    so nothing from other rows (or from the test set) can leak in."""
    df = df.copy()
    income = df["Income"].astype(float).replace(0, np.nan)   # zero income -> missing (imputed later), never divide by zero
    # How big is the loan compared with a year of income?
    df["LoanToIncome"] = df["LoanAmount"] / income
    # Rough monthly repayment (ignores interest) compared with monthly income.
    df["MonthlyPaymentToIncome"] = (df["LoanAmount"] / df["LoanTerm"]) / (income / 12)
    return df


def split_data(df: pd.DataFrame):
    """60% train / 20% validation / 20% test, keeping the default rate equal in each."""
    df = add_features(df).drop(columns=ID_COLUMNS, errors="ignore")
    X, y = df[FEATURE_COLUMNS], df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    """Numbers: fill gaps with the median, then standardise.
    Categories: fill gaps with the most common value, then one-hot encode.
    This object is placed INSIDE each model pipeline, so it is fitted on training data only."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_COLUMNS + ENGINEERED_COLUMNS),
        ("cat", categorical, CATEGORICAL_COLUMNS),
    ])
