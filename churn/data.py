"""Loading, cleaning and splitting of the Telco Customer Churn dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_customer_churn.csv"

ID_COL = "customerID"
TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
RAW_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw customer table.

    * ``TotalCharges`` is read as text because 11 rows are blank. Those rows
      are customers with ``tenure == 0`` who have not been billed yet, so the
      correct value is ``MonthlyCharges * tenure`` (i.e. 0), not the median.
    * ``SeniorCitizen`` is stored as 0/1 while every other flag is Yes/No;
      it is converted so all categorical columns share one convention.
    """
    df = df.copy()
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = total.fillna(df["MonthlyCharges"] * df["tenure"]).astype(float)
    df["tenure"] = df["tenure"].astype(float)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["SeniorCitizen"] = (
        df["SeniorCitizen"].astype(str).map({"0": "No", "1": "Yes", "No": "No", "Yes": "Yes"})
    )
    return df


def load_dataset(path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the CSV and return ``(X, y)`` with ``y = 1`` for churners."""
    df = clean(pd.read_csv(path))
    y = (df[TARGET] == "Yes").astype(int)
    return df[RAW_FEATURES], y


def split(X: pd.DataFrame, y: pd.Series):
    """Stratified 80/20 split. The test set is used once, for the final report."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
