"""Inference helpers shared by the Streamlit app and the tests."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from churn.data import CATEGORICAL, RAW_FEATURES, ROOT, clean
from churn.features import INTERNET_ADDONS

MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"


def load_artifacts(model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
    """Return ``(pipeline, metadata)``. Run ``python -m churn.train`` first."""
    pipeline = joblib.load(model_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return pipeline, metadata


def make_customer(**values) -> pd.DataFrame:
    """Build a one-row frame in the raw dataset format from keyword arguments.

    Impossible combinations are corrected instead of being sent to the model:
    a customer without internet cannot have internet add-ons, and a customer
    without phone service cannot have multiple lines.
    """
    missing = [c for c in RAW_FEATURES if c not in values]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    row = {c: values[c] for c in RAW_FEATURES}
    if row["InternetService"] == "No":
        for col in INTERNET_ADDONS:
            row[col] = "No internet service"
    if row["PhoneService"] == "No":
        row["MultipleLines"] = "No phone service"
    return pd.DataFrame([row])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a table of customers given in the raw dataset format."""
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return clean(df)[RAW_FEATURES]


def predict_proba(pipeline, df: pd.DataFrame) -> pd.Series:
    """Churn probability for each row of ``df`` (raw dataset format)."""
    return pd.Series(pipeline.predict_proba(prepare(df))[:, 1], index=df.index, name="churn_probability")


def explain(pipeline, df_row: pd.DataFrame) -> pd.Series | None:
    """Per-feature contribution to the score of one customer (linear models only).

    Contribution = coefficient x transformed value, summed over the one-hot
    columns of each original feature. Positive values push towards churn.
    Returns ``None`` when the model is not linear.
    """
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return None
    transformed = pipeline[:-1].transform(prepare(df_row))
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    contrib = pd.Series(transformed[0] * clf.coef_[0], index=names)

    def origin(name: str) -> str:
        for col in CATEGORICAL:
            if name.startswith(col + "_"):
                return col
        return "tenure" if name == "log_tenure" else name

    grouped = contrib.groupby(origin).sum()
    return grouped.reindex(grouped.abs().sort_values(ascending=False).index)
