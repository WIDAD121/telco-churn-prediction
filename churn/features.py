"""Feature engineering and preprocessing, packaged as one scikit-learn pipeline.

Everything that learns from data (scaling, encoding) lives inside the pipeline,
so it is re-fitted on the training part of every cross-validation fold and the
exact same transformation is applied at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn.data import CATEGORICAL, NUMERIC

INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
ENGINEERED = ["n_addons", "log_tenure"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature engineering (safe to apply before any split).

    * "No internet service" / "No phone service" are folded into "No": that
      information is already carried by ``InternetService`` and
      ``PhoneService``, and keeping it creates six perfectly collinear columns.
    * ``n_addons``: number of internet add-ons the customer subscribes to.
    * ``log_tenure``: churn risk drops fast in the first months and then
      flattens, which a log scale captures better than raw months.
    """
    X = X.copy()
    for col in INTERNET_ADDONS:
        X[col] = X[col].replace("No internet service", "No")
    X["MultipleLines"] = X["MultipleLines"].replace("No phone service", "No")
    X["n_addons"] = (X[INTERNET_ADDONS] == "Yes").sum(axis=1).astype(float)
    X["log_tenure"] = np.log1p(X["tenure"].astype(float))
    return X


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC + ENGINEERED),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(add_features, validate=False)),
            ("preprocess", build_preprocessor()),
            ("clf", classifier),
        ]
    )
