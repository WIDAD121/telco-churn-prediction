"""Run with:  pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.linear_model import LogisticRegression

from churn.data import DATA_PATH, RAW_FEATURES, clean, load_dataset, split
from churn.features import add_features, build_pipeline
from churn.predict import METADATA_PATH, MODEL_PATH, explain, load_artifacts, make_customer, predict_proba

RISKY = dict(
    gender="Female", SeniorCitizen="Yes", Partner="No", Dependents="No", tenure=2,
    PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic", OnlineSecurity="No",
    OnlineBackup="No", DeviceProtection="No", TechSupport="No", StreamingTV="Yes",
    StreamingMovies="Yes", Contract="Month-to-month", PaperlessBilling="Yes",
    PaymentMethod="Electronic check", MonthlyCharges=95.0, TotalCharges=190.0,
)
LOYAL = dict(
    gender="Male", SeniorCitizen="No", Partner="Yes", Dependents="Yes", tenure=60,
    PhoneService="Yes", MultipleLines="Yes", InternetService="DSL", OnlineSecurity="Yes",
    OnlineBackup="Yes", DeviceProtection="Yes", TechSupport="Yes", StreamingTV="No",
    StreamingMovies="No", Contract="Two year", PaperlessBilling="No",
    PaymentMethod="Credit card (automatic)", MonthlyCharges=65.0, TotalCharges=3900.0,
)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_dataset_shape_and_target(dataset):
    X, y = dataset
    assert list(X.columns) == RAW_FEATURES
    assert len(X) == 7043
    assert set(y.unique()) == {0, 1}
    assert 0.25 < y.mean() < 0.28


def test_clean_handles_blank_total_charges():
    raw = pd.read_csv(DATA_PATH)
    blanks = pd.to_numeric(raw["TotalCharges"], errors="coerce").isna()
    assert blanks.sum() == 11 and (raw.loc[blanks, "tenure"] == 0).all()
    cleaned = clean(raw)
    assert cleaned["TotalCharges"].notna().all()
    assert (cleaned.loc[blanks, "TotalCharges"] == 0).all()
    assert set(cleaned["SeniorCitizen"]) == {"Yes", "No"}


def test_add_features_does_not_modify_input(dataset):
    X, _ = dataset
    before = X.copy()
    out = add_features(X)
    pd.testing.assert_frame_equal(X, before)
    assert out["n_addons"].between(0, 6).all()
    assert not out.isin(["No internet service", "No phone service"]).any().any()


def test_split_is_stratified_and_disjoint(dataset):
    X, y = dataset
    X_train, X_test, y_train, y_test = split(X, y)
    assert len(X_test) == 1409
    assert abs(y_train.mean() - y_test.mean()) < 0.005
    assert set(X_train.index).isdisjoint(X_test.index)


def test_pipeline_learns_something(dataset):
    X, y = dataset
    X_train, X_test, y_train, y_test = split(X, y)
    model = build_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    assert proba.min() >= 0 and proba.max() <= 1
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y_test, proba) > 0.80


def test_make_customer_fixes_impossible_combinations():
    row = make_customer(**{**LOYAL, "InternetService": "No", "PhoneService": "No"})
    assert row["OnlineSecurity"].iloc[0] == "No internet service"
    assert row["MultipleLines"].iloc[0] == "No phone service"
    with pytest.raises(ValueError):
        make_customer(gender="Male")


def _saved_model_usable() -> bool:
    if not (MODEL_PATH.exists() and METADATA_PATH.exists()):
        return False
    return json.loads(METADATA_PATH.read_text(encoding="utf-8")).get("sklearn_version") == sklearn.__version__


@pytest.mark.skipif(not _saved_model_usable(), reason="run `python -m churn.train` with this scikit-learn version")
def test_saved_model_ranks_risky_above_loyal():
    pipeline, meta = load_artifacts()
    risky = predict_proba(pipeline, make_customer(**RISKY)).iloc[0]
    loyal = predict_proba(pipeline, make_customer(**LOYAL)).iloc[0]
    assert risky > meta["threshold"] > loyal
    contributions = explain(pipeline, make_customer(**RISKY))
    assert contributions is None or np.isfinite(contributions).all()
