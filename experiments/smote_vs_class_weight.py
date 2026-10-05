"""Optional experiment: class weights vs SMOTE, and where SMOTE must be applied.

    pip install imbalanced-learn
    python -m experiments.smote_vs_class_weight

Four ways of handling the 26.5% / 73.5% imbalance are cross-validated on the
training set, for a logistic regression and a random forest:

A. nothing (threshold chosen later)    -> what churn/train.py does
B. class weights
C. SMOTE inside the pipeline           -> resampling is re-done on the training
                                          part of each fold (correct)
D. SMOTE once, before cross-validation -> what the first notebook did. The
                                          validation folds then contain synthetic
                                          points interpolated from training rows,
                                          so the scores are inflated.

Read PR-AUC and ROC-AUC to compare A, B and C: they do not depend on the
threshold. F1 is computed at 0.5, which penalises A because its threshold has
not been tuned yet; it is shown because it makes the inflation of D obvious.
"""
import warnings

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import FunctionTransformer

from churn.data import RANDOM_STATE, load_dataset, split
from churn.features import add_features, build_pipeline, build_preprocessor

SCORING = ["f1", "average_precision", "roc_auc"]


def smote_pipeline(clf) -> ImbPipeline:
    return ImbPipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("preprocess", build_preprocessor()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", clf),
    ])


def main() -> None:
    warnings.filterwarnings("ignore")
    X, y = load_dataset()
    X_train, _, y_train, _ = split(X, y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }
    rows = []
    for name, clf in models.items():
        weighted = clone(clf).set_params(class_weight="balanced")
        runs = {"A. no resampling": (build_pipeline(clone(clf)), X_train, y_train),
                "B. class weights": (build_pipeline(weighted), X_train, y_train),
                "C. SMOTE inside each fold": (smote_pipeline(clone(clf)), X_train, y_train)}
        # D: preprocess + SMOTE on the whole training set first, then cross-validate
        prep = build_pipeline("passthrough")
        X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(prep.fit_transform(X_train), y_train)
        runs["D. SMOTE before CV (leaky)"] = (clone(clf), X_res, y_res)

        for label, (estimator, X_cv, y_cv) in runs.items():
            scores = cross_validate(estimator, X_cv, y_cv, cv=cv, scoring=SCORING, n_jobs=-1)
            rows.append({"model": name, "strategy": label,
                         **{m: round(scores[f"test_{m}"].mean(), 3) for m in SCORING}})
    table = pd.DataFrame(rows).rename(columns={"average_precision": "pr_auc"})
    print(table.to_string(index=False))
    print("\nStrategy D looks better only because its validation folds are not real customers.")


if __name__ == "__main__":
    main()
