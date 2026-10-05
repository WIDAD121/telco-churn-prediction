"""Train, select and evaluate the churn model.

Run from the repository root:

    python -m churn.train

Protocol
--------
1. Stratified 80/20 split. The test set is not looked at until step 5.
2. Each candidate is a full pipeline (features -> scaling/encoding -> model),
   tuned by 5-fold cross-validation on the training set. Because the
   preprocessing is inside the pipeline, nothing leaks between folds.
3. Model selection on cross-validated PR-AUC with the one-standard-error rule:
   among the candidates within one standard error of the best score, keep the
   simplest one.
4. The decision threshold is chosen on out-of-fold predictions of the training
   set, never on the test set: the highest threshold that still catches
   TARGET_RECALL of the churners (a missed churner costs more than a
   retention offer sent to a loyal customer).
5. The selected pipeline is refitted on the whole training set and evaluated
   once on the test set.
"""
from __future__ import annotations

import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from churn import evaluate as ev
from churn.data import RANDOM_STATE, ROOT, load_dataset, split
from churn.features import build_pipeline

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
N_SPLITS = 5
TARGET_RECALL = 0.80  # business choice: catch at least 80% of churners

# Candidates, ordered from the simplest to the most complex.
# The class imbalance (26.5% churners) is not "fixed" by resampling or class
# weights: both distort the predicted probabilities without improving how well
# customers are ranked (see experiments/smote_vs_class_weight.py). It is handled
# where it matters, at the decision threshold (step 4 of the protocol).
CANDIDATES = {
    "Logistic Regression": (
        LogisticRegression(max_iter=5000),
        {"clf__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"clf__max_depth": [3, 4, 5, 6, 8], "clf__min_samples_leaf": [20, 50, 100]},
    ),
    "SVM (RBF)": (
        SVC(probability=True, random_state=RANDOM_STATE),
        {"clf__C": [0.3, 1.0, 3.0]},
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        {"clf__max_depth": [6, 10, None], "clf__min_samples_leaf": [5, 10, 20]},
    ),
    "Gradient Boosting": (
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        {"clf__learning_rate": [0.02, 0.05], "clf__max_iter": [200, 400],
         "clf__max_leaf_nodes": [4, 8]},
    ),
}


def tune_candidates(X_train, y_train, cv) -> dict:
    """Grid-search every candidate and collect cross-validated scores."""
    results = {}
    for name, (clf, grid) in CANDIDATES.items():
        search = GridSearchCV(
            build_pipeline(clf), grid, cv=cv, n_jobs=-1, refit="average_precision",
            scoring={"average_precision": "average_precision", "roc_auc": "roc_auc"},
        )
        search.fit(X_train, y_train)
        i = search.best_index_
        fold_pr = np.array([search.cv_results_[f"split{k}_test_average_precision"][i]
                            for k in range(N_SPLITS)])
        # Out-of-fold probabilities of the tuned pipeline (used for threshold metrics)
        oof = cross_val_predict(search.best_estimator_, X_train, y_train, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        results[name] = {
            "pipeline": search.best_estimator_,
            "params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_pr_auc": float(fold_pr.mean()),
            "cv_pr_auc_std": float(fold_pr.std(ddof=1)),
            "cv_pr_auc_se": float(fold_pr.std(ddof=1) / np.sqrt(N_SPLITS)),
            "cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][i]),
            "cv_roc_auc_std": float(search.cv_results_["std_test_roc_auc"][i]),
            "oof_proba": oof,
        }
        table = ev.threshold_table(y_train, oof)
        thr = ev.threshold_for_recall(table, TARGET_RECALL)
        results[name]["oof_threshold"] = thr
        results[name]["oof_at_target_recall"] = ev.threshold_metrics(y_train, oof, thr)
        print(f"  {name:20s} CV PR-AUC {fold_pr.mean():.4f} +/- {fold_pr.std(ddof=1):.4f}"
              f"   CV ROC-AUC {results[name]['cv_roc_auc']:.4f}   {results[name]['params']}")
    return results


def select_model(results: dict) -> tuple[str, str]:
    """One-standard-error rule: simplest candidate statistically tied with the best."""
    best = max(results, key=lambda n: results[n]["cv_pr_auc"])
    floor = results[best]["cv_pr_auc"] - results[best]["cv_pr_auc_se"]
    for name in CANDIDATES:  # ordered simplest -> most complex
        if results[name]["cv_pr_auc"] >= floor:
            return name, best
    return best, best


def churn_drivers(pipeline, X_test, y_test) -> tuple[pd.Series, str]:
    """Top factors: coefficients for a linear model, permutation importance otherwise."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        coefs = pd.Series(clf.coef_[0], index=names)
        top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(12)
        return top, "Coefficient (standardised features): right = higher churn risk"
    imp = permutation_importance(pipeline, X_test, y_test, scoring="average_precision",
                                 n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    series = pd.Series(imp.importances_mean, index=X_test.columns)
    return series.sort_values(ascending=False).head(12), "Drop in PR-AUC when the feature is shuffled"


def write_results_markdown(path: Path, results: dict, selected: str, best: str,
                           threshold: float, threshold_mcc: float, test: dict) -> None:
    lines = ["## Cross-validation on the training set (5 folds)", "",
             f"| Model | PR-AUC | ROC-AUC | Precision at {TARGET_RECALL:.0%} recall | MCC at {TARGET_RECALL:.0%} recall | Best parameters |",
             "|---|---|---|---|---|---|"]
    for name, r in results.items():
        o = r["oof_at_target_recall"]
        mark = " **(selected)**" if name == selected else ""
        params = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        lines.append(f"| {name}{mark} | {r['cv_pr_auc']:.3f} ± {r['cv_pr_auc_std']:.3f} | "
                     f"{r['cv_roc_auc']:.3f} ± {r['cv_roc_auc_std']:.3f} | {o['precision']:.3f} | "
                     f"{o['mcc']:.3f} | {params} |")
    lines += ["", f"Highest CV PR-AUC: {best}. Selected with the one-standard-error rule: {selected}.",
              f"Decision threshold chosen on out-of-fold predictions for {TARGET_RECALL:.0%} recall: "
              f"{threshold:.2f} (the threshold that maximises MCC would be {threshold_mcc:.2f}).", "",
              "## Test set (1,409 customers, used once)", "",
              "| Threshold | Accuracy | Precision | Recall | F1 | MCC |", "|---|---|---|---|---|---|"]
    for label, key in [(f"{threshold:.2f} (chosen, recall target)", "at_threshold"),
                       (f"{threshold_mcc:.2f} (max MCC)", "at_max_mcc_threshold")]:
        m = test[key]
        lines.append(f"| {label} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['f1']:.3f} | {m['mcc']:.3f} |")
    ci = test["ci95"]
    lines += ["", f"ROC-AUC {test['roc_auc']:.3f} (95% CI {ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f}), "
                  f"PR-AUC {test['pr_auc']:.3f} (95% CI {ci['pr_auc'][0]:.3f}-{ci['pr_auc'][1]:.3f}). "
                  f"Brier score {test['brier']:.3f} (always predicting the base rate: {test['brier_baseline']:.3f}).", "",
              "### All candidates on the test set (for reference only, not used for selection)", "",
              "| Model | ROC-AUC | PR-AUC |", "|---|---|---|"]
    for name, m in test["all_models"].items():
        lines.append(f"| {name} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | churn rate {y.mean():.1%}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    print("\nTuning candidates with 5-fold cross-validation:")
    results = tune_candidates(X_train, y_train, cv)

    selected, best = select_model(results)
    print(f"\nHighest CV PR-AUC: {best}  ->  selected (one-standard-error rule): {selected}")

    # Threshold from out-of-fold predictions on the training set
    table = ev.threshold_table(y_train, results[selected]["oof_proba"])
    threshold = ev.threshold_for_recall(table, TARGET_RECALL)
    threshold_mcc = float(table["mcc"].idxmax())
    print(f"Decision threshold for {TARGET_RECALL:.0%} out-of-fold recall: {threshold:.2f}"
          f"  (max-MCC alternative: {threshold_mcc:.2f})")

    # Final evaluation on the held-out test set
    pipeline = results[selected]["pipeline"]  # already refitted on the full training set
    test_probas = {n: r["pipeline"].predict_proba(X_test)[:, 1] for n, r in results.items()}
    proba = test_probas[selected]
    test = {
        **ev.ranking_metrics(y_test, proba),
        "ci95": ev.bootstrap_ci(y_test, proba),
        "brier": ev.brier(y_test, proba),
        "brier_baseline": ev.brier(y_test, np.full(len(y_test), y_train.mean())),
        "at_0.5": ev.threshold_metrics(y_test, proba, 0.5),
        "at_threshold": ev.threshold_metrics(y_test, proba, threshold),
        "at_max_mcc_threshold": ev.threshold_metrics(y_test, proba, threshold_mcc),
        "all_models": {n: ev.ranking_metrics(y_test, p) for n, p in test_probas.items()},
    }
    print(f"\nTest ROC-AUC {test['roc_auc']:.4f} | PR-AUC {test['pr_auc']:.4f} | Brier {test['brier']:.4f}")
    for key in ("at_threshold", "at_max_mcc_threshold"):
        print(f"  {key:13s}", {k: round(v, 4) for k, v in test[key].items()})

    # Figures
    ev.plot_roc_pr(y_test, test_probas, selected, FIG_DIR)
    ev.plot_threshold(table, threshold, FIG_DIR, TARGET_RECALL)
    ev.plot_confusion(y_test, proba, threshold, FIG_DIR)
    ev.plot_calibration(y_test, proba, FIG_DIR)
    drivers, xlabel = churn_drivers(pipeline, X_test, y_test)
    ev.plot_drivers(drivers, f"Main churn drivers ({selected})", xlabel, FIG_DIR)

    # Artifacts
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "churn_pipeline.joblib", compress=3)
    metadata = {
        "model": selected,
        "params": results[selected]["params"],
        "threshold": threshold,
        "target_recall": TARGET_RECALL,
        "test_roc_auc": round(test["roc_auc"], 4),
        "test_pr_auc": round(test["pr_auc"], 4),
        "test_recall_at_threshold": round(test["at_threshold"]["recall"], 4),
        "test_precision_at_threshold": round(test["at_threshold"]["precision"], 4),
        "n_train": int(len(X_train)),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = {
        "selected_model": selected,
        "best_cv_model": best,
        "threshold": threshold,
        "threshold_max_mcc": threshold_mcc,
        "target_recall": TARGET_RECALL,
        "cross_validation": {
            n: {k: v for k, v in r.items() if k not in ("pipeline", "oof_proba")}
            for n, r in results.items()
        },
        "test": test,
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    table.loc[[t for t in table.index if round(t * 100) % 5 == 0]].round(3).to_csv(REPORT_DIR / "threshold_table.csv")
    write_results_markdown(REPORT_DIR / "results.md", results, selected, best, threshold, threshold_mcc, test)
    size_kb = (MODEL_DIR / "churn_pipeline.joblib").stat().st_size / 1024
    print(f"\nSaved models/churn_pipeline.joblib ({size_kb:.0f} KB), reports/metrics.json, "
          f"reports/results.md and figures in reports/figures/")


if __name__ == "__main__":
    main()
