"""Metrics and figures used by the training script."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

THRESHOLD_GRID = np.round(np.arange(0.10, 0.91, 0.01), 2)


def threshold_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Metrics that depend on the decision threshold."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
    }


def ranking_metrics(y_true, proba) -> dict:
    """Threshold-free metrics: how well the model orders customers by risk."""
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def brier(y_true, proba) -> float:
    """Mean squared error of the probabilities (lower is better)."""
    return float(brier_score_loss(y_true, proba))


def threshold_table(y_true, proba, grid=THRESHOLD_GRID) -> pd.DataFrame:
    rows = [{"threshold": t, **threshold_metrics(y_true, proba, t)} for t in grid]
    return pd.DataFrame(rows).set_index("threshold")


def threshold_for_recall(table: pd.DataFrame, target_recall: float) -> float:
    """Highest threshold whose recall still reaches the target."""
    ok = table.index[table["recall"] >= target_recall]
    return float(ok.max()) if len(ok) else float(table.index.min())


def bootstrap_ci(y_true, proba, n_boot: int = 1000, seed: int = 42) -> dict:
    """95% bootstrap confidence intervals for ROC-AUC and PR-AUC on the test set."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    roc, pr = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        roc.append(roc_auc_score(y_true[idx], proba[idx]))
        pr.append(average_precision_score(y_true[idx], proba[idx]))
    return {
        "roc_auc": [float(np.percentile(roc, 2.5)), float(np.percentile(roc, 97.5))],
        "pr_auc": [float(np.percentile(pr, 2.5)), float(np.percentile(pr, 97.5))],
    }


# ----------------------------------------------------------------- figures --

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_pr(y_true, probas: dict, selected: str, out_dir: Path) -> None:
    """ROC and precision-recall curves of every candidate on the test set."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, proba in probas.items():
        style = {"linewidth": 2.6} if name == selected else {"linewidth": 1.2, "alpha": 0.75}
        fpr, tpr, _ = roc_curve(y_true, proba)
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_true, proba):.3f})", **style)
        precision, recall, _ = precision_recall_curve(y_true, proba)
        axes[1].step(recall, precision, where="post",
                     label=f"{name} ({average_precision_score(y_true, proba):.3f})", **style)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate (recall)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set_title("ROC curves on the test set (AUC)")
    axes[1].axhline(np.mean(y_true), color="k", linestyle="--", linewidth=0.8)
    axes[1].set_title("Precision-recall curves on the test set (average precision)")
    for ax in axes:
        ax.legend(fontsize=8, loc="best")
    _save(fig, out_dir / "roc_pr_curves.png")


def plot_threshold(table: pd.DataFrame, chosen: float, out_dir: Path, target_recall: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col in ["precision", "recall", "f1", "mcc"]:
        ax.plot(table.index, table[col], label=col.upper() if col in ("f1", "mcc") else col.capitalize())
    ax.axvline(chosen, color="red", linestyle="--", label=f"Chosen threshold ({chosen:.2f})")
    if target_recall is not None:
        ax.axhline(target_recall, color="grey", linestyle=":", linewidth=1, label=f"Recall target ({target_recall:.0%})")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score (out-of-fold predictions, training set)")
    ax.set_title("Effect of the decision threshold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    _save(fig, out_dir / "threshold_analysis.png")


def plot_confusion(y_true, proba, threshold: float, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    cm = confusion_matrix(y_true, (np.asarray(proba) >= threshold).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Test set, threshold = {threshold:.2f}")
    _save(fig, out_dir / "confusion_matrix.png")


def plot_calibration(y_true, proba, out_dir: Path, n_bins: int = 10) -> None:
    """Reliability diagram: do customers scored around 30% churn about 30% of the time?"""
    from sklearn.calibration import calibration_curve

    observed, predicted = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.plot(predicted, observed, "o-", label="Model (10 equal-size groups)")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration on the test set")
    ax.legend(fontsize=8)
    _save(fig, out_dir / "calibration.png")


def plot_drivers(drivers: pd.Series, title: str, xlabel: str, out_dir: Path) -> None:
    drivers = drivers.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(drivers) + 1.4))
    colors = ["#c0392b" if v > 0 else "#1f7a4d" for v in drivers.values]
    ax.barh(drivers.index, drivers.values, color=colors)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    _save(fig, out_dir / "churn_drivers.png")
