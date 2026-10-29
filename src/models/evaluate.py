"""
Evaluation metrics, threshold optimization, and visualization engine for Predictive Maintenance.
Strictly separates out-of-fold validation from held-out test evaluation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import OUTPUTS_DIR


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes comprehensive classification metrics at a given decision threshold.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class (failure = 1).
    threshold : float
        Decision cutoff threshold (default 0.5).

    Returns
    -------
    dict
        Dictionary containing threshold-dependent and threshold-independent metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # Threshold-independent metrics
    try:
        avg_precision = float(average_precision_score(y_true, y_prob))
    except Exception:
        avg_precision = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    # Threshold-dependent metrics
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    fn_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    fp_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "average_precision_pr_auc": round(avg_precision, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "f2_score": round(f2, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "false_negative_rate": round(fn_rate, 4),
            "false_positive_rate": round(fp_rate, 4),
        },
        "operational_tradeoff_note": (
            f"At threshold {threshold:.2f}: {fn} failures were missed (False Negatives - catastrophic shutdown risk), "
            f"and {fp} normal machines triggered false alarms (False Positives - unnecessary inspection cost)."
        ),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f2",
    step: float = 0.01,
) -> Dict[str, Any]:
    """
    Evaluates candidate thresholds strictly on training or validation probabilities.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels from training/validation fold.
    y_prob : array-like
        Out-of-fold predicted probabilities for class 1.
    metric : str
        Metric to maximize ('f2' or 'f1'). Defaults to 'f2' (prioritizing recall).
    step : float
        Grid step for thresholds (default: 0.01).

    Returns
    -------
    dict
        Optimal threshold, best score, and comparison table.
    """
    thresholds = np.arange(0.05, 0.95 + step, step)
    best_score = -1.0
    best_threshold = 0.5
    curve_records = []

    for t in thresholds:
        metrics = calculate_classification_metrics(y_true, y_prob, threshold=t)
        score = metrics["f2_score"] if metric == "f2" else metrics["f1_score"]
        curve_records.append(
            {
                "threshold": round(float(t), 2),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "f2_score": metrics["f2_score"],
            }
        )
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    default_metrics = calculate_classification_metrics(y_true, y_prob, threshold=0.5)
    optimal_metrics = calculate_classification_metrics(y_true, y_prob, threshold=best_threshold)

    return {
        "optimization_metric": metric,
        "optimal_threshold": round(best_threshold, 2),
        "best_score": round(best_score, 4),
        "default_0_5_metrics": default_metrics,
        "optimal_threshold_metrics": optimal_metrics,
        "threshold_curve": curve_records,
        "method_documentation": (
            "Threshold selected via grid search on out-of-fold training probabilities to maximize "
            f"{metric.upper()} score. In predictive maintenance, missing a breakdown (False Negative) "
            "is significantly more detrimental than a preventative check (False Positive), justifying "
            "an F2-weighting (beta=2). The test set was never accessed during threshold search."
        ),
    }


def plot_confusion_matrix(
    cm_dict: Dict[str, int],
    title: str,
    output_path: Path,
) -> None:
    """Plots and saves a professional confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(
        [
            [cm_dict["true_negatives"], cm_dict["false_positives"]],
            [cm_dict["false_negatives"], cm_dict["true_positives"]],
        ]
    )

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.85)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            color = "white" if val > matrix.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Normal", "Predicted Failure"])
    ax.set_yticklabels(["Actual Normal", "Actual Failure"])
    ax.set_title(f"{title}\n[Synthetic Benchmark Data]", fontsize=10, pad=12)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_precision_recall_curves(
    curves_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    baseline_rate: float,
    output_path: Path,
) -> None:
    """Plots Precision-Recall curves comparing models against empirical baseline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    for label, (y_true, y_prob, ap_score) in curves_data.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(recall, precision, lw=2, label=f"{label} (AP = {ap_score:.3f})")

    # Baseline no-skill classifier
    ax.axhline(
        y=baseline_rate,
        color="grey",
        linestyle="--",
        label=f"No-Skill Baseline ({baseline_rate * 100:.2f}%)",
    )

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Precision-Recall Curves (PR-AUC)\n[Synthetic Benchmark Data - AI4I 2020]", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curves(
    curves_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: Path,
) -> None:
    """Plots ROC curves comparing models against chance diagonal."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    for label, (y_true, y_prob, roc_auc) in curves_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (ROC-AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Chance (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC)\n[Synthetic Benchmark Data - AI4I 2020]", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_model_comparison_bar(
    metrics_summary: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plots a bar chart comparing performance across models and partitions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metrics_summary)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    x = np.arange(len(df))
    width = 0.2

    metrics = [("average_precision_pr_auc", "PR-AUC (AP)"), ("f1_score", "F1 Score"), ("f2_score", "F2 Score"), ("balanced_accuracy", "Bal Acc")]
    for idx, (m_col, m_name) in enumerate(metrics):
        offset = (idx - 1.5) * width
        ax.bar(x + offset, df[m_col], width, label=m_name)

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison (Out-of-Fold vs. Test Split)\n[Synthetic Benchmark Data]", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df["evaluation_label"], rotation=15, ha="right", fontsize=8)
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
