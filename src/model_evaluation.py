"""
Model Evaluation Module
------------------------
Computes classification metrics, generates comparison tables,
plots ROC curves, confusion matrices, and feature importance charts.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import IMAGES_DIR, FIGURE_DPI, COLOR_CHURN, COLOR_NO_CHURN


os.makedirs(IMAGES_DIR, exist_ok=True)

# ─── Single Model Metrics ──────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """
    Compute standard classification metrics for a single model.
    Returns a metrics dictionary.
    """
    y_pred  = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "Model":     model_name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC AUC":   round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else None,
    }
    return metrics


# ─── Comparison Table ──────────────────────────────────────────────────────────

def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Build a sorted comparison DataFrame from a list of metrics dicts.
    """
    df = pd.DataFrame(results).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


# ─── ROC Curve Plot ────────────────────────────────────────────────────────────

def plot_roc_curves(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> str:
    """
    Plot overlapping ROC curves for all trained models.
    """
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    colours = ["#0F766E", "#059669", "#10B981", "#D97706", "#3B82F6", "#EC4899"]

    for (name, model), colour in zip(models.items(), colours):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, lw=2, color=colour, label=f"{name}  (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", lw=1, alpha=0.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", color="#1F2937", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="#1F2937", fontsize=12)
    ax.set_title("ROC Curves – All Models", color="#1F2937", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", facecolor="#FFFFFF", edgecolor="#E5E7EB", labelcolor="#1F2937")
    ax.tick_params(colors="#1F2937")
    for spine in ax.spines.values():
        spine.set_edgecolor("#E5E7EB")

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "roc_curve.png")
    if save:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[Evaluation] ROC curve saved → {path}")
    plt.close(fig)
    return path


# ─── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Best Model",
    save: bool = True,
) -> str:
    """
    Heatmap confusion matrix for the given model.
    """
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#FFFFFF")
    sns.heatmap(
        cm,
        annot=True, fmt="d",
        cmap="BuGn",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax, linewidths=.5, cbar=False,
        annot_kws={"size": 14, "weight": "bold", "color": "#1F2937"},
    )
    ax.set_xlabel("Predicted", color="#1F2937", fontsize=12)
    ax.set_ylabel("Actual",    color="#1F2937", fontsize=12)
    ax.set_title(f"Confusion Matrix – {model_name}", color="#1F2937", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#1F2937")

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "confusion_matrix.png")
    if save:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[Evaluation] Confusion matrix saved → {path}")
    plt.close(fig)
    return path


# ─── Feature Importance ────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list[str],
    model_name: str = "Best Model",
    top_n: int = 20,
    save: bool = True,
) -> str:
    """
    Bar chart of top-N feature importances.
    Works with tree-based models (feature_importances_) and
    linear models (coef_).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print("[Evaluation] Model has no feature_importances_ or coef_. Skipping.")
        return ""

    fi = pd.Series(importances, index=feature_names).nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    bars = ax.barh(
        fi.index, fi.values,
        color="#0F766E",
        edgecolor="#E5E7EB", linewidth=0.5,
    )
    ax.set_xlabel("Importance", color="#1F2937", fontsize=12)
    ax.set_title(f"Top-{top_n} Feature Importances – {model_name}",
                 color="#1F2937", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#1F2937")
    for spine in ax.spines.values():
        spine.set_edgecolor("#E5E7EB")

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "feature_importance.png")
    if save:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[Evaluation] Feature importance saved → {path}")
    plt.close(fig)
    return path


# ─── Model Comparison Bar Chart ────────────────────────────────────────────────

def plot_model_comparison(results: list[dict], save: bool = True) -> str:
    """
    Grouped bar chart comparing all models across key metrics.
    """
    df = pd.DataFrame(results)
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    x = np.arange(len(df["Model"]))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    colours = ["#0F766E", "#059669", "#D97706", "#DC2626", "#3B82F6"]

    for i, (metric, colour) in enumerate(zip(metrics, colours)):
        offset = (i - 2) * width
        rects  = ax.bar(x + offset, df[metric], width, label=metric, color=colour, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], color="#1F2937", rotation=15, ha="right")
    ax.set_ylabel("Score", color="#1F2937", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison Dashboard", color="#1F2937", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#FFFFFF", edgecolor="#E5E7EB", labelcolor="#1F2937")
    ax.tick_params(colors="#1F2937")
    ax.yaxis.grid(True, color="#E5E7EB", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#E5E7EB")

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "model_comparison.png")
    if save:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[Evaluation] Model comparison saved → {path}")
    plt.close(fig)
    return path
