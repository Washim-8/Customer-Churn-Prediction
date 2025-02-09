"""
Explainability Module (SHAP)
-----------------------------
Generates SHAP-based feature importance and individual prediction explanations.
Handles graceful fallback when SHAP is unavailable or the model is unsupported.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import IMAGES_DIR, FIGURE_DPI

os.makedirs(IMAGES_DIR, exist_ok=True)


def _get_shap_explainer(model, X_background: pd.DataFrame):
    """
    Return the appropriate SHAP explainer for the given model.
    Tree-based → TreeExplainer; otherwise → KernelExplainer (slow but universal).
    """
    try:
        import shap
        tree_types = (
            "XGBClassifier", "RandomForestClassifier",
            "GradientBoostingClassifier", "DecisionTreeClassifier",
        )
        model_type = type(model).__name__
        if model_type in tree_types:
            return shap.TreeExplainer(model), shap
        else:
            bg = shap.sample(X_background, 100)
            return shap.KernelExplainer(model.predict_proba, bg), shap
    except ImportError:
        return None, None


def shap_summary_plot(
    model,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    save: bool = True,
) -> str:
    """
    SHAP beeswarm / summary plot for top features.
    Falls back to an empty placeholder if SHAP is not installed.
    """
    explainer, shap = _get_shap_explainer(model, X_sample)
    path = os.path.join(IMAGES_DIR, "shap_summary.png")

    if explainer is None:
        _save_placeholder("SHAP not available\n(pip install shap)", path)
        return path

    try:
        shap_values = explainer.shap_values(X_sample)
        # For binary classifiers shap_values is a list [neg, pos]
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0F172A")
        shap.summary_plot(
            sv, X_sample,
            feature_names=feature_names,
            show=False,
            plot_type="dot",
        )
        plt.gcf().set_facecolor("#0F172A")
        plt.tight_layout()
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight",
                    facecolor="#0F172A")
        plt.close()
        print(f"[Explainability] SHAP summary saved → {path}")
    except Exception as exc:
        print(f"[Explainability] SHAP summary failed: {exc}")
        _save_placeholder(f"SHAP error:\n{exc}", path)

    return path


def shap_waterfall_plot(
    model,
    X_row: pd.DataFrame,
    feature_names: list[str],
    save: bool = True,
) -> str:
    """
    SHAP waterfall plot for a single prediction (web result page).
    """
    explainer, shap = _get_shap_explainer(model, X_row)
    path = os.path.join(IMAGES_DIR, "shap_waterfall.png")

    if explainer is None:
        _save_placeholder("SHAP not available", path)
        return path

    try:
        shap_values = explainer(X_row)
        if hasattr(shap_values, "values"):
            sv = shap_values[0]
        else:
            sv = shap_values

        fig = plt.figure(figsize=(10, 6), facecolor="#0F172A")
        shap.plots.waterfall(sv, show=False)
        plt.gcf().set_facecolor("#0F172A")
        plt.tight_layout()
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight",
                    facecolor="#0F172A")
        plt.close()
        print(f"[Explainability] SHAP waterfall saved → {path}")
    except Exception as exc:
        print(f"[Explainability] Waterfall failed: {exc}")
        _save_placeholder(f"SHAP error:\n{exc}", path)

    return path


def _save_placeholder(text: str, path: str):
    """Save a dark placeholder image when SHAP is unavailable."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0F172A")
    ax.set_facecolor("#0F172A")
    ax.text(
        0.5, 0.5, text,
        ha="center", va="center",
        fontsize=14, color="#94A3B8",
        transform=ax.transAxes,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="#0F172A")
    plt.close(fig)
