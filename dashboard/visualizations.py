"""
Visualization Dashboard Module
--------------------------------
Generates static EDA and dashboard charts saved to static/images/.
All plots follow the dark SaaS aesthetic (#0F172A background).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import IMAGES_DIR, FIGURE_DPI, COLOR_CHURN, COLOR_NO_CHURN

os.makedirs(IMAGES_DIR, exist_ok=True)

DARK_BG  = "#FFFFFF"
DARK_AX  = "#FFFFFF"
GRID_COL = "#E5E7EB"
TEXT_COL = "#1F2937"


def _base_style(fig, ax_list=None):
    """Apply consistent dark theme to a figure."""
    fig.patch.set_facecolor(DARK_BG)
    if ax_list is None:
        ax_list = fig.get_axes()
    for ax in ax_list:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        ax.title.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)


# ─── 1  Churn Distribution ────────────────────────────────────────────────────

def plot_churn_distribution(df: pd.DataFrame, save: bool = True) -> str:
    churn_counts = df["Churn"].map({1: "Churn", 0: "No Churn"}).value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _base_style(fig, [ax1, ax2])

    colours = [COLOR_CHURN, COLOR_NO_CHURN]
    ax1.bar(churn_counts.index, churn_counts.values, color=colours, edgecolor=GRID_COL, width=0.5)
    ax1.set_title("Churn Count", fontweight="bold")
    ax1.set_ylabel("Customers")
    ax1.yaxis.grid(True, color=GRID_COL, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)
    for bar, val in zip(ax1.patches, churn_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 str(val), ha="center", color=TEXT_COL, fontweight="bold")

    wedge_props = {"edgecolor": DARK_BG, "linewidth": 2}
    ax2.pie(churn_counts.values, labels=churn_counts.index, colors=colours,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=wedge_props,
            textprops={"color": TEXT_COL, "fontsize": 11})
    ax2.set_title("Churn Rate", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "churn_distribution.png")
    if save:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Dashboard] churn_distribution saved → {path}")
    return path


# ─── 2  Correlation Heatmap ───────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> str:
    num_df = df.select_dtypes(include=[np.number])
    corr   = num_df.corr()

    # Keep top 15 correlated with Churn
    if "Churn" in corr:
        top_cols = corr["Churn"].abs().nlargest(15).index.tolist()
        corr = corr.loc[top_cols, top_cols]

    fig, ax = plt.subplots(figsize=(12, 10))
    _base_style(fig)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=ax, linewidths=0.5,
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap (Top 15 Features)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(IMAGES_DIR, "correlation_heatmap.png")
    if save:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Dashboard] correlation_heatmap saved → {path}")
    return path


# ─── 3  Churn vs Contract ─────────────────────────────────────────────────────

def plot_churn_by_contract(df_raw: pd.DataFrame, save: bool = True) -> str:
    """Expects the raw (pre-encoded) DataFrame."""
    if "Contract" not in df_raw.columns or "Churn" not in df_raw.columns:
        return ""

    churn_col = df_raw["Churn"].map({"Yes": "Churn", "No": "No Churn"}) \
        if df_raw["Churn"].dtype == object else \
        df_raw["Churn"].map({1: "Churn", 0: "No Churn"})

    plot_df = pd.crosstab(df_raw["Contract"], churn_col, normalize="index") * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    _base_style(fig)
    plot_df.plot(
        kind="bar", ax=ax, color=[COLOR_NO_CHURN, COLOR_CHURN],
        edgecolor=DARK_BG, width=0.65,
    )
    ax.set_title("Churn Rate by Contract Type", fontsize=13, fontweight="bold")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.yaxis.grid(True, color=GRID_COL, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(facecolor=DARK_AX, edgecolor=GRID_COL, labelcolor=TEXT_COL)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "churn_by_contract.png")
    if save:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Dashboard] churn_by_contract saved → {path}")
    return path


# ─── 4  Churn vs Tenure ───────────────────────────────────────────────────────

def plot_churn_by_tenure(df_raw: pd.DataFrame, save: bool = True) -> str:
    if "tenure" not in df_raw.columns or "Churn" not in df_raw.columns:
        return ""

    churn_num = df_raw["Churn"].map({"Yes": 1, "No": 0}) \
        if df_raw["Churn"].dtype == object else df_raw["Churn"]

    fig, ax = plt.subplots(figsize=(10, 5))
    _base_style(fig)
    ax.hist(
        df_raw.loc[churn_num == 0, "tenure"], bins=30,
        alpha=0.7, color=COLOR_NO_CHURN, label="No Churn", edgecolor=DARK_BG,
    )
    ax.hist(
        df_raw.loc[churn_num == 1, "tenure"], bins=30,
        alpha=0.7, color=COLOR_CHURN, label="Churn", edgecolor=DARK_BG,
    )
    ax.set_title("Tenure Distribution by Churn", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Count")
    ax.legend(facecolor=DARK_AX, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    ax.yaxis.grid(True, color=GRID_COL, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "churn_by_tenure.png")
    if save:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Dashboard] churn_by_tenure saved → {path}")
    return path


# ─── 5  Churn vs Payment Method ──────────────────────────────────────────────

def plot_churn_by_payment(df_raw: pd.DataFrame, save: bool = True) -> str:
    if "PaymentMethod" not in df_raw.columns or "Churn" not in df_raw.columns:
        return ""

    churn_col = df_raw["Churn"].map({"Yes": "Churn", "No": "No Churn"}) \
        if df_raw["Churn"].dtype == object else \
        df_raw["Churn"].map({1: "Churn", 0: "No Churn"})

    plot_df = pd.crosstab(df_raw["PaymentMethod"], churn_col, normalize="index") * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    _base_style(fig)
    plot_df.plot(
        kind="barh", ax=ax, color=[COLOR_NO_CHURN, COLOR_CHURN],
        edgecolor=DARK_BG, width=0.6,
    )
    ax.set_title("Churn Rate by Payment Method", fontsize=13, fontweight="bold")
    ax.set_xlabel("Percentage (%)")
    ax.set_ylabel("")
    ax.xaxis.grid(True, color=GRID_COL, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(facecolor=DARK_AX, edgecolor=GRID_COL, labelcolor=TEXT_COL)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "churn_by_payment.png")
    if save:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Dashboard] churn_by_payment saved → {path}")
    return path


# ─── 6  Monthly Charges Distribution ─────────────────────────────────────────

def plot_monthly_charges(df_raw: pd.DataFrame, save: bool = True) -> str:
    if "MonthlyCharges" not in df_raw.columns or "Churn" not in df_raw.columns:
        return ""

    churn_num = df_raw["Churn"].map({"Yes": 1, "No": 0}) \
        if df_raw["Churn"].dtype == object else df_raw["Churn"]

    fig, ax = plt.subplots(figsize=(10, 5))
    _base_style(fig)
    ax.hist(
        df_raw.loc[churn_num == 0, "MonthlyCharges"], bins=35,
        alpha=0.7, color=COLOR_NO_CHURN, label="No Churn", edgecolor=DARK_BG,
    )
    ax.hist(
        df_raw.loc[churn_num == 1, "MonthlyCharges"], bins=35,
        alpha=0.7, color=COLOR_CHURN, label="Churn", edgecolor=DARK_BG,
    )
    ax.set_title("Monthly Charges by Churn Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("Monthly Charges ($)")
    ax.set_ylabel("Count")
    ax.legend(facecolor=DARK_AX, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    ax.yaxis.grid(True, color=GRID_COL, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(IMAGES_DIR, "monthly_charges_dist.png")
    if save:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Dashboard] monthly_charges_dist saved → {path}")
    return path


# ─── Master Plot Generator ────────────────────────────────────────────────────

def generate_all_eda_plots(df_raw: pd.DataFrame, df_encoded: pd.DataFrame):
    """Run all EDA visualisations. df_raw is pre-encoding, df_encoded is post."""
    plot_churn_distribution(df_encoded)
    plot_correlation_heatmap(df_encoded)
    plot_churn_by_contract(df_raw)
    plot_churn_by_tenure(df_raw)
    plot_churn_by_payment(df_raw)
    plot_monthly_charges(df_raw)
    print("[Dashboard] All EDA plots generated.")
