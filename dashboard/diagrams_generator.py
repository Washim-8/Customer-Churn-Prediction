"""
System Diagram Generator
-------------------------
Creates the three required system/design diagrams using Matplotlib:
  • System Architecture Diagram
  • DFD Level 0
  • DFD Level 1
  • ER Diagram

Output directory: diagrams/
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DIAGRAMS_DIR, FIGURE_DPI

os.makedirs(DIAGRAMS_DIR, exist_ok=True)

DARK_BG   = "#FFFFFF"
BOX_FILL  = "#FFFFFF"
BOX_EDGE  = "#E5E7EB"
ARROW_COL = "#0F766E"
TEXT_COL  = "#1F2937"
ACC_COL   = "#0F766E"
RED_COL   = "#DC2626"


def _draw_box(ax, x, y, w, h, label, sublabel="", color=BOX_FILL, edge=BOX_EDGE, fs=10):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=edge, linewidth=1.5, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y + (0.07 if sublabel else 0), label,
            ha="center", va="center", fontsize=fs,
            color=TEXT_COL, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.12, sublabel,
                ha="center", va="center", fontsize=7,
                color="#94A3B8", zorder=4)


def _arrow(ax, x1, y1, x2, y2, color=ARROW_COL):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color,
            lw=1.8, mutation_scale=15,
        ), zorder=2,
    )


# ─── 1  System Architecture ───────────────────────────────────────────────────

def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(10, 14), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    ax.text(5, 13.3, "System Architecture",
            ha="center", va="center", fontsize=16,
            color=TEXT_COL, fontweight="bold")
    ax.text(5, 12.9, "Customer Churn Prediction System",
            ha="center", va="center", fontsize=10, color="#94A3B8")

    nodes = [
        (5, 12.2, "👤  User",            "Browser / CLI"),
        (5, 10.8, "🌐  Web Interface",   "Flask Templates (HTML/CSS/JS)"),
        (5,  9.4, "⚙️  Flask Backend",   "app.py  – REST Endpoints"),
        (5,  8.0, "🔧  ML Engine",       "src/prediction.py"),
        (5,  6.6, "📦  Model File",       "models/churn_model.pkl"),
        (5,  5.2, "📊  Visualization",    "Dashboard Charts (PNG)"),
        (5,  3.8, "🎯  Prediction Output","Churn Label + Probability"),
    ]
    for x, y, label, sub in nodes:
        _draw_box(ax, x, y, 6.5, 0.75, label, sub)

    for i in range(len(nodes) - 1):
        _arrow(ax, nodes[i][0], nodes[i][1] - 0.38,
                   nodes[i+1][0], nodes[i+1][1] + 0.38)

    # Side note: data flow
    for i, (x, y, *_) in enumerate(nodes):
        ax.text(8.6, y, f"Step {i+1}", fontsize=7,
                color=ACC_COL, va="center", ha="left")

    plt.tight_layout()
    path = os.path.join(DIAGRAMS_DIR, "system_architecture.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Diagrams] system_architecture.png saved.")
    return path


# ─── 2  DFD Level 0 ───────────────────────────────────────────────────────────

def generate_dfd_level0():
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(6, 5.6, "Data Flow Diagram – Level 0 (Context Diagram)",
            ha="center", fontsize=14, color=TEXT_COL, fontweight="bold")

    # External entities (squares)
    for x, y, lbl, col in [(1.5, 3, "USER\n(External Entity)", "#0F766E"),
                             (10.5, 3, "Prediction\nResult", ACC_COL)]:
        rect = FancyBboxPatch((x-1.2, y-0.7), 2.4, 1.4,
                               boxstyle="square,pad=0.05",
                               facecolor="#F3F4F6" if col != "#0F766E" else "#E5E7EB",
                               edgecolor=col, lw=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, lbl, ha="center", va="center", fontsize=10,
                color=TEXT_COL, fontweight="bold", zorder=4)

    # Process circle
    circle = plt.Circle((6, 3), 1.4, facecolor="#1E293B", edgecolor=BOX_EDGE,
                          linewidth=2, zorder=3)
    ax.add_patch(circle)
    ax.text(6, 3.18, "Churn Prediction", ha="center", va="center",
            fontsize=9, color=TEXT_COL, fontweight="bold", zorder=4)
    ax.text(6, 2.82, "System", ha="center", va="center",
            fontsize=9, color="#94A3B8", zorder=4)

    # Arrows
    _arrow(ax, 2.7, 3, 4.6, 3)
    _arrow(ax, 7.4, 3, 9.3, 3)
    ax.text(3.65, 3.15, "Customer Data", ha="center", fontsize=8, color="#94A3B8")
    ax.text(8.35, 3.15, "Churn Label +\nProbability", ha="center",
            fontsize=8, color="#94A3B8")

    plt.tight_layout()
    path = os.path.join(DIAGRAMS_DIR, "dfd_level0.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Diagrams] dfd_level0.png saved.")
    return path


# ─── 3  DFD Level 1 ───────────────────────────────────────────────────────────

def generate_dfd_level1():
    fig, ax = plt.subplots(figsize=(10, 14), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    ax.text(5, 13.5, "Data Flow Diagram – Level 1",
            ha="center", fontsize=14, color=TEXT_COL, fontweight="bold")

    processes = [
        (5, 12.4, "User Input",          "Web Form / API"),
        (5, 10.8, "Data Preprocessing",  "Cleaning · Imputation · Encoding"),
        (5,  9.2, "Feature Engineering", "Tenure Group · ARPU · Service Count"),
        (5,  7.6, "ML Model Inference",  "Best Model  (XGBoost / RF / GBM)"),
        (5,  6.0, "Prediction Output",   "Churn Label + Probability + Risk"),
        (5,  4.4, "Visualization",       "Charts · SHAP · Dashboard"),
        (5,  2.8, "User Dashboard",      "Web Interface Result Page"),
    ]

    data_stores = {
        10.4:  (6.8, "D1  Model File",       "models/churn_model.pkl"),
        10.4:  (5.2, "D2  Scaler",           "StandardScaler artefact"),
    }

    for x, y, label, sub in processes:
        _draw_box(ax, x, y, 6.8, 0.78, label, sub)

    for i in range(len(processes) - 1):
        _arrow(ax, processes[i][0], processes[i][1] - 0.4,
                   processes[i+1][0], processes[i+1][1] + 0.4)

    # Data store annotations
    for y, (x, ds_label, ds_sub) in zip([7.6, 9.2], [(8.5, "D1  Model", "churn_model.pkl"),
                                                       (8.5, "D2  Dataset", "telco_churn.csv")]):
        _draw_box(ax, 8.5, y, 2.6, 0.65, ds_label, ds_sub,
                  color="#0F2027", edge="#334155", fs=8)
        _arrow(ax, 6.9, y, 7.2, y, color="#334155")

    plt.tight_layout()
    path = os.path.join(DIAGRAMS_DIR, "dfd_level1.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Diagrams] dfd_level1.png saved.")
    return path


# ─── 4  ER Diagram ────────────────────────────────────────────────────────────

def generate_er_diagram():
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7, 8.6, "Entity Relationship Diagram",
            ha="center", fontsize=14, color=TEXT_COL, fontweight="bold")

    # Central entity
    entity = FancyBboxPatch((4.5, 3.5), 5, 4.5,
                              boxstyle="round,pad=0.1",
                              facecolor=BOX_FILL, edgecolor=BOX_EDGE, lw=2, zorder=3)
    ax.add_patch(entity)
    ax.text(7, 7.7, "CUSTOMER", ha="center", fontsize=13,
            color=BOX_EDGE, fontweight="bold", zorder=4)

    attrs = [
        ("🔑 customer_id",  "PK – Unique identifier"),
        ("👤 gender",       "Male / Female"),
        ("👴 SeniorCitizen","0 / 1"),
        ("📅 tenure",       "Months with company"),
        ("📋 contract_type","Month-to-month / One year / Two year"),
        ("💳 payment_method","Electronic / Mailed / Bank / Card"),
        ("💰 monthly_charges","USD per month"),
        ("💵 total_charges", "Cumulative USD"),
        ("⚠️  churn",       "Target: Yes / No"),
    ]
    y_start = 7.2
    for i, (attr, desc) in enumerate(attrs):
        y = y_start - 0.38 * (i + 1)
        ax.text(5.0, y, attr,  fontsize=8.5, color=TEXT_COL, zorder=4, fontweight="bold")
        ax.text(9.2, y, desc,  fontsize=7.5, color="#94A3B8", zorder=4, ha="right")
        ax.plot([4.6, 9.4], [y - 0.12, y - 0.12],
                color="#334155", lw=0.5, alpha=0.5, zorder=2)

    # Relationship diamond
    diamond_xs = [7, 7.6, 7, 6.4, 7]
    diamond_ys = [2.4, 3.0, 3.5, 3.0, 2.4]
    ax.fill(diamond_xs, diamond_ys, facecolor="#1B1F3A", edgecolor=ARROW_COL, lw=1.5, zorder=3)
    ax.text(7, 2.95, "PREDICTS", ha="center", fontsize=8,
            color=ARROW_COL, fontweight="bold", zorder=4)

    # Prediction entity
    pred = FancyBboxPatch((9.5, 2), 3.8, 1.8,
                           boxstyle="round,pad=0.1",
                           facecolor=BOX_FILL, edgecolor=ACC_COL, lw=1.5, zorder=3)
    ax.add_patch(pred)
    ax.text(11.4, 3.35, "PREDICTION",   ha="center", fontsize=11,
            color=ACC_COL, fontweight="bold", zorder=4)
    ax.text(11.4, 2.95, "churn_label",  ha="center", fontsize=8, color=TEXT_COL, zorder=4)
    ax.text(11.4, 2.60, "probability",  ha="center", fontsize=8, color=TEXT_COL, zorder=4)
    ax.text(11.4, 2.25, "risk_level",   ha="center", fontsize=8, color=TEXT_COL, zorder=4)

    # Connector lines
    ax.plot([7.6, 9.5], [3, 2.9], color=ARROW_COL, lw=1.5, zorder=2)
    ax.plot([7, 7], [3.5, 8.0],   color="#334155",  lw=1,   zorder=2, linestyle="--")

    plt.tight_layout()
    path = os.path.join(DIAGRAMS_DIR, "er_diagram.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[Diagrams] er_diagram.png saved.")
    return path


# ─── Master Generator ─────────────────────────────────────────────────────────

def generate_all_diagrams():
    generate_system_architecture()
    generate_dfd_level0()
    generate_dfd_level1()
    generate_er_diagram()
    print("[Diagrams] All diagrams generated.")


if __name__ == "__main__":
    generate_all_diagrams()
