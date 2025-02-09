"""
Feature Engineering Module
---------------------------
Creates derived / transformed features that improve model performance.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import NUMERIC_FEATURES


# ─── Derived Feature Builders ──────────────────────────────────────────────────

def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin *tenure* (months) into labelled cohorts.
    Groups: 0-12  → 'New'
            13-24 → 'Developing'
            25-48 → 'Established'
            49-60 → 'Loyal'
            61+   → 'Champion'
    """
    df = df.copy()
    bins   = [0, 12, 24, 48, 60, 72]
    labels = ["New", "Developing", "Established", "Loyal", "Champion"]
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, right=True
    )
    # OHE the new column
    df = pd.get_dummies(df, columns=["tenure_group"], drop_first=True)
    return df


def add_arpu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average Revenue Per User: TotalCharges / (tenure + 1)
    The +1 avoids division-by-zero for brand-new customers.
    """
    df = df.copy()
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["ARPU"] = df["TotalCharges"] / (df["tenure"] + 1)
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many add-on services a customer subscribes to.
    Uses raw column names (before OHE), so must be called on the cleaned
    but NOT yet encoded DataFrame.
    """
    df = df.copy()
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    existing = [c for c in service_cols if c in df.columns]

    def _is_yes(val) -> int:
        return 1 if str(val).strip().lower() == "yes" else 0

    df["service_count"] = df[existing].applymap(_is_yes).sum(axis=1)
    return df


def add_high_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag: monthly charges > 75 (high-spend indicator).
    """
    df = df.copy()
    if "MonthlyCharges" in df.columns:
        df["high_monthly"] = (df["MonthlyCharges"] > 75).astype(int)
    return df


# ─── Full Pipeline ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in order.
    Must be called BEFORE encoding (some steps use raw categorical columns).
    """
    df = add_service_count(df)   # uses raw columns
    df = add_tenure_group(df)     # bins → OHE internally
    df = add_arpu(df)
    df = add_high_risk_flags(df)
    print(f"[FeatureEngineering] After engineering: {df.shape}")
    return df
