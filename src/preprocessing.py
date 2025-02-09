"""
Data Preprocessing Module
--------------------------
Handles cleaning, type-casting, encoding, and scaling of the churn dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DROP_COLUMNS, TARGET_COLUMN, NUMERIC_FEATURES,
    CATEGORICAL_FEATURES, RANDOM_STATE,
)


# ─── Step 1: Basic Cleaning ────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform foundational cleaning steps:
      • Convert TotalCharges to numeric (coerces blanks to NaN)
      • Fill NaN TotalCharges with tenure × MonthlyCharges
      • Drop specified irrelevant columns
      • Convert binary Yes/No target to 0/1
    """
    df = df.copy()

    # Convert TotalCharges from string → numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Impute missing TotalCharges
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = (
        df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]
    )

    # Convert target to binary integer
    if df[TARGET_COLUMN].dtype == object:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    # Drop unneeded columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    print(f"[Preprocessing] After cleaning: {df.shape}  |  Missing: {df.isnull().sum().sum()}")
    return df


# ─── Step 2: Encode Categoricals ───────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-Hot Encode all categorical features.
    Drops the first level (dummy variable trap avoidance).
    """
    df = df.copy()
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"[Preprocessing] After encoding: {df.shape}")
    return df


# ─── Step 3: Scale Numerics ────────────────────────────────────────────────────

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: list | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on *X_train* and transform both splits.
    Returns scaled DataFrames and the fitted scaler (needed for inference).
    """
    if num_cols is None:
        num_cols = [c for c in NUMERIC_FEATURES if c in X_train.columns]

    scaler  = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"[Preprocessing] Scaled columns: {num_cols}")
    return X_train, X_test, scaler


# ─── Full Pipeline Wrapper ─────────────────────────────────────────────────────

def preprocess_pipeline(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run the full preprocessing pipeline:
      clean → encode → return (X, y)

    Note: Scaling is done AFTER the train/test split (inside model_training.py)
    to prevent data leakage.
    """
    df = clean_data(df)
    df = encode_categoricals(df)

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y
