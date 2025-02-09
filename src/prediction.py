"""
Prediction Module
-----------------
Loads the saved model artefact and exposes a clean predict() interface
used by both the Flask app and standalone scripts.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_PATH


# ─── Loader (cached) ───────────────────────────────────────────────────────────

_cache: dict | None = None


def load_model_payload(path: str = MODEL_PATH) -> dict:
    """
    Load model payload from disk. Uses a module-level cache so the
    model is only deserialised once per process.
    """
    global _cache
    if _cache is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found at '{path}'. "
                "Run 'python src/model_training.py' first."
            )
        _cache = joblib.load(path)
        print(f"[Prediction] Model loaded: {_cache['model_name']}")
    return _cache


# ─── Input Builder ─────────────────────────────────────────────────────────────

def build_input_df(user_data: dict, feature_names: list[str]) -> pd.DataFrame:
    """
    Transform raw user input (from web form or dict) into a DataFrame
    that matches the training feature matrix.

    Steps
    -----
    1. Create a zero-row template with all training features.
    2. Fill in values from user_data after one-hot encoding conventions.
    3. Fill remaining NaN with 0 (absence of categorical level).
    """
    row = {feat: 0.0 for feat in feature_names}

    numeric_map = {
        "tenure":         "tenure",
        "MonthlyCharges": "MonthlyCharges",
        "TotalCharges":   "TotalCharges",
    }
    for src, dst in numeric_map.items():
        if src in user_data and dst in row:
            row[dst] = float(user_data[src])

    # Derived features
    tenure  = float(user_data.get("tenure", 0))
    monthly = float(user_data.get("MonthlyCharges", 0))
    total   = float(user_data.get("TotalCharges", monthly * tenure))

    if "ARPU" in row:
        row["ARPU"] = total / (tenure + 1)
    if "service_count" in row:
        svc_cols = ["PhoneService", "MultipleLines", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
        row["service_count"] = sum(
            1 for c in svc_cols if user_data.get(c, "No") == "Yes"
        )
    if "high_monthly" in row:
        row["high_monthly"] = 1 if monthly > 75 else 0

    # One-hot categorical columns
    def _ohe(col_base: str, value: str):
        """Set the OHE column for *value* of *col_base* if it exists in features."""
        ohe_key = f"{col_base}_{value}"
        if ohe_key in row:
            row[ohe_key] = 1

    _ohe("gender",          user_data.get("gender", "Male"))
    _ohe("Partner",         user_data.get("Partner", "No"))
    _ohe("Dependents",      user_data.get("Dependents", "No"))
    _ohe("PhoneService",    user_data.get("PhoneService", "No"))
    _ohe("MultipleLines",   user_data.get("MultipleLines", "No"))
    _ohe("InternetService", user_data.get("InternetService", "No"))
    _ohe("OnlineSecurity",  user_data.get("OnlineSecurity", "No"))
    _ohe("OnlineBackup",    user_data.get("OnlineBackup", "No"))
    _ohe("DeviceProtection",user_data.get("DeviceProtection","No"))
    _ohe("TechSupport",     user_data.get("TechSupport", "No"))
    _ohe("StreamingTV",     user_data.get("StreamingTV", "No"))
    _ohe("StreamingMovies", user_data.get("StreamingMovies","No"))
    _ohe("Contract",        user_data.get("Contract", "Month-to-month"))
    _ohe("PaperlessBilling",user_data.get("PaperlessBilling","No"))
    _ohe("PaymentMethod",   user_data.get("PaymentMethod", "Electronic check"))

    # SeniorCitizen is numeric (0/1)
    if "SeniorCitizen" in row:
        row["SeniorCitizen"] = int(user_data.get("SeniorCitizen", 0))

    df = pd.DataFrame([row], columns=feature_names)
    return df


# ─── Main Predict Function ─────────────────────────────────────────────────────

def predict_churn(user_data: dict) -> dict:
    """
    Predict churn for a single customer.

    Parameters
    ----------
    user_data : dict
        Raw field values from the web form.

    Returns
    -------
    dict
        {
          "prediction":   0 or 1,
          "label":        "YES" or "NO",
          "probability":  float (0-100),
          "model_name":   str,
          "risk_level":   "Low" | "Medium" | "High",
        }
    """
    payload       = load_model_payload()
    model         = payload["model"]
    scaler        = payload["scaler"]
    feature_names = payload["feature_names"]

    # Build aligned input DataFrame
    X = build_input_df(user_data, feature_names)

    # Scale numeric columns (same scaler used during training)
    from src.config import NUMERIC_FEATURES
    num_cols = [c for c in NUMERIC_FEATURES if c in X.columns]
    X[num_cols] = scaler.transform(X[num_cols])

    # Predict
    pred   = int(model.predict(X)[0])
    proba  = float(model.predict_proba(X)[0][1]) * 100  # % churn probability

    # Risk level
    if proba < 30:
        risk = "Low"
    elif proba < 65:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "prediction":  pred,
        "label":       "YES" if pred == 1 else "NO",
        "probability": round(proba, 1),
        "model_name":  payload["model_name"],
        "risk_level":  risk,
    }
