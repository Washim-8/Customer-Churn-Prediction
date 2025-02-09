"""
Data Loader Module
------------------
Handles loading the Telco Churn dataset. If the CSV is not present,
it synthesizes a realistic dataset so the project runs out-of-the-box.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATASET_PATH, RANDOM_STATE


# ─── Synthetic Dataset Generator ───────────────────────────────────────────────

def _generate_synthetic_dataset(n: int = 7043, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Creates a realistic synthetic Telco-style dataset when the CSV is absent.
    Distributions are calibrated to mimic the real Telco dataset.
    """
    rng = np.random.default_rng(seed)

    n_churn     = int(n * 0.265)   # ~26.5 % churn rate (mirrors real dataset)
    n_no_churn  = n - n_churn
    labels      = [1] * n_churn + [0] * n_no_churn
    rng.shuffle(labels)
    churn = np.array(labels)

    # Tenure: churners leave earlier
    tenure = np.where(
        churn == 1,
        rng.integers(1, 30, size=n),
        rng.integers(1, 72, size=n),
    )

    # Monthly charges
    monthly = np.where(
        churn == 1,
        rng.uniform(55, 100, size=n),
        rng.uniform(20, 80,  size=n),
    ).round(2)

    # Total charges derived from tenure × monthly (with noise)
    total = (tenure * monthly * rng.uniform(0.9, 1.1, size=n)).round(2)

    choices = lambda opts, p=None: rng.choice(opts, size=n, p=p)

    data = pd.DataFrame({
        "customerID":      [f"CUST-{i:05d}" for i in range(n)],
        "gender":          choices(["Male", "Female"]),
        "SeniorCitizen":   choices([0, 1], p=[0.84, 0.16]),
        "Partner":         choices(["Yes", "No"]),
        "Dependents":      choices(["Yes", "No"], p=[0.30, 0.70]),
        "tenure":          tenure,
        "PhoneService":    choices(["Yes", "No"], p=[0.90, 0.10]),
        "MultipleLines":   choices(["No phone service", "No", "Yes"], p=[0.10, 0.42, 0.48]),
        "InternetService": choices(["DSL", "Fiber optic", "No"], p=[0.34, 0.44, 0.22]),
        "OnlineSecurity":  choices(["Yes", "No", "No internet service"], p=[0.29, 0.50, 0.21]),
        "OnlineBackup":    choices(["Yes", "No", "No internet service"], p=[0.34, 0.44, 0.22]),
        "DeviceProtection":choices(["Yes", "No", "No internet service"], p=[0.34, 0.44, 0.22]),
        "TechSupport":     choices(["Yes", "No", "No internet service"], p=[0.29, 0.49, 0.22]),
        "StreamingTV":     choices(["Yes", "No", "No internet service"], p=[0.38, 0.40, 0.22]),
        "StreamingMovies": choices(["Yes", "No", "No internet service"], p=[0.38, 0.40, 0.22]),
        "Contract":        choices(["Month-to-month", "One year", "Two year"],
                                   p=[0.55, 0.21, 0.24]),
        "PaperlessBilling":choices(["Yes", "No"], p=[0.59, 0.41]),
        "PaymentMethod":   choices(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
            p=[0.34, 0.23, 0.22, 0.21]
        ),
        "MonthlyCharges":  monthly,
        "TotalCharges":    total,
        "Churn":           np.where(churn == 1, "Yes", "No"),
    })

    return data


# ─── Main Loader ───────────────────────────────────────────────────────────────

def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the Telco Churn dataset from *path*.

    If the file is missing, a synthetic dataset is generated and saved so
    subsequent runs are deterministic.

    Returns
    -------
    pd.DataFrame
        Raw (unprocessed) dataset.
    """
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"[DataLoader] Loaded dataset → {path}  {df.shape}")
            return df
        except Exception as exc:
            print(f"[DataLoader] Failed to read CSV ({exc}). Generating synthetic data.")

    print("[DataLoader] Dataset not found – generating realistic synthetic dataset …")
    df = _generate_synthetic_dataset()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[DataLoader] Synthetic dataset saved → {path}  {df.shape}")
    return df


def get_basic_info(df: pd.DataFrame) -> dict:
    """Return a summary dictionary with basic dataset statistics."""
    return {
        "rows":        len(df),
        "columns":     len(df.columns),
        "missing":     df.isnull().sum().sum(),
        "churn_count": (df["Churn"] == "Yes").sum() if "Churn" in df else None,
        "churn_rate":  f"{(df['Churn'] == 'Yes').mean() * 100:.1f} %" if "Churn" in df else None,
        "dtypes":      df.dtypes.value_counts().to_dict(),
    }


if __name__ == "__main__":
    df   = load_data()
    info = get_basic_info(df)
    print("\nDataset Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(df.head())
