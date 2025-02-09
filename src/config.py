"""
Configuration settings for the Customer Churn Prediction System.
Centralizes all paths, hyperparameters, and constants.
"""

import os

# ─── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "telco_churn.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "models", "churn_model.pkl")
IMAGES_DIR   = os.path.join(BASE_DIR, "static", "images")
DIAGRAMS_DIR = os.path.join(BASE_DIR, "diagrams")

# ─── Target Column ─────────────────────────────────────────────────────────────
TARGET_COLUMN = "Churn"

# ─── Columns to Drop ───────────────────────────────────────────────────────────
DROP_COLUMNS = ["customerID"]

# ─── Numeric Features ──────────────────────────────────────────────────────────
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# ─── Categorical Features ──────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

# ─── Train/Test Split ──────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ─── Cross-Validation ──────────────────────────────────────────────────────────
CV_FOLDS = 5

# ─── Model Hyperparameters ─────────────────────────────────────────────────────
MODEL_PARAMS = {
    "Logistic Regression": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    },
    "Decision Tree": {
        "max_depth": 6,
        "min_samples_split": 20,
        "random_state": RANDOM_STATE,
    },
    "Random Forest": {
        "n_estimators": 200,
        "max_depth": 8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "SVM": {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,
        "random_state": RANDOM_STATE,
    },
    "Gradient Boosting": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "random_state": RANDOM_STATE,
    },
    "XGBoost": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
    },
}

# ─── Visualization ─────────────────────────────────────────────────────────────
FIGURE_DPI  = 150
FIGURE_SIZE = (10, 6)
COLOR_CHURN = "#DC2626"
COLOR_NO_CHURN = "#0F766E"
PALETTE = "Set2"
