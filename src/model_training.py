"""
Model Training Module
----------------------
Trains multiple ML classifiers, runs cross-validation, picks the best model,
and saves it to disk.

Usage (standalone):
    python src/model_training.py
"""

import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MODEL_PATH, TEST_SIZE, RANDOM_STATE, CV_FOLDS, MODEL_PARAMS
)
from src.data_loader        import load_data
from src.feature_engineering import engineer_features
from src.preprocessing      import clean_data, encode_categoricals, scale_features
from src.model_evaluation   import (
    evaluate_model, compare_models,
    plot_roc_curves, plot_confusion_matrix,
    plot_feature_importance, plot_model_comparison,
)


# ─── Model Registry ────────────────────────────────────────────────────────────

def _build_models() -> dict:
    """Instantiate all classifiers using config hyperparameters."""
    p = MODEL_PARAMS
    return {
        "Logistic Regression": LogisticRegression(**p["Logistic Regression"]),
        "Decision Tree":       DecisionTreeClassifier(**p["Decision Tree"]),
        "Random Forest":       RandomForestClassifier(**p["Random Forest"]),
        "SVM":                 SVC(**p["SVM"]),
        "Gradient Boosting":   GradientBoostingClassifier(**p["Gradient Boosting"]),
        "XGBoost":             XGBClassifier(**p["XGBoost"]),
    }


# ─── Full Training Pipeline ────────────────────────────────────────────────────

def run_training_pipeline():
    """
    End-to-end training:
      1. Load data
      2. Feature engineering (before encoding)
      3. Clean & encode
      4. Train/test split
      5. Scale numeric features
      6. Cross-validate all models
      7. Test-set evaluation
      8. Select & save best model
      9. Generate evaluation plots
    """

    # ── 1. Load ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" CUSTOMER CHURN PREDICTION – TRAINING PIPELINE")
    print("="*60)
    df = load_data()

    # ── 2. Feature Engineering ─────────────────────────────────────────────────
    df = clean_data(df)                # basic cleaning first
    # Feature engineering uses partially cleaned data (before OHE)
    # We add service_count before OHE; tenure_group also gets OHE inside
    from src.feature_engineering import (
        add_service_count, add_tenure_group,
        add_arpu, add_high_risk_flags
    )
    df = add_service_count(df)
    df = add_arpu(df)
    df = add_high_risk_flags(df)

    # ── 3. Encode ──────────────────────────────────────────────────────────────
    df = encode_categoricals(df)

    # Separate features / target
    from src.config import TARGET_COLUMN
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    feature_names = X.columns.tolist()
    print(f"\nFeature matrix: {X.shape}  |  Churn rate: {y.mean()*100:.1f} %")

    # ── 4. Train / Test Split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ── 5. Scale ───────────────────────────────────────────────────────────────
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # ── 6. Cross-Validation ────────────────────────────────────────────────────
    print("\n── Cross-Validation (ROC AUC) ──────────────────────────────")
    models   = _build_models()
    cv       = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        cv_scores[name] = scores.mean()
        print(f"  {name:<25}  CV AUC = {scores.mean():.4f}  ±  {scores.std():.4f}")

    # ── 7. Train All Models on Full Training Set ──────────────────────────────
    print("\n── Training on full training set …")
    trained_models = {}
    results        = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        print(f"  {name:<25}  Acc={metrics['Accuracy']:.4f}  "
              f"F1={metrics['F1 Score']:.4f}  AUC={metrics['ROC AUC']:.4f}")

    # ── 8. Select Best Model ───────────────────────────────────────────────────
    comparison_df = compare_models(results)
    print("\n── Model Comparison (sorted by ROC AUC) ─────────────────────")
    print(comparison_df.to_string(index=False))

    best_name  = comparison_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    print(f"\n✅ Best Model: {best_name}")

    # ── 9. Save Model + Metadata ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    save_payload = {
        "model":         best_model,
        "scaler":        scaler,
        "feature_names": feature_names,
        "model_name":    best_name,
        "metrics":       comparison_df.iloc[0].to_dict(),
        "all_results":   results,
    }
    joblib.dump(save_payload, MODEL_PATH)
    print(f"[Training] Model saved → {MODEL_PATH}")

    # ── 10. Generate Evaluation Plots ─────────────────────────────────────────
    print("\n── Generating evaluation plots …")
    plot_roc_curves(trained_models, X_test, y_test)
    plot_confusion_matrix(best_model, X_test, y_test, best_name)
    plot_feature_importance(best_model, feature_names, best_name)
    plot_model_comparison(results)
    print("\n[Training] All plots saved to static/images/")

    return save_payload


if __name__ == "__main__":
    run_training_pipeline()
    print("\n✅ Training pipeline complete!")
