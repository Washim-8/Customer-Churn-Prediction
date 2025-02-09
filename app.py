"""
app.py – Flask Web Application
================================
Customer Churn Prediction System
Runs on: http://127.0.0.1:5000
"""

import os
import sys
import json
import traceback
from flask import Flask, render_template, request, jsonify, redirect, url_for

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.prediction import predict_churn, load_model_payload

app = Flask(__name__)
app.secret_key = "churn_prediction_2024"

# ─── Lazy model load on startup ───────────────────────────────────────────────

MODEL_READY = False

def _ensure_model():
    """Check that the trained model exists; return True/False."""
    from src.config import MODEL_PATH
    return os.path.exists(MODEL_PATH)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Home / landing page."""
    model_ready = _ensure_model()

    # Load evaluation results if available
    metrics = None
    if model_ready:
        try:
            payload = load_model_payload()
            metrics = {
                "model_name": payload["model_name"],
                "accuracy":   round(payload["metrics"].get("Accuracy", 0) * 100, 1),
                "roc_auc":    round((payload["metrics"].get("ROC AUC") or 0) * 100, 1),
                "f1":         round(payload["metrics"].get("F1 Score", 0) * 100, 1),
            }
            # Build comparison table
            all_results = payload.get("all_results", [])
            comparison  = sorted(all_results,
                                 key=lambda r: r.get("ROC AUC") or 0,
                                 reverse=True)
        except Exception:
            comparison = []
    else:
        comparison = []

    return render_template(
        "index.html",
        model_ready=model_ready,
        metrics=metrics,
        comparison=comparison,
    )


@app.route("/predict", methods=["GET"])
def predict_page():
    """Prediction form page."""
    model_ready = _ensure_model()
    return render_template("predict.html", model_ready=model_ready)


@app.route("/predict", methods=["POST"])
def predict_submit():
    """Handle form submission and return prediction result."""
    if not _ensure_model():
        return render_template(
            "result.html",
            error="Model not trained yet. Please run 'python src/model_training.py' first.",
        )

    try:
        form = request.form
        user_data = {
            "gender":          form.get("gender", "Male"),
            "SeniorCitizen":   int(form.get("SeniorCitizen", 0)),
            "Partner":         form.get("Partner", "No"),
            "Dependents":      form.get("Dependents", "No"),
            "tenure":          float(form.get("tenure", 1)),
            "PhoneService":    form.get("PhoneService", "No"),
            "MultipleLines":   form.get("MultipleLines", "No"),
            "InternetService": form.get("InternetService", "No"),
            "OnlineSecurity":  form.get("OnlineSecurity", "No"),
            "OnlineBackup":    form.get("OnlineBackup", "No"),
            "DeviceProtection":form.get("DeviceProtection", "No"),
            "TechSupport":     form.get("TechSupport", "No"),
            "StreamingTV":     form.get("StreamingTV", "No"),
            "StreamingMovies": form.get("StreamingMovies", "No"),
            "Contract":        form.get("Contract", "Month-to-month"),
            "PaperlessBilling":form.get("PaperlessBilling", "No"),
            "PaymentMethod":   form.get("PaymentMethod", "Electronic check"),
            "MonthlyCharges":  float(form.get("MonthlyCharges", 0)),
            "TotalCharges":    float(form.get("TotalCharges", 0)),
        }

        result = predict_churn(user_data)
        return render_template("result.html", result=result, user_data=user_data)

    except Exception as exc:
        traceback.print_exc()
        return render_template("result.html", error=str(exc))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for programmatic access."""
    if not _ensure_model():
        return jsonify({"error": "Model not trained"}), 503

    try:
        data   = request.get_json(force=True)
        result = predict_churn(data)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/dashboard")
def dashboard():
    """EDA & visualisation dashboard."""
    images = {
        "churn_distribution":  "images/churn_distribution.png",
        "correlation_heatmap": "images/correlation_heatmap.png",
        "churn_by_contract":   "images/churn_by_contract.png",
        "churn_by_tenure":     "images/churn_by_tenure.png",
        "churn_by_payment":    "images/churn_by_payment.png",
        "monthly_charges":     "images/monthly_charges_dist.png",
        "roc_curve":           "images/roc_curve.png",
        "confusion_matrix":    "images/confusion_matrix.png",
        "feature_importance":  "images/feature_importance.png",
        "model_comparison":    "images/model_comparison.png",
    }
    # Filter to existing images
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    available  = {k: v for k, v in images.items()
                  if os.path.exists(os.path.join(static_dir, v))}

    return render_template("dashboard.html", images=available)


@app.route("/about")
def about():
    """About & Contact page."""
    return render_template("about_contact.html")


@app.route("/diagrams")
def diagrams():
    """System design diagrams page."""
    diag_images = {
        "System Architecture": "images/system_architecture.png",
        "DFD Level 0":         "images/dfd_level0.png",
        "DFD Level 1":         "images/dfd_level1.png",
        "ER Diagram":          "images/er_diagram.png",
    }
    # Copy diagrams to static/images if not present
    import shutil
    diagrams_dir = os.path.join(os.path.dirname(__file__), "diagrams")
    images_dir   = os.path.join(os.path.dirname(__file__), "static", "images")
    for fname in ["system_architecture.png", "dfd_level0.png",
                  "dfd_level1.png", "er_diagram.png"]:
        src = os.path.join(diagrams_dir, fname)
        dst = os.path.join(images_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    available  = {k: v for k, v in diag_images.items()
                  if os.path.exists(os.path.join(static_dir, v))}

    return render_template("diagrams.html", images=available)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Customer Churn Prediction System")
    print("  http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    if not _ensure_model():
        print("⚠️  Model not found. Run: python src/model_training.py")
    app.run(debug=True, port=5000)
