"""
app.py – Flask Web Application
================================
Customer Churn Prediction System
Runs on: http://127.0.0.1:5000
"""

import os
import sys
import json
import threading
import time
import traceback
import urllib.request
from functools import lru_cache
from flask import Flask, render_template, request, jsonify, redirect, url_for, make_response
from flask_compress import Compress

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.prediction import predict_churn, load_model_payload

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "churn_prediction_2024")

# ─── Gzip / Brotli compression for all responses ──────────────────────────────
app.config["COMPRESS_REGISTER"] = True
app.config["COMPRESS_LEVEL"] = 6          # balanced speed vs ratio
app.config["COMPRESS_MIN_SIZE"] = 500     # don't compress tiny responses
Compress(app)

# ─── Model warm-up on startup (once, in main process) ─────────────────────────
# Resolves MODEL_PATH once at import time — avoids repeated os.path.exists() calls.
from src.config import MODEL_PATH

_MODEL_EXISTS: bool = os.path.exists(MODEL_PATH)
_MODEL_CACHE: dict | None = None          # populated on first prediction request


def _ensure_model() -> bool:
    """Check model existence using the module-level cached flag."""
    return _MODEL_EXISTS


def _get_model_payload() -> dict | None:
    """Return the model payload, loading once and caching for the process lifetime."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None and _MODEL_EXISTS:
        try:
            _MODEL_CACHE = load_model_payload()
        except Exception as exc:
            print(f"[Startup] Failed to load model: {exc}")
    return _MODEL_CACHE


# Pre-load model payload at import time so the first user request is instant.
if _MODEL_EXISTS:
    _get_model_payload()


# ─── Dashboard / Diagrams image lists cached at startup ───────────────────────
@lru_cache(maxsize=1)
def _dashboard_images() -> dict:
    """Resolve available dashboard images once; cached for process lifetime."""
    candidates = {
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
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    return {k: v for k, v in candidates.items()
            if os.path.exists(os.path.join(static_dir, v))}


@lru_cache(maxsize=1)
def _diagram_images() -> dict:
    """Copy diagram PNGs to static/images (once) and return available map."""
    import shutil
    candidates = {
        "System Architecture": "images/system_architecture.png",
        "DFD Level 0":         "images/dfd_level0.png",
        "DFD Level 1":         "images/dfd_level1.png",
        "ER Diagram":          "images/er_diagram.png",
    }
    fnames = {
        "System Architecture": "system_architecture.png",
        "DFD Level 0":         "dfd_level0.png",
        "DFD Level 1":         "dfd_level1.png",
        "ER Diagram":          "er_diagram.png",
    }
    diagrams_dir = os.path.join(os.path.dirname(__file__), "diagrams")
    images_dir   = os.path.join(os.path.dirname(__file__), "static", "images")
    for label, fname in fnames.items():
        src = os.path.join(diagrams_dir, fname)
        dst = os.path.join(images_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    return {k: v for k, v in candidates.items()
            if os.path.exists(os.path.join(static_dir, v))}


# ─── Cache-Control helper ──────────────────────────────────────────────────────
def _cached(response, seconds: int = 3600):
    """Attach Cache-Control header to a Flask response."""
    response.cache_control.max_age = seconds
    response.cache_control.public  = True
    return response


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
@app.route("/healthz", methods=["GET"])
def health():
    """Health check endpoint — handles both /health and /healthz (Render monitor)."""
    return jsonify({"status": "healthy"}), 200


# ─── Keep-Alive Self-Ping (prevents Render free-tier sleep) ───────────────────

def _keep_alive():
    """
    Daemon thread that pings this service's own /health endpoint every
    10 minutes so Render's free-tier never idles the instance to sleep.

    The public URL is read from the RENDER_EXTERNAL_URL environment variable
    which Render injects automatically into every running service.
    Only starts pinging once the server is ready (waits 30 s on first run).
    """
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not render_url:
        # Not running on Render (local dev) – skip silently.
        return

    ping_url = f"{render_url}/health"
    print(f"[KeepAlive] Self-ping enabled → {ping_url} every 10 min")

    # Wait 30 s on cold start to let Gunicorn fully initialise first.
    time.sleep(30)

    while True:
        try:
            with urllib.request.urlopen(ping_url, timeout=10) as resp:
                print(f"[KeepAlive] Pinged {ping_url} → HTTP {resp.status}")
        except Exception as exc:
            print(f"[KeepAlive] Ping failed: {exc}")
        time.sleep(600)  # 10 minutes


# Start keep-alive thread at import time so it works under Gunicorn too.
_t = threading.Thread(target=_keep_alive, name="keep-alive", daemon=True)
_t.start()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Home / landing page."""
    model_ready = _ensure_model()

    metrics    = None
    comparison = []
    if model_ready:
        try:
            payload    = _get_model_payload()
            metrics    = {
                "model_name": payload["model_name"],
                "accuracy":   round(payload["metrics"].get("Accuracy", 0) * 100, 1),
                "roc_auc":    round((payload["metrics"].get("ROC AUC") or 0) * 100, 1),
                "f1":         round(payload["metrics"].get("F1 Score", 0) * 100, 1),
            }
            all_results = payload.get("all_results", [])
            comparison  = sorted(all_results,
                                 key=lambda r: r.get("ROC AUC") or 0,
                                 reverse=True)
        except Exception:
            comparison = []

    resp = make_response(render_template(
        "index.html",
        model_ready=model_ready,
        metrics=metrics,
        comparison=comparison,
    ))
    # Home page can be cached for 5 minutes (metrics don't change between deploys)
    return _cached(resp, seconds=300)


@app.route("/predict", methods=["GET"])
def predict_page():
    """Prediction form page."""
    model_ready = _ensure_model()
    resp = make_response(render_template("predict.html", model_ready=model_ready))
    return _cached(resp, seconds=600)


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
    resp = make_response(render_template("dashboard.html", images=_dashboard_images()))
    # Dashboard images are static between deployments — cache for 1 hour
    return _cached(resp, seconds=3600)


@app.route("/about")
def about():
    """About & Contact page."""
    resp = make_response(render_template("about_contact.html"))
    return _cached(resp, seconds=3600)


@app.route("/diagrams")
def diagrams():
    """System design diagrams page."""
    resp = make_response(render_template("diagrams.html", images=_diagram_images()))
    return _cached(resp, seconds=3600)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    print("\n" + "=" * 55)
    print("  Customer Churn Prediction System")
    print(f"  Running on: http://0.0.0.0:{port}")
    print("=" * 55 + "\n")
    if not _ensure_model():
        print("⚠️  Model not found. Run: python src/model_training.py")
    app.run(host="0.0.0.0", port=port, debug=debug)
