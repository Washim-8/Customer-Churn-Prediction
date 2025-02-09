<div align="center">

# ⚡ ChurnIQ — Customer Churn Prediction System

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Inter&size=21&duration=3000&pause=800&color=0F766E&center=true&vCenter=true&width=720&lines=Predict+Customer+Churn+Before+It+Happens;End-to-End+ML+Pipeline+%2B+Flask+Dashboard;6+Models+Benchmarked+%E2%80%94+Best+One+Auto-Selected;Built+for+Real-World+Business+Impact" alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-2.0+-AC0404?style=for-the-badge" />
  <img src="https://img.shields.io/badge/SHAP-Explainability-5A67D8?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/Washim-8?style=social" />
  &nbsp;
  <img src="https://img.shields.io/github/followers/Washim-8?style=social" />
</p>

</div>

---

## 📌 What This Project Does

Businesses lose revenue every time a customer quietly walks away. The problem is that most companies only realise it *after* it happens. **ChurnIQ** solves this by giving you an early warning system.

Feed it customer data — tenure, contract type, billing info, subscribed services — and it tells you two things: **will this customer churn?** and **how confident are we?** It doesn't stop at a prediction either. The dashboard breaks down *why* a customer is at risk, so your team can take targeted action before it's too late.

This isn't a Jupyter notebook experiment. It's a complete system — ML pipeline, REST API, and a clean web interface — built to be used, not just demonstrated.

---

## ✨ Features

- 🔮 **Churn Probability Score** — Not just a Yes/No. Get a precise percentage with a risk tier (Low / Medium / High).
- 🤖 **AutoML Pipeline** — Trains 6 algorithms simultaneously (Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting, XGBoost) and auto-selects the best using ROC AUC.
- 🧩 **Smart Feature Engineering** — Automatically derives new signals like Average Revenue Per User (ARPU), service count, tenure cohorts, and high-risk flags from raw data.
- 🧠 **Explainable AI (SHAP)** — Every prediction is backed by feature-level reasoning, not just a black-box score.
- 📊 **Interactive Analytics Dashboard** — EDA charts, ROC curves, confusion matrix, feature importance — all auto-generated.
- 📐 **System Design Diagrams** — Includes auto-generated Architecture, DFD Level 0 & 1, and ER diagrams.
- 🌐 **Production-Ready Flask App** — Clean SaaS-style UI with a REST API endpoint for integration into external systems.
- 💾 **Synthetic Data Fallback** — No dataset? The system auto-generates a realistic synthetic dataset so you can run it out of the box.
- 🔌 **JSON REST API** — Expose predictions to any external app via `/api/predict`.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **ML Models** | Scikit-learn (LR, DT, RF, SVM, GB), XGBoost |
| **Explainability** | SHAP |
| **Data** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn, Plotly |
| **Backend** | Flask 3.0+ |
| **Frontend** | HTML5, Vanilla CSS3, Vanilla JavaScript |
| **Model Persistence** | joblib |
| **Environment** | Jupyter Notebook (EDA / experimentation) |

---

## 📂 Project Structure

```text
ChurnIQ/
│
├── app.py                        ← Flask web server — run this to start everything
│
├── src/                          ← Core ML engine
│   ├── config.py                 ← All paths, hyperparameters, constants in one place
│   ├── data_loader.py            ← Loads CSV or generates synthetic data if missing
│   ├── preprocessing.py          ← Cleaning, missing values, type casting
│   ├── feature_engineering.py    ← ARPU, service count, tenure groups, risk flags
│   ├── model_training.py         ← Full training pipeline: 6 models → best saved
│   ├── model_evaluation.py       ← Metrics, cross-val scores, all output charts
│   ├── explainability.py         ← SHAP-based feature importance
│   └── prediction.py             ← Real-time prediction from web form input
│
├── dashboard/
│   ├── visualizations.py         ← EDA chart generation scripts
│   └── diagrams_generator.py     ← Architecture, DFD, ER diagram generation
│
├── templates/                    ← Flask HTML templates
│   ├── index.html                ← Landing page
│   ├── predict.html              ← Customer data input form
│   ├── result.html               ← Churn result + probability gauge
│   ├── dashboard.html            ← Analytics gallery
│   ├── diagrams.html             ← System design diagrams
│   └── about_contact.html        ← Developer profile & contact
│
├── static/
│   ├── css/style.css             ← Premium light SaaS styling
│   ├── js/main.js                ← UI interactions & animations
│   └── images/                   ← Auto-generated charts land here
│
├── dataset/
│   └── telco_churn.csv           ← Input dataset (auto-generated if missing)
│
├── models/
│   └── churn_model.pkl           ← Trained model bundle (output of training step)
│
├── diagrams/                     ← Generated system design diagrams
├── notebooks/
│   └── churn_analysis.ipynb      ← Jupyter notebook for EDA experimentation
└── requirements.txt
```

---

## ⚙️ How It Works

Here's the full journey from raw data to a prediction on screen:

```
Raw CSV Data
     │
     ▼
Data Loader → Preprocessing → Feature Engineering
                                      │
                                      ▼
                              Encode + Scale
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  Train 6 Models (Cross-Validation)  │
                    │  LR · DT · RF · SVM · GB · XGBoost  │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                           Best Model (ROC AUC) → Saved as .pkl
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                   Web UI Form              /api/predict
                         │
                         ▼
               Churn Probability + Risk Level + SHAP Insights
```

1. **Data Ingestion** — Loads `telco_churn.csv` or generates a synthetic equivalent.
2. **Cleaning** — Handles TotalCharges whitespace issues, type casting, missing value imputation.
3. **Feature Engineering** — Adds ARPU (`MonthlyCharges / max(tenure,1)`), service count, tenure cohort labels, and high-risk contract flags before encoding.
4. **Encoding + Scaling** — One-hot encodes categoricals, StandardScaler on numerics.
5. **Training** — 5-fold stratified cross-validation on all 6 classifiers. Best ROC AUC wins.
6. **Saving** — Model, scaler, and feature names bundled together in a single `.pkl` file.
7. **Prediction** — Web form input is preprocessed through the same pipeline and scored live.
8. **Output** — Churn label, probability %, risk tier, and recommended action.

---

## ▶️ Installation & Setup

**Prerequisites:** Python 3.10+, pip

```bash
# 1. Clone the repository
git clone https://github.com/Washim-8/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Step 1 — Train the Models

```bash
python src/model_training.py
```

This runs the full training pipeline:
- Evaluates 6 ML algorithms with cross-validation
- Saves the best model to `models/churn_model.pkl`
- Generates all evaluation charts in `static/images/`

> **No dataset?** No problem. The system auto-generates a realistic synthetic dataset if `dataset/telco_churn.csv` is not found.

### Step 2 — Generate System Diagrams (Optional)

```bash
python dashboard/diagrams_generator.py
```

Generates Architecture, DFD Level 0 & 1, and ER diagrams in `diagrams/`.

### Step 3 — Launch the Web App

```bash
python app.py
```

Open your browser: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🖥️ Using the Application

### 🔮 Predict Churn
Navigate to `/predict` → Fill in customer details across three sections:
- **Demographics** — Gender, Senior Citizen, Partner, Dependents, Tenure
- **Services** — Phone, Internet, Security, Streaming, Tech Support
- **Billing** — Contract type, Payment method, Monthly & Total charges

Hit **Predict Churn** → The result page shows:
- ✅ or ❌ Churn decision
- Probability gauge (0–100%)
- Risk badge: 🟢 Low / 🟡 Medium / 🔴 High
- Business action recommendation

### 📊 Dashboard
Analytics gallery with auto-generated charts — churn distribution, correlation heatmap, model comparison, ROC curves, confusion matrix, feature importances.

### 📐 Diagrams
System design gallery — software architecture, data flow diagrams (DFD L0 & L1), entity relationship diagram.

### 👤 About & Contact
Developer profile and contact section at `/about`.

---

## 🔌 REST API

Integrate predictions into any system without the UI:

**Endpoint:** `POST /api/predict`

**Request:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.50,
  "TotalCharges": 1026.00
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "YES",
  "probability": 72.5,
  "model_name": "Gradient Boosting",
  "risk_level": "High"
}
```

---

## 📸 Screenshots

> Add screenshots of the following pages to show the full experience:

| Page | What to Capture |
|---|---|
| `index.html` | Hero section with accuracy metrics pills |
| `predict.html` | 3-section customer data form |
| `result.html` | Churn result with probability gauge + risk badge |
| `dashboard.html` | Analytics charts grid |
| `diagrams.html` | System design diagrams |
| `about_contact.html` | About + Contact cards |

---

## 🎥 Demo GIF Ideas

Create short screen recordings for maximum impact on GitHub:

| GIF | How to Record |
|---|---|
| **Form → Prediction flow** | Fill the form, submit, show the result page with gauge animation |
| **Dashboard scroll** | Slow scroll through the analytics dashboard |
| **Model training console** | Terminal output during `python src/model_training.py` |
| **API call in Postman** | POST request → JSON response |

**Recommended tool:** [ScreenToGif](https://www.screentogif.com/) (Windows) or OBS Studio

---

## 🚀 Future Improvements

- [ ] **Cloud deployment** — Host on AWS (EC2 + S3) or Render with CI/CD
- [ ] **Real-time database** — Replace CSV with PostgreSQL / SQLite for live customer records
- [ ] **User authentication** — Multi-user support with role-based access (admin / analyst)
- [ ] **Deep learning model** — Add a neural network option (TensorFlow / PyTorch) to the benchmark
- [ ] **Batch predictions** — Upload a CSV and predict churn for 1000 customers at once
- [ ] **Email alerts** — Trigger automated alerts when high-risk customers are identified
- [ ] **SHAP waterfall charts** — Per-prediction explanations shown directly on the result page

---

## 👨‍💻 About the Developer

I'm **Washim Shaikh**, a Computer Science Engineering student who builds things that actually solve problems. My work sits at the intersection of machine learning and software engineering — I care about the full picture, from how data flows through a pipeline to how results are communicated in a clean interface.

Projects like **AgriTrade** (an e-auction platform removing middlemen between farmers and buyers), an **AI chatbot system**, and a **fraud detection engine** for Indian transaction patterns have shaped how I think about building. Each one started with a real problem and ended with a working system.

Currently deepening my experience in cloud infrastructure and production-grade AI through internships at **Coincent** (AI/Python), **Yhills** (ML), **1Stop** (Full Stack), and **iStudio** (AWS — ongoing). The goal is always the same: ship things that work and create actual value.

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=Washim-8&show_icons=true&theme=default&title_color=0F766E&icon_color=0F766E&border_color=E5E7EB&bg_color=F8FAF9" alt="GitHub Stats" />
  &nbsp;&nbsp;
  <img src="https://github-readme-streak-stats.herokuapp.com/?user=Washim-8&theme=default&ring=0F766E&fire=0D9488&currStreakLabel=0F766E&border=E5E7EB&background=F8FAF9" alt="GitHub Streak" />
</p>

---

## 📬 Contact

Got a project idea, a collaboration in mind, or just want to connect? Reach out — I'm always open to interesting conversations.

| Channel | Link |
|---|---|
| 📧 **Email** | [washimshaikh33@gmail.com](mailto:washimshaikh33@gmail.com) |
| 📱 **Phone** | +91 8884958185 |
| 💻 **GitHub** | [github.com/Washim-8](https://github.com/Washim-8) |
| 🔗 **LinkedIn** | [Washim Shaikh](https://www.linkedin.com/in/washim-shaikh-349868281/) |

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

<div align="center">

*Built with a focus on real-world impact, clean engineering, and scalable AI systems.*

⭐ If this project helped you, give it a star — it genuinely helps.

</div>
