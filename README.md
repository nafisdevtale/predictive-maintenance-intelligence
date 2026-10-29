# Predictive Maintenance Intelligence System

An end-to-end, college-level Data Science and Machine Learning project for industrial machine failure prediction, statistical condition monitoring, and live sensor simulation.

> **EDUCATIONAL & ETHICAL DISCLAIMER**  
> This system is built for educational, academic, and research demonstration purposes utilizing the benchmark UCI AI4I 2020 Predictive Maintenance dataset. Model outputs, simulated sensor streams, and failure predictions are educational statistical estimations and must **not** be used for safety-critical diagnostics, operational shutdown controls, or life-critical machinery environments.

---

## 📌 Project Overview
Industrial downtime costs manufacturing facilities thousands of dollars per hour. Predictive Maintenance (PdM) leverages machine sensor telemetries to anticipate equipment breakdown before failure occurs.

This project implements:
1. **Strict Data Leakage Prevention:** Total exclusion of identification tags (`UDI`, `Product ID`) and failure-cause indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) from the predictor feature space when forecasting `Machine failure`.
2. **Reproducible Preprocessing & Modeling:** Encapsulation of scalers, one-hot encoders, and estimators in unified scikit-learn `Pipeline` artifacts, fitted strictly on training data (stratified 80/20 split).
3. **Imbalance-Aware Learning:** Baseline Logistic Regression and tuned Random Forest models calibrated using balanced class weights, evaluated through Precision-Recall AUC (PR-AUC), ROC-AUC, Precision, Recall, and Confusion Matrices.
4. **Statistical Rigor with SciPy:** Two-sample Mann-Whitney U tests with Benjamini-Hochberg False Discovery Rate (FDR) corrections, and Chi-square contingency tests evaluating operational differences across failure states.
5. **Modern Decoupled Web Application:** High-performance Next.js 14 (App Router, TypeScript, Tailwind CSS, Recharts) frontend coupled with a stateless FastAPI Python prediction backend designed for local development and dual-project Vercel serverless deployment.

---

## 🏗️ Architecture & Directory Structure
```text
predictive-maintenance-intelligence/
├── app.py                         # Root FastAPI application entrypoint (Local & Vercel Backend)
├── api/
│   └── index.py                   # Vercel Serverless Function entrypoint (imports app)
├── vercel.json                    # Vercel serverless Python rewrite configuration
├── .vercelignore                  # Vercel bundle optimization (excludes frontend, tests, raw data)
├── .python-version                # Explicit Python 3.12 runtime instruction for Vercel
├── .env.example                   # Backend environment configuration template (ALLOWED_ORIGINS)
├── requirements.txt               # Lean production dependencies (FastAPI, scikit-learn, etc.)
├── requirements-dev.txt           # Local development & testing dependencies (pytest, matplotlib, etc.)
├── .gitignore                     # Git rules (tracks deployable models & outputs; excludes caches)
├── README.md                      # Comprehensive project documentation
├── data/
│   ├── README.md                  # Dataset attribution, DOI citation & schema details
│   └── ai4i2020.csv               # Raw dataset
├── models/
│   ├── logistic_regression_pipeline.joblib  # Fitted baseline pipeline artifact (3.7 KB)
│   ├── random_forest_pipeline.joblib        # Fitted production pipeline artifact (2.5 MB)
│   └── metadata.json                        # Model threshold and deployment metadata
├── outputs/
│   ├── model_evaluation_metrics.json        # Machine-readable out-of-fold & test metrics (44 KB)
│   ├── statistical_analysis_report.json     # SciPy hypothesis test results & interpretations (5.6 KB)
│   ├── statistical_tests_numeric.csv        # Non-parametric tests & FDR adjustments (1.0 KB)
│   ├── contingency_table_type.csv           # Chi-Square contingency counts (67 B)
│   └── figures/                             # Publication-grade Matplotlib plots
├── src/
│   ├── __init__.py
│   ├── config.py                  # Core configuration, paths, and leakage boundaries
│   ├── data/                      # Data downloader & schema validation loader
│   ├── features/                  # FeatureEngineer (Temp diff, Power) & Stratified split
│   ├── models/                    # Pipeline factories, cross-validation, & ProductionPredictor
│   ├── stats/                     # Mann-Whitney U & Chi-Square testing engine
│   └── simulator/                 # Stateless sensor stream mean-reverting stochastic engine
├── tests/                         # 55 automated offline unit & API test suites
│   ├── test_data.py               # Schema & integrity tests (10 tests)
│   ├── test_features.py           # Feature engineering & transformation tests (8 tests)
│   ├── test_statistics.py         # SciPy statistical testing & FDR tests (6 tests)
│   ├── test_models.py             # Model fitting, scoring & serialization tests (8 tests)
│   ├── test_simulator.py          # Deterministic sensor generator tests (7 tests)
│   ├── test_predictor.py          # ProductionPredictor inference tests (7 tests)
│   └── test_api.py                # FastAPI HTTP & Vercel entrypoint tests (9 tests)
└── frontend/                      # Next.js 14 App Router Web Application
    ├── package.json               # Frontend dependencies (React 18, Tailwind, Recharts)
    ├── package-lock.json          # Deterministic lockfile for reproducible npm builds
    ├── tsconfig.json              # Strict TypeScript configuration
    ├── tailwind.config.ts         # Custom industrial & alert design palette
    ├── postcss.config.mjs         # PostCSS configuration
    ├── next.config.mjs            # Next.js configuration
    ├── .eslintrc.json             # Next.js core web vitals linting
    ├── .env.example               # Frontend template (NEXT_PUBLIC_API_BASE_URL)
    └── src/
        ├── app/
        │   ├── globals.css        # Base styling
        │   ├── layout.tsx         # Responsive HTML wrapper & metadata
        │   └── page.tsx           # Main application state container with tabbed views
        ├── components/
        │   ├── OverviewView.tsx   # Overview, KPIs, architecture, sensor limits
        │   ├── SimulatorView.tsx  # Live simulator, gauges, and Recharts line charts
        │   ├── ModelLabView.tsx   # Model comparison, confusion matrix, feature gains
        │   ├── StatisticsView.tsx # SciPy hypothesis testing & FDR report
        │   ├── ReportsView.tsx    # Live session table & client-side CSV exporter
        │   └── Disclaimers.tsx    # Academic notices & API status banner
        └── lib/
            └── api.ts             # Typed REST API client
```

---

## 🚀 Local Development Setup

### 1. Backend Setup (Terminal 1)
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Start FastAPI development server
uvicorn app:app --port 8000 --reload
```
* Interactive Swagger API documentation: `http://localhost:8000/docs`
* Health check: `http://localhost:8000/health`

### 2. Frontend Setup (Terminal 2)
```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Configure local environment
cp .env.example .env.local

# 3. Start Next.js development server
npm run dev
```
* Interactive Web Application: `http://localhost:3000`

---

## ☁️ Vercel Deployment Guide (Dual Project Setup)

The application is architected to deploy as **two independent Vercel projects** connected to the same GitHub repository:

```mermaid
graph LR
    User[Web Browser] --> Frontend[Vercel Project 1: Next.js Frontend]
    Frontend -- NEXT_PUBLIC_API_BASE_URL --> Backend[Vercel Project 2: FastAPI Python Backend]
    Backend --> ML[Serialized Pipeline: Random Forest + Scalers]
```

### Preflight Verification Summary
* **Serverless Bundle Size:** The primary model artifact (`models/random_forest_pipeline.joblib`) is only **2.5 MB**. With all precomputed JSON metrics (<50 KB), the entire backend function package is ~2.6 MB, far below Vercel's 50 MB compressed / 250 MB uncompressed function limit.
* **Bundle Optimization:** `.vercelignore` strictly excludes `frontend/`, `tests/`, raw CSV data, and cache folders from the backend deployment bundle.
* **Portable Path Resolution:** Model and output loaders use relative path resolution with fallback to `Path(__file__)`, eliminating hardcoded local development paths.
* **Python Version:** `.python-version` pins `3.12` explicitly for Vercel's Python builder.

---

### Step 1: Push Repository to GitHub
In your local repository:
```bash
git init
git add .
git commit -m "Initial commit: Predictive Maintenance Intelligence System"
git branch -M main
git remote add origin https://github.com/<your-username>/predictive-maintenance-intelligence.git
git push -u origin main
```

---

### Step 2: Deploy Backend Project (Vercel Project 1)
1. Go to [Vercel Dashboard](https://vercel.com/dashboard) and click **Add New... > Project**.
2. Select your `predictive-maintenance-intelligence` GitHub repository.
3. Configure the Project:
   * **Project Name:** `predictive-maintenance-backend` (or your choice)
   * **Framework Preset:** `Other`
   * **Root Directory:** `./` (Leave as default root)
   * **Build & Development Settings:** Leave all defaults empty (Vercel automatically installs `requirements.txt` and uses `api/index.py` via `vercel.json`).
4. **Environment Variables:**
   * `ALLOWED_ORIGINS`: Set to `*` initially, or update to your frontend Vercel URL once Project 2 is deployed.
5. Click **Deploy**.
6. Once deployed, note down the production backend URL (e.g. `https://predictive-maintenance-backend.vercel.app`).

---

### Step 3: Deploy Frontend Project (Vercel Project 2)
1. In the Vercel Dashboard, click **Add New... > Project**.
2. Select the **same** `predictive-maintenance-intelligence` GitHub repository again.
3. Configure the Project:
   * **Project Name:** `predictive-maintenance-frontend` (or your choice)
   * **Framework Preset:** `Next.js`
   * **Root Directory:** Click **Edit** and select `frontend`
4. **Environment Variables:**
   * `NEXT_PUBLIC_API_BASE_URL`: Enter your deployed backend URL from Step 2 (e.g. `https://predictive-maintenance-backend.vercel.app` without trailing slash).
5. Click **Deploy**.
6. Once deployed, copy your production frontend URL (e.g. `https://predictive-maintenance-frontend.vercel.app`).

---

### Step 4: Secure Backend CORS (Post-Deployment Step)
1. Return to your **Backend Project** on Vercel: **Settings > Environment Variables**.
2. Update `ALLOWED_ORIGINS` to your exact frontend domain:
   ```text
   ALLOWED_ORIGINS=https://predictive-maintenance-frontend.vercel.app
   ```
3. Redeploy the backend to apply the restricted CORS allowlist.

---

## 🔍 Post-Deployment Smoke Testing Checklist

Once deployed, verify each endpoint on your live Vercel backend and frontend:

| Check | Endpoint / Action | Expected Result |
|---|---|---|
| **1. Health** | `GET /health` | `{"status": "healthy", "model_ready": true, "model_name": "Random Forest"}` |
| **2. API Docs** | `GET /docs` | Interactive Swagger UI renders all 8 endpoints |
| **3. Overview** | `GET /api/overview` | Returns 10,000 total observations, 3.39% failure rate, cutoff 0.31 |
| **4. Statistics** | `GET /api/statistics` | Returns 7 numeric tests with Benjamini-Hochberg FDR adjustments |
| **5. Model Metrics** | `GET /api/model/metrics` | Returns genuine out-of-fold and held-out test evaluation scores |
| **6. Predict API** | `POST /api/predict` | With valid sensor JSON, returns anomaly score and operational status |
| **7. Simulator Step** | `POST /api/simulator/step` | With state payload, returns tick + 1, sensor reading, and prediction |
| **8. Web Dashboard** | `https://<frontend-url>.vercel.app` | Header shows `Backend: Online (Random Forest)`; all 5 tabs render |
| **9. Live Simulation** | Live Simulator Tab | Click **Start**; sensor values update every 1.5s with live Recharts curves |
| **10. CSV Export** | Reports & Data Tab | Click **Export Session CSV**; downloads clean CSV in browser |

---

## 🧪 Automated Test Suite Execution

```bash
# Run all 55 Python unit, integration, and API tests
.venv/bin/python -m pytest -q

# Run frontend ESLint and Next.js static build check
cd frontend && npm run lint && npm run build
```
