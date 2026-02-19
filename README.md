# 🦟 DengueMLOps — End-to-End MLOps Pipeline for Dengue Alert Prediction

[![CI](https://github.com/jmponcebe/DengueMLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/jmponcebe/DengueMLOps/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/deploy-Docker-2496ED.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/cloud-AWS-FF9900.svg)](https://aws.amazon.com/)
[![Evidently](https://img.shields.io/badge/monitoring-Evidently-6C3EC2.svg)](https://www.evidentlyai.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> 🇪🇸 [Leer en español](README.es.md) · 🇧🇷 [Ler em português](README.pt-br.md)

Production-grade ML pipeline that predicts **dengue alert levels** (1-4) across 5,500+ Brazilian municipalities using only non-leaking epidemiological features. Built as a showcase of modern MLOps practices: experiment tracking, model serving, containerization, CI/CD, cloud deployment, and data drift monitoring.

> **Master's Thesis Project** — CIDaeN, Universidad de Castilla-La Mancha (UCLM)

---

## Highlights

| What | How |
| --- | --- |
| **Data** | 4.5M weekly records (2010-2025) from [Mosqlimate API](https://api.mosqlimate.org/) |
| **Features** | 15 engineered features, zero target leakage, climate lags based on vector biology |
| **Model** | XGBoost + balanced weights, Optuna-tuned (40 trials). macro_f1=0.39 on 2024 test |
| **Tracking** | 5 MLflow experiments, 90+ runs, Model Registry with champion alias |
| **Serving** | FastAPI REST API + Streamlit dashboard with interactive Brazil choropleth map |
| **Containers** | Multi-image Docker setup, Compose orchestration with health checks |
| **CI/CD** | GitHub Actions — tests/lint on push, deploy to AWS ECR/ECS on version tags |
| **Cloud** | AWS ECS/Fargate + S3 + ECR, Infrastructure as Code (CloudFormation) |
| **Monitoring** | Evidently data drift detection, prediction logging with auto-flush |
| **Tests** | 88 unit tests, 97% coverage on feature engineering |

---

## Demo

### Streamlit Dashboard — Brazil Alert Map

<p align="center">
  <img src="docs/images/streamlit_dashboard.png" alt="Streamlit Dashboard" width="700">
</p>

### FastAPI — Auto-generated Swagger Docs

<p align="center">
  <img src="docs/images/swagger_ui.png" alt="Swagger UI" width="700">
</p>

### MLflow — Experiment Tracking

<p align="center">
  <img src="docs/images/mlflow_experiments.png" alt="MLflow Experiments" width="700">
</p>

---

## Architecture

### Docker Compose

<p align="center">
  <img src="docs/images/architecture_docker.png" alt="Docker Architecture" width="700">
</p>

### AWS Deployment (ECS/Fargate)

<p align="center">
  <img src="docs/images/architecture_aws.png" alt="AWS Architecture" width="700">
</p>

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)

### Installation

```bash
git clone https://github.com/jmponcebe/DengueMLOps.git
cd DengueMLOps
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Download Data & Model

```bash
python scripts/setup_data.py                      # Downloads champion model + GeoJSON
python scripts/setup_data.py --latest              # + current year data (quick start)
python scripts/setup_data.py --data --from 2023    # Only data from 2023 to present
python scripts/setup_data.py --all                 # Full historical dataset (2010-present)
```

The setup script downloads:

- **Champion model** from [GitHub Releases](https://github.com/jmponcebe/DengueMLOps/releases) (~6 MB)
- **Brazil GeoJSON** from IBGE API for the dashboard map
- **Historical data** from [Mosqlimate API](https://api.mosqlimate.org/) (optional, requires free API key)

> **Note**: Without data download, the API prediction works normally. The dashboard alert map requires at least `--latest` for current year data, or `--all` for the full historical view (~1.5 GB).

### Run with Docker (recommended)

```bash
docker compose up --build
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
# Swagger:   http://localhost:8000/docs
```

### Run locally

```bash
# Terminal 1: API
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Dashboard
streamlit run app/streamlit_app.py
```

### Run tests

```bash
pytest tests/ -v --cov=src
```

---

## Project Structure

```text
├── app/                     # Deployment
│   ├── api.py               # FastAPI REST API (5 endpoints)
│   ├── schemas.py           # Pydantic models for validation
│   └── streamlit_app.py     # Interactive dashboard (4 pages)
├── src/                     # Production code
│   ├── data/                # Ingestion (API client + Parquet loader)
│   ├── features/            # DengueFeatureEngineer (15 features, anti-leakage)
│   ├── models/              # Training, evaluation, MLflow integration
│   └── monitoring/          # Evidently drift detection
├── notebooks/               # Exploration & experimentation
│   ├── 01-exploratory-data-analysis.ipynb
│   ├── 02-feature-engineering.ipynb
│   └── 03-modeling.ipynb
├── configs/                 # MLflow & project configuration
├── tests/                   # 88 unit tests
├── scripts/                 # AWS deploy & entrypoint scripts
├── aws/                     # CloudFormation IaC template
├── .github/workflows/       # CI/CD pipelines
├── Dockerfile.api           # API container
├── Dockerfile.dashboard     # Dashboard container
└── docker-compose.yml       # Multi-service orchestration
```

---

## MLOps Practices

### Experiment Tracking (MLflow)

5 sequential experiments, each building on the previous one's insights:

| # | Experiment | Runs | Key Finding |
| --- | --- | --- | --- |
| 01 | baselines | 4 | Floor: macro_f1=0.23 (dummy) |
| 02 | model-selection | 5 | Algorithm choice barely matters with imbalanced data |
| 03 | hyperparameter-tuning | 2+80 | Optuna + nested runs, minimal improvement |
| 04 | class-imbalance | 2 | **Balanced weights: +38% macro_f1** (the breakthrough) |
| 05 | final-evaluation | 3 | Champion on held-out 2024 test data |

Key insight: **class imbalance handling** (balanced sample weights) has far more impact than algorithm selection or hyperparameter tuning when dealing with 460:1 class ratios.

### Feature Engineering (Zero Leakage)

The biggest challenge: dengue datasets contain epidemiological variables (cases, Rt, incidence) that are **derived from the target**. Using them produces artificially inflated metrics.

Our 15 production features use **only** temporal, climate (with biological lag), and geographic information:

- Climate variables lagged 4-8 weeks (matching the vector lifecycle: egg → adult → bite → diagnosis)
- `shift(1)` before `rolling()` to prevent current-observation leakage
- Region encoding via domain knowledge (5 Brazilian macro-regions), not target encoding

### Model Serving

- **FastAPI** REST API with Pydantic validation, auto-generated OpenAPI docs
- **Resilient loading**: MLflow Registry → local artifact fallback
- **Streamlit** dashboard with Brazil choropleth map (Plotly + IBGE GeoJSON)

### CI/CD Pipeline

```text
Push to main → CI (test → lint → docker build + smoke test)
Git tag v* → CD (build images → push to ECR → update ECS services)
```

### Monitoring

- Evidently batch drift detection (KS test for numerical, chi² for categorical)
- Prediction logging with auto-flush buffer (100 predictions → CSV)
- Interactive drift reports accessible from the Streamlit dashboard

### Cloud Deployment (AWS)

- **Infrastructure as Code**: CloudFormation template for S3 + ECR + ECS/Fargate
- **Two deployment options**: EC2 (simple/cheap) and ECS/Fargate (scalable/managed)
- **Data separation**: Images don't contain data — containers download from S3 at startup
- **Automated setup**: 4-phase script (stack → S3 upload → ECR push → ECS services)

---

## Key Technical Decisions

| Decision | Rationale |
| --- | --- |
| Separate API and Dashboard images | Independent scaling (API is CPU-bound, Dashboard is I/O-bound) |
| Climate lags 4-8 weeks | Matches vector biology cycle (egg → adult → transmission → diagnosis) |
| `pop_log` as top feature | Dengue is fundamentally urban; population captures density and vector habitats |
| Balanced sample weights | With 93.1% class 1, models without weighting ignore minority alerts |
| Temporal train/val/test split | Respects time ordering; 2024 as test (record dengue year in Brazil) |
| MLflow champion alias | API loads model by alias, decoupled from specific run IDs |

---

## Tech Stack

| Category | Technologies |
| --- | --- |
| **Core** | Python 3.13, Pandas, NumPy, Scikit-learn |
| **ML** | XGBoost, Optuna, SHAP |
| **MLOps** | MLflow (tracking + registry + evaluate), Evidently |
| **App** | FastAPI, Streamlit, Pydantic, Plotly |
| **Testing** | Pytest (88 tests, 97% coverage on features) |
| **Infra** | Docker, Docker Compose, GitHub Actions |
| **Cloud** | AWS S3, ECR, ECS/Fargate, CloudFormation |

---

## Data

| Aspect | Detail |
| --- | --- |
| **Source** | [Mosqlimate API](https://api.mosqlimate.org/) — epidemiological + climate data |
| **Volume** | 4.5M weekly records, 5,500+ municipalities |
| **Period** | 2010-2025 |
| **Target** | Alert level 1-4 (93.1% level 1 — extreme imbalance) |
| **Train** | 2010-2021 (3.48M records) |
| **Validation** | 2022-2023 (584K records) |
| **Test** | 2024 (290K records) — record dengue year in Brazil |

---

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check for Docker/load balancers |
| `GET` | `/model/info` | Loaded model metadata |
| `POST` | `/predict` | Single prediction (15 features → alert level + probabilities) |
| `POST` | `/predict/batch` | Batch prediction |
| `POST` | `/monitoring/flush` | Force prediction buffer flush to CSV |

---

## Data Drift Monitoring

The project includes data drift detection with [Evidently](https://www.evidentlyai.com/). To generate a demo report:

```bash
python scripts/monitoring_demo.py              # Uses real data if available, otherwise synthetic
python scripts/monitoring_demo.py --simulate   # Force synthetic data with artificial drift
```

With historical data (`setup_data.py --all`), the demo uses train data (2010-2021) as reference and recent data (2023+) as production. Without data, it generates synthetic features with configurable drift.

The generated report is viewable in the Streamlit dashboard ("Monitoreo" tab) or directly as an HTML file in `monitoring/reports/`.

---

## Roadmap

- [ ] **DVC integration** — Version data and model artifacts with [DVC](https://dvc.org/) for full reproducibility (`dvc pull` to get everything)
- [ ] **Mosqlimate community models** — Integrate public prediction models from [Mosqlimate](https://api.mosqlimate.org/) for ensemble or benchmark comparison
- [ ] **Automated data pipeline** — Periodic sync from Mosqlimate API to S3, keeping production data up-to-date without manual intervention
- [ ] **Automated retraining** — End-to-end pipeline triggered by data drift: sync data → retrain → evaluate → promote champion
- [ ] **Advanced monitoring** — Model performance monitoring with ground truth feedback loop
- [ ] **Multi-model serving** — A/B testing between model versions via the API
- [ ] **Feature store** — Centralized feature computation and serving

---

## License

This project is licensed under the [MIT License](LICENSE).

Developed as a Master's Thesis at [CIDaeN](https://cidaen.uclm.es/), Universidad de Castilla-La Mancha (UCLM).

---

*Built by [Jose María Ponce Bernabé](https://github.com/jmponcebe) — 2026*
