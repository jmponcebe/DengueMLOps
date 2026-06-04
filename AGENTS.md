# DengueMLOps — Project Instructions

## Project Overview

End-to-end MLOps pipeline for dengue alert prediction in Brazil. Master's thesis (TFM) at UCLM (CIDaeN). All components complete: data, EDA, feature engineering, modeling, API, dashboard, Docker, CI/CD, AWS IaC, Evidently monitoring, TFM thesis.

**Repo**: <https://github.com/jmponcebe/DengueMLOps>

## Key Components

- **Data**: 4.5M weekly records from Mosqlimate API, 5,500+ municipalities (2010–2025)
- **Features**: 15 engineered features, zero target leakage, climate lags based on vector biology
- **Model**: XGBoost with balanced sample weights, Optuna-tuned (40 trials). Test macro_f1=0.39
- **Tracking**: MLflow 5 experiments, 90+ runs, Model Registry with "champion" alias
- **Serving**: FastAPI REST API + Streamlit dashboard with Brazil choropleth map
- **Monitoring**: Evidently drift detection (KS + chi²), prediction logging with auto-flush
- **Deploy**: Docker multi-image, GitHub Actions CI/CD, AWS ECS/Fargate + S3 + ECR, CloudFormation IaC
- **TFM**: LaTeX thesis in `docs/memoria/`, CIDaeN UCLM template, XeLaTeX, 78 pages, 5 chapters + 2 appendices

## Critical Design Constraints

- **No target leakage**: epidemiological variables (casos, Rt, p_inc100k) are NEVER used as features. They are consequences of the target, not predictors.
- **Climate with lag only**: climate variables available only with 4-8 week lag (mosquito biological cycle). Today's temperature doesn't predict today's dengue — temperature 6 weeks ago does.
- **`shift()` before `rolling()`**: always exclude current observation in temporal aggregations.
- **Region encoding**: 5 Brazilian macro-regions (domain knowledge), NOT target encoding.
- **Temporal splits**: Train 2010-2021, Validation 2022-2023, Test 2024.
- **Label encoding**: XGBoost requires 0-indexed labels. Use `encode_labels()`/`decode_labels()` from `src/models/constants.py`.
- **Class imbalance**: 93.1% level 1. Balanced sample weights (inverse frequency) are critical. Without them, models ignore minority classes.
- **Model Registry**: champion model as `dengue-alertlevel-classifier` with alias "champion" in MLflow.

## TFM Thesis

LaTeX in `docs/memoria/`. CIDaeN UCLM template, `book` class, Calibri font, blue theme. Compiles with XeLaTeX + BibTeX.

- **Ch1**: Introduccion (MLOps motivation, objectives, tech stack)
- **Ch2**: Fundamentos (ML, MLOps, dengue in Brazil)
- **Ch3**: Metodologia (data, features, MLflow experiments, Model Registry)
- **Ch4**: Resultados (FastAPI, Streamlit, Docker, CI/CD, AWS, Evidently)
- **Ch5**: Conclusiones y Trabajo Futuro
- **Terminology**: use "ingesta" (not "ingestion"), "contenerizacion" (not "containerizacion") in Spanish
- **Style**: academic narrative, explain WHY (design decisions, trade-offs), avoid bullet-point lists

```bash
cd docs/memoria && xelatex TFM.tex && bibtex TFM && xelatex TFM.tex && xelatex TFM.tex
```

## Deployment Architecture

- **API**: FastAPI port 8000, loads XGBoost model from MLflow registry or S3
- **Dashboard**: Streamlit port 8501, connects to API via `API_URL` env var
- **Docker**: `docker-compose.yml` orchestrates api + dashboard. AWS images download model/data from S3 at startup via entrypoint scripts
- **AWS**: S3 (data/model ~275MB parquet + GeoJSON + champion model), ECR (images), ECS/Fargate (compute). CloudFormation IaC.
- **CI/CD**: `ci.yml` tests+lint+docker build on push; `deploy.yml` pushes to ECR and updates ECS on version tags
