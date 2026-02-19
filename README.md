# Predicción de Dengue en Brasil - Proyecto MLOps para TFM

## Descripción del Proyecto

Pipeline MLOps completo para la predicción del nivel de alerta de dengue por municipio en Brasil, desarrollado como Trabajo de Fin de Máster (TFM). Utiliza datos históricos (2010-2025) de la API mosqlimate y variables climáticas para generar predicciones a nivel municipal respetando restricciones epidemiológicas reales.

## Objetivos

- **Pipeline MLOps completo**: Desde ingesta de datos hasta monitoreo en producción
- **Predicción sin leakage**: Solo variables disponibles en tiempo real de predicción
- **Visualización interactiva**: App Streamlit con mapa de Brasil
- **Monitoreo de calidad**: Detección de data drift con Evidently
- **Experimentación**: Tracking completo con MLflow

## Arquitectura del Proyecto

```text
tfm-mlops/
├── data/                        # Gestión de datos
│   ├── raw/                     # Datos originales (parquet jerárquico)
│   ├── interim/                 # Datos intermedios y resúmenes del EDA
│   ├── processed/               # Datasets finales para modelado
│   └── external/                # Datos de terceros (shapefiles, etc.)
├── notebooks/                   # Jupyter notebooks de exploración
│   ├── 01-exploratory-data-analysis.ipynb
│   ├── 02-feature-engineering.ipynb
│   └── 03-modeling.ipynb
├── src/                         # Código de producción
│   ├── data/                    # Ingesta y procesamiento
│   │   ├── api_client.py        # Cliente API mosqlimate (sync_dataset)
│   │   └── mosqlimate_loader.py # Lectura de parquet local
│   ├── features/                # Feature engineering
│   │   ├── constants.py         # Clasificación de variables y constantes
│   │   └── feature_engineer.py  # Pipeline DengueFeatureEngineer
│   ├── models/                  # Entrenamiento y evaluación
│   │   ├── constants.py         # Label maps, clases, balanced weights
│   │   ├── evaluation.py        # Métricas, confusion matrix, reports
│   │   └── trainer.py           # DengueModelTrainer con MLflow logging
│   └── monitoring/              # Monitoreo y alertas
├── app/                         # Aplicación Streamlit y API
├── models/                      # Modelos entrenados y artefactos
├── monitoring/                  # Reportes de monitoreo
├── docs/                        # Documentación técnica y académica
│   └── memoria/                 # Memoria TFM (LaTeX, CIDaeN UCLM)
├── tests/                       # Tests unitarios
│   ├── test_data_api.py         # Tests del cliente API
│   └── test_features.py         # Tests de feature engineering (26 tests)
└── configs/                     # Archivos de configuración
```

## Estado Actual del Proyecto

### Fase 1: Datos y Exploración — Completada

- **Ingesta**: Cliente API mosqlimate + loader de parquet local (27 estados, 2010-2025)
- **EDA**: Análisis completo con 4.3M registros, 5,563 municipios
- **Hallazgos clave**:
  - Estacionalidad marcada: pico Mar-Abr, valle Sep (ratio 9.9x)
  - Desbalance extremo del target: 93.1% nivel 1 (verde)
  - Heterogeneidad regional significativa entre estados
  - Clima opera con 4-8 semanas de lag sobre transmisión vectorial

### Fase 2: Feature Engineering — Completada

- **15 features engineered** sin target leakage:
  - Temporales (6): encoding cíclico de mes/SE, indicador temporada pico, trimestre
  - Climáticas (4): lags 4-8 semanas, rolling means con shift, interacciones temp×humedad
  - Geográficas (5): log población, macro-región brasileña (one-hot 5 regiones)
- **Principios aplicados**:
  - `shift()` antes de `rolling()` para evitar leakage temporal
  - Region encoding por conocimiento de dominio (no target encoding)
  - Variables epidemiológicas y clima simultáneo eliminados de producción
- **Validación con modelo**: RF baseline vs RF+FE en validación temporal
- **Módulo productivo**: `src/features/` con 26 tests unitarios (97% cobertura)
- **Datasets de producción**: 25 columnas (9 ID + 15 features + target)

### Fase 3: Modelado — Completada

- **5 experimentos MLflow** organizados por fase (baselines → evaluación final)
- **Baselines**: Dummy, Logistic Regression, Decision Tree (macro_f1 ~0.23-0.32)
- **Model Selection**: RF, XGBoost, LightGBM, CatBoost por defecto (~0.27)
- **HP Tuning con Optuna**: 40 trials para XGBoost y LightGBM con nested runs, subsample 15%
- **Class imbalance**: Balanced sample weights (macro_f1 0.27→0.38) y custom weights para salud pública
- **Champion model**: XGBoost + balanced weights + Optuna HP → macro_f1=0.39, kappa=0.33 en test 2024
- **MLflow avanzado**: Model Registry con alias "champion", dataset lineage, mlflow.evaluate(), SHAP
- **Interpretabilidad**: SHAP TreeExplainer — `pop_log` y `month_sin` son los features más importantes
- **Módulo productivo**: `src/models/` con constants, evaluation y trainer

### Fase 4: Deployment y Monitoreo — Completada

- **API REST** (FastAPI): endpoints `/predict`, `/predict/batch`, `/health`, `/model/info`
- **Streamlit Dashboard**: 4 páginas — mapa choropleth de Brasil por UF, predicción individual, info del modelo, monitoreo
- **Mapa de alerta**: Visualización geoespacial con datos reales por semana epidemiológica, 3 métodos de agregación, detalle municipal
- **Monitoreo**: DriftDetector con Evidently (DataDriftPreset, DataSummaryPreset), logging de predicciones a CSV
- **Docker**: Multi-imagen (API + Dashboard), docker-compose con healthchecks y volúmenes
- **CI/CD**: GitHub Actions — tests/lint en push, deploy a AWS ECR/ECS en tags
- **AWS**: Guía de despliegue para EC2 y ECS/Fargate con Learner Lab

### Fase 5: Memoria TFM — Completada

- **Plantilla**: CIDaeN UCLM, compilación con XeLaTeX + BibTeX
- **Estructura MLOps-focused** (5 capítulos + 2 apéndices):
  - Cap 1: Introducción (motivación MLOps, objetivos, estructura)
  - Cap 2: Fundamentos (ML, MLOps, dengue en Brasil)
  - Cap 3: Metodología y Desarrollo (datos, features, MLflow, serving, Docker, CI/CD, AWS, monitoreo)
  - Cap 4: Resultados (métricas, trazabilidad MLflow, sistema desplegado, pipeline CI/CD, AWS, monitoreo)
  - Cap 5: Conclusiones y Trabajo Futuro
  - Apéndice A: Stack tecnológico (23 herramientas con versiones)
  - Apéndice B: Anexo técnico (19 fragmentos de código)
- **Bibliografía**: 18 referencias (epidemiología, ML, MLOps, frameworks)
- **Capturas incluidas**: MLflow (6 screenshots), Swagger UI, Streamlit (mapa + predicción + monitoreo), Docker Compose, pytest + cobertura, GitHub Actions (CI + CD), AWS (ECR, EC2, ECS), confession matrix, Evidently drift report
- **3 diagramas propios**: Ciclo vectorial dengue, arquitectura Docker, arquitectura AWS

## Datos

### Fuentes

- **Dengue**: API mosqlimate (2010-2025), estructura jerárquica `uf={estado}/year={año}/`
- **Clima**: Variables meteorológicas por municipio (temperatura, humedad)

### Target

- `nivel`: Nivel de alerta epidemiológica (1-4)
  - Nivel 1 (verde): 93.1% — Sin alerta
  - Nivel 2 (amarillo): 3.9% — Atenção
  - Nivel 3 (naranja): 0.2% — Alerta
  - Nivel 4 (rojo): 2.8% — Alerta crítico

### Splits Temporales

- **Train**: 2010-2021 (3.48M registros)
- **Validation**: 2022-2023 (584K registros)
- **Test**: 2024 (290K registros)

## Uso Rápido

### Instalación

```bash
git clone <repo-url>
cd tfm-mlops
python -m venv .venv
.venv\Scripts\activate  # Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Configurar variables de entorno

```bash
# Copiar el ejemplo y rellenar con valores reales
cp .env.example .env
# Variables principales: MOSQLIMATE_API_KEY, AWS_ACCESS_KEY_ID, etc.
```

### Sincronizar datos

```python
from src.data import sync_dataset
sync_dataset(year_start=2010)
```

### Feature engineering

```python
from src.features import DengueFeatureEngineer
import pandas as pd

df = pd.read_parquet('data/processed/dengue_train_2010_2021.parquet')
fe = DengueFeatureEngineer()
df_full = fe.transform(df)
df_prod = fe.get_production_dataset(df_full)
```

### Levantar API + Dashboard (Docker)

```bash
docker compose up --build
# API: http://localhost:8000  |  Dashboard: http://localhost:8501
```

### Levantar API + Dashboard (local)

```bash
# Terminal 1: API
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Dashboard
streamlit run app/streamlit_app.py
```

### Tests

```bash
pytest tests/ -v
```

## Tecnologías

- **Core**: Python 3.13, Pandas 2.3, NumPy 2.3, Scikit-learn 1.7
- **ML**: XGBoost 3.0, Optuna 4.5, SHAP 0.50
- **MLOps**: MLflow 3.3 (tracking, registry, evaluate), Evidently 0.7
- **App**: Streamlit 1.49, FastAPI 0.116, Pydantic 2.11
- **Viz**: Plotly 6.3, Matplotlib, Seaborn
- **Testing**: Pytest 8.4 (con cobertura)
- **Infra**: Docker 28.2, Docker Compose 2.37, GitHub Actions
- **Cloud**: AWS ECR/ECS/Fargate/EC2
- **Thesis**: LaTeX (XeLaTeX), BibTeX, MiKTeX

---

## Última actualización

Febrero 2026
