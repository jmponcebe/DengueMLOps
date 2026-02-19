# 🦟 DengueMLOps — Pipeline MLOps Completo para Predicción de Alertas de Dengue

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

> 🇬🇧 [Read in English](README.md) · 🇧🇷 [Ler em português](README.pt-br.md)

Pipeline de ML de nivel producción que predice **niveles de alerta de dengue** (1-4) en más de 5.500 municipios brasileños, usando únicamente variables epidemiológicas sin fuga de datos. Desarrollado como demostración de prácticas MLOps modernas: tracking de experimentos, servicio de modelos, contenedorización, CI/CD, despliegue en la nube y monitorización de data drift.

> **Trabajo de Fin de Máster** — CIDaeN, Universidad de Castilla-La Mancha (UCLM)

---

## Puntos Clave

| Qué | Cómo |
| --- | --- |
| **Datos** | 4,5M registros semanales (2010-2025) de la [API Mosqlimate](https://api.mosqlimate.org/) |
| **Features** | 15 variables ingenierizadas, cero fuga de datos, lags climáticos basados en biología vectorial |
| **Modelo** | XGBoost + pesos balanceados, optimizado con Optuna (40 trials). macro_f1=0,39 en test 2024 |
| **Tracking** | 5 experimentos MLflow, +90 ejecuciones, Model Registry con alias champion |
| **Servicio** | API REST con FastAPI + dashboard Streamlit con mapa coroplético interactivo de Brasil |
| **Contenedores** | Setup Docker multi-imagen, orquestación con Compose y health checks |
| **CI/CD** | GitHub Actions — tests/lint en push, despliegue a AWS ECR/ECS en tags de versión |
| **Cloud** | AWS ECS/Fargate + S3 + ECR, Infraestructura como Código (CloudFormation) |
| **Monitorización** | Detección de data drift con Evidently, logging de predicciones con auto-flush |
| **Tests** | 88 tests unitarios, 97% de cobertura en ingeniería de features |

---

## Demo

### Dashboard Streamlit — Mapa de Alertas de Brasil

<p align="center">
  <img src="docs/images/streamlit_dashboard.png" alt="Dashboard Streamlit" width="700">
</p>

### FastAPI — Documentación Swagger Autogenerada

<p align="center">
  <img src="docs/images/swagger_ui.png" alt="Swagger UI" width="700">
</p>

### MLflow — Tracking de Experimentos

<p align="center">
  <img src="docs/images/mlflow_experiments.png" alt="Experimentos MLflow" width="700">
</p>

---

## Arquitectura

### Docker Compose

<p align="center">
  <img src="docs/images/architecture_docker.png" alt="Arquitectura Docker" width="700">
</p>

### Despliegue en AWS (ECS/Fargate)

<p align="center">
  <img src="docs/images/architecture_aws.png" alt="Arquitectura AWS" width="700">
</p>

---

## Inicio Rápido

### Requisitos previos

- Python 3.11+
- Docker y Docker Compose (para despliegue contenedorizado)

### Instalación

```bash
git clone https://github.com/jmponcebe/DengueMLOps.git
cd DengueMLOps
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Descargar Datos y Modelo

```bash
python scripts/setup_data.py                      # Descarga modelo champion + GeoJSON
python scripts/setup_data.py --latest              # + datos del año actual (inicio rápido)
python scripts/setup_data.py --data --from 2023    # Solo datos desde 2023 hasta hoy
python scripts/setup_data.py --all                 # Dataset histórico completo (2010-presente)
```

El script de setup descarga:
- **Modelo champion** desde [GitHub Releases](https://github.com/jmponcebe/DengueMLOps/releases) (~6 MB)
- **GeoJSON de Brasil** desde la API del IBGE para el mapa del dashboard
- **Datos históricos** desde la [API de Mosqlimate](https://api.mosqlimate.org/) (opcional, requiere API key gratuita)

> **Nota**: Sin descargar datos, la predicción de la API funciona normalmente. El mapa de alertas del dashboard requiere al menos `--latest` para datos del año actual, o `--all` para la vista histórica completa (~1.5 GB).

### Ejecutar con Docker (recomendado)

```bash
docker compose up --build
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
# Swagger:   http://localhost:8000/docs
```

### Ejecutar en local

```bash
# Terminal 1: API
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Dashboard
streamlit run app/streamlit_app.py
```

### Ejecutar tests

```bash
pytest tests/ -v --cov=src
```

---

## Estructura del Proyecto

```text
├── app/                     # Despliegue
│   ├── api.py               # API REST FastAPI (5 endpoints)
│   ├── schemas.py           # Modelos Pydantic para validación
│   └── streamlit_app.py     # Dashboard interactivo (4 páginas)
├── src/                     # Código de producción
│   ├── data/                # Ingesta (cliente API + cargador Parquet)
│   ├── features/            # DengueFeatureEngineer (15 features, anti-fuga)
│   ├── models/              # Entrenamiento, evaluación, integración MLflow
│   └── monitoring/          # Detección de drift con Evidently
├── notebooks/               # Exploración y experimentación
│   ├── 01-exploratory-data-analysis.ipynb
│   ├── 02-feature-engineering.ipynb
│   └── 03-modeling.ipynb
├── configs/                 # Configuración MLflow y proyecto
├── tests/                   # 88 tests unitarios
├── scripts/                 # Scripts de despliegue AWS y entrypoints
├── aws/                     # Template CloudFormation (IaC)
├── .github/workflows/       # Pipelines CI/CD
├── Dockerfile.api           # Contenedor API
├── Dockerfile.dashboard     # Contenedor Dashboard
└── docker-compose.yml       # Orquestación multi-servicio
```

---

## Prácticas MLOps

### Tracking de Experimentos (MLflow)

5 experimentos secuenciales, cada uno construyendo sobre las conclusiones del anterior:

| # | Experimento | Runs | Hallazgo Clave |
| --- | --- | --- | --- |
| 01 | baselines | 4 | Suelo: macro_f1=0,23 (dummy) |
| 02 | model-selection | 5 | La elección de algoritmo apenas importa con datos desbalanceados |
| 03 | hyperparameter-tuning | 2+80 | Optuna + runs anidados, mejora mínima |
| 04 | class-imbalance | 2 | **Pesos balanceados: +38% macro_f1** (descubrimiento clave) |
| 05 | final-evaluation | 3 | Campeón en datos de test 2024 no vistos |

Conclusión clave: **la gestión del desbalance de clases** (pesos balanceados por frecuencia inversa) tiene mucho más impacto que la selección de algoritmo o el tuning de hiperparámetros cuando hay ratios de clase de 460:1.

### Ingeniería de Features (Cero Fuga de Datos)

El mayor reto: los datasets de dengue contienen variables epidemiológicas (casos, Rt, incidencia) que están **derivadas del target**. Usarlas produce métricas artificialmente infladas.

Nuestras 15 features de producción usan **únicamente** información temporal, climática (con lag biológico) y geográfica:

- Variables climáticas con lag de 4-8 semanas (coincidiendo con el ciclo vectorial: huevo → adulto → picadura → diagnóstico)
- `shift(1)` antes de `rolling()` para evitar fuga de la observación actual
- Codificación regional por conocimiento de dominio (5 macrorregiones brasileñas), no target encoding

### Servicio de Modelos

- API REST **FastAPI** con validación Pydantic y documentación OpenAPI autogenerada
- **Carga resiliente**: MLflow Registry → fallback a artefacto local
- Dashboard **Streamlit** con mapa coroplético de Brasil (Plotly + GeoJSON del IBGE)

### Pipeline CI/CD

```text
Push a main → CI (test → lint → docker build + smoke test)
Git tag v* → CD (build imágenes → push a ECR → actualizar servicios ECS)
```

### Monitorización

- Detección batch de drift con Evidently (test KS para numéricas, chi² para categóricas)
- Logging de predicciones con buffer auto-flush (100 predicciones → CSV)
- Informes de drift interactivos accesibles desde el dashboard Streamlit

### Despliegue en Cloud (AWS)

- **Infraestructura como Código**: Template CloudFormation para S3 + ECR + ECS/Fargate
- **Dos opciones de despliegue**: EC2 (simple/económico) y ECS/Fargate (escalable/gestionado)
- **Separación de datos**: Las imágenes no contienen datos — los contenedores descargan de S3 al arrancar
- **Setup automatizado**: Script de 4 fases (stack → upload S3 → push ECR → servicios ECS)

---

## Decisiones Técnicas Clave

| Decisión | Justificación |
| --- | --- |
| Imágenes separadas para API y Dashboard | Escalado independiente (API usa CPU, Dashboard usa I/O) |
| Lags climáticos de 4-8 semanas | Coincide con el ciclo biológico del vector (huevo → adulto → transmisión → diagnóstico) |
| `pop_log` como feature principal | El dengue es fundamentalmente urbano; la población captura densidad y hábitats vectoriales |
| Pesos balanceados | Con 93,1% de clase 1, los modelos sin pesos ignoran las alertas minoritarias |
| Split temporal train/val/test | Respeta el orden temporal; 2024 como test (año récord de dengue en Brasil) |
| Alias champion en MLflow | La API carga el modelo por alias, desacoplado de IDs de ejecución específicos |

---

## Stack Tecnológico

| Categoría | Tecnologías |
| --- | --- |
| **Core** | Python 3.13, Pandas, NumPy, Scikit-learn |
| **ML** | XGBoost, Optuna, SHAP |
| **MLOps** | MLflow (tracking + registry + evaluate), Evidently |
| **App** | FastAPI, Streamlit, Pydantic, Plotly |
| **Testing** | Pytest (88 tests, 97% cobertura en features) |
| **Infra** | Docker, Docker Compose, GitHub Actions |
| **Cloud** | AWS S3, ECR, ECS/Fargate, CloudFormation |

---

## Datos

| Aspecto | Detalle |
| --- | --- |
| **Fuente** | [API Mosqlimate](https://api.mosqlimate.org/) — datos epidemiológicos + climáticos |
| **Volumen** | 4,5M registros semanales, +5.500 municipios |
| **Periodo** | 2010-2025 |
| **Target** | Nivel de alerta 1-4 (93,1% nivel 1 — desbalance extremo) |
| **Train** | 2010-2021 (3,48M registros) |
| **Validación** | 2022-2023 (584K registros) |
| **Test** | 2024 (290K registros) — año récord de dengue en Brasil |

---

## Endpoints de la API

| Método | Endpoint | Descripción |
| --- | --- | --- |
| `GET` | `/health` | Health check para Docker/balanceadores de carga |
| `GET` | `/model/info` | Metadatos del modelo cargado |
| `POST` | `/predict` | Predicción individual (15 features → nivel de alerta + probabilidades) |
| `POST` | `/predict/batch` | Predicción por lotes |
| `POST` | `/monitoring/flush` | Forzar flush del buffer de predicciones a CSV |

---

## Hoja de Ruta

- [ ] **Integración DVC** — Versionar datos y artefactos de modelo con [DVC](https://dvc.org/) para reproducibilidad completa (`dvc pull` para obtener todo)
- [ ] **Modelos comunitarios Mosqlimate** — Integrar modelos públicos de predicción de [Mosqlimate](https://api.mosqlimate.org/) para ensemble o comparación benchmark
- [ ] **Pipeline de datos automático** — Sincronización periódica desde la API de Mosqlimate hacia S3, manteniendo los datos de producción actualizados sin intervención manual
- [ ] **Reentrenamiento automático** — Pipeline extremo a extremo activado por drift: sincronizar datos → reentrenar → evaluar → promover champion
- [ ] **Monitoreo avanzado** — Monitoreo del rendimiento del modelo con bucle de retroalimentación
- [ ] **Multi-model serving** — Pruebas A/B entre versiones de modelo vía la API
- [ ] **Feature store** — Computación y servicio centralizado de features

---

## Licencia

Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).

Desarrollado como Trabajo de Fin de Máster en el [CIDaeN](https://cidaen.uclm.es/), Universidad de Castilla-La Mancha (UCLM).

---

*Desarrollado por [Jose María Ponce Bernabé](https://github.com/jmponcebe) — 2025*
