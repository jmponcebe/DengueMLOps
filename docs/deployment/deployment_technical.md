# Despliegue y Operacionalización — Contenido para la Memoria del TFM

## Estructura del capítulo propuesto

Este documento contiene el material técnico y las explicaciones necesarias
para redactar el capítulo de despliegue, monitoreo y CI/CD de la memoria.

---

## 1. Serving del Modelo: API REST con FastAPI

### 1.1. Justificación de la arquitectura

Se optó por un servicio RESTful con FastAPI frente a alternativas como
Flask o MLflow Serving por las siguientes razones:

- **Validación automática**: Pydantic permite definir schemas tipados
  para las 15 features de producción, validando rangos y tipos antes
  de la inferencia. Esto previene errores silenciosos en producción.
- **Documentación auto-generada**: FastAPI produce un Swagger UI
  interactivo (`/docs`) y especificación OpenAPI, facilitando la
  integración con otros sistemas.
- **Control de la lógica de negocio**: A diferencia de `mlflow models serve`,
  una API custom permite integrar el decode de labels (0-3 → 1-4),
  logging de predicciones para monitoreo, y respuestas enriquecidas
  (probabilidades por clase, color del nivel de alerta, etc.).
- **Rendimiento**: FastAPI es async-native, lo que permite manejar
  concurrencia sin necesidad de workers adicionales para I/O-bound tasks.

### 1.2. Carga del modelo

El modelo champion (XGBoost) se carga al iniciar el servicio mediante
el patrón **lifespan** de FastAPI (anteriormente `startup_event`).
Se implementó una estrategia de fallback:

1. **Intento 1**: Cargar desde MLflow Model Registry
   (`models:/dengue-alertlevel-classifier@champion`). Requiere que
   el tracking server esté accesible.
2. **Intento 2**: Cargar directamente desde el artefacto local
   (`mlflow-artifacts/models/.../artifacts/model.xgb`). Funciona
   sin servidor MLflow, ideal para contenedores standalone.

Esta estrategia permite flexibilidad: en desarrollo local se usa el
registro, en producción containerizada se embebe el artefacto en la
imagen Docker.

### 1.3. Endpoints

| Endpoint | Método | Latencia típica | Descripción |
| --- | --- | --- | --- |
| `/health` | GET | <5ms | Health check para load balancers y orquestadores |
| `/model/info` | GET | <5ms | Metadata: algoritmo, features, métricas, versión |
| `/predict` | POST | ~15ms | Predicción individual con probabilidades |
| `/predict/batch` | POST | ~15ms × N | Predicción por lotes (máx 1000) |
| `/monitoring/flush` | POST | Variable | Fuerza escritura del buffer de predicciones |

### 1.4. Logging de predicciones

Cada predicción se almacena en un buffer en memoria que se escribe
a disco (CSV) cada 100 predicciones. Este log sirve como *current dataset*
para la detección de drift con Evidently.

Campos registrados: timestamp, features de entrada, nivel predicho.

---

## 2. Dashboard Interactivo: Streamlit

### 2.1. Diseño de la interfaz

El dashboard se organizó en cuatro páginas:

1. **Mapa de Alerta** (página principal): Visualización geoespacial de
   Brasil mostrando el nivel de alerta de dengue por estado (UF). Usa
   `px.choropleth_mapbox` con un GeoJSON simplificado del IBGE (91%
   reducción de vértices vía Douglas-Peucker).
   Permite filtrar por año/semana epidemiológica y método de agregación
   (moda, media, máximo). Incluye detalle por municipio para cada estado.

2. **Predicción**: Formulario interactivo donde el usuario selecciona
   estado, mes, semana epidemiológica, población y datos climáticos.
   La app calcula automáticamente las transformaciones de features
   (encoding cíclico, log de población, one-hot de región) y envía
   la petición a la API. El resultado se muestra con código de color
   y distribución de probabilidades.

3. **Información del Modelo**: Muestra las métricas del champion
   (macro_f1, kappa, accuracy), las features agrupadas por categoría,
   y los artefactos visuales (matriz de confusión, SHAP summary plot).

4. **Monitoreo**: Distribución de predicciones recientes, acceso a
   reportes de drift de Evidently, comandos CLI para generar reportes
   offline (`scripts/monitoring_demo.py`), y botón de flush del buffer.

### 2.2. Integración API-Dashboard

Streamlit se conecta a la API REST vía HTTP. En Docker Compose,
la variable `API_URL=http://api:8000` permite la comunicación
inter-containers por nombre de servicio. El dashboard incluye un
health check visual en el sidebar que indica el estado de la API.

---

## 3. Monitoreo: Data Drift con Evidently

### 3.1. Estrategia de monitoreo

El dominio epidemiológico opera en escala semanal (semanas epidemiológicas),
por lo que el monitoreo en tiempo real no aporta valor. Se implementó
un sistema **batch** de detección de drift:

- **Dataset de referencia**: Datos de entrenamiento (2010-2021),
  con las 15 features de producción. Representa la distribución
  "esperada" por el modelo.
- **Dataset actual**: Predicciones recientes logueadas por la API.
  Se analiza una ventana deslizante (últimas 500 predicciones por defecto).

### 3.2. Métricas de drift

Evidently calcula tests estadísticos por feature:

- **Numéricas** (tempmed_lag8w, pop_log, etc.): Kolmogorov-Smirnov
  test con umbral por defecto.
- **Categóricas** (is_peak_season, region_*): Chi-squared test.

El reporte incluye:

- **Data Drift**: Proporción de features con drift significativo.
- **Data Summary**: Estadísticas descriptivas de cada feature
  comparando referencia vs. actual.

### 3.3. Interpretación para el TFM

El drift puede indicar:

- **Drift temporal**: Cambios estacionales en las features climáticas.
  Esperado y no necesariamente problemático si el modelo fue entrenado
  con suficiente variabilidad temporal.
- **Drift geográfico**: Sesgo en los municipios consultados (e.g.,
  consultas solo de Sudeste vs. entrenamiento con todo Brasil).
- **Drift de concepto**: Cambio en la relación features-target, que
  requeriría reentrenamiento. Detectable solo con labels reales.

### 3.4. Demo offline de monitoreo

El script `scripts/monitoring_demo.py` permite generar reportes de drift
sin depender de la API desplegada. Puede operar con datos históricos
(parquet de producción) o con features sintéticas con drift configurable,
lo que resulta útil para demos, testing y validación del pipeline de
monitoreo de forma reproducible.

---

## 4. Containerización: Docker

### 4.1. Estrategia multi-imagen

Se crearon dos Dockerfiles especializados:

- `Dockerfile.api`: Imagen para la API FastAPI (~500MB).
  Incluye el artefacto del modelo embebido. Imagen auto-contenida
  que no requiere acceso a MLflow.
- `Dockerfile.dashboard`: Imagen para Streamlit (~500MB).
  Se conecta a la API vía HTTP, incluye artefactos visuales.

### 4.2. Preparación de datos local

Antes de levantar los contenedores, es necesario disponer del modelo
champion y los datos. El script `scripts/setup_data.py` automatiza
esta descarga:

```bash
# Mínimo: modelo champion (GitHub Releases) + GeoJSON (IBGE)
python scripts/setup_data.py

# Quick start con datos del año actual para el dashboard
python scripts/setup_data.py --latest

# Dataset completo histórico (~1.5 GB, requiere API key)
python scripts/setup_data.py --all
```

### 4.3. Docker Compose

Para desarrollo local, `docker-compose.yml` orquesta ambos servicios
con health checks y dependencias. El servicio MLflow UI es opcional
(perfil `full`).

---

## 5. CI/CD: GitHub Actions

### 5.1. Pipeline de Integración Continua

El workflow `ci.yml` se ejecuta en cada push y pull request:

1. **Lint**: `flake8` verifica estilo de código en `src/` y `app/`.
2. **Tests**: `pytest` ejecuta los 26+ unit tests con coverage.
3. **Docker Build**: Construye ambas imágenes y verifica que la API
   arranca correctamente (health check).

### 5.2. Pipeline de Despliegue Continuo

El workflow `deploy.yml` se activa al crear un tag de release (`v*`):

1. **Build**: Construye las imágenes Docker.
2. **Push**: Las sube a Amazon ECR.
3. **Deploy**: Actualiza los servicios ECS para que usen la nueva imagen.

Este flujo asegura que cada release es reproducible y trazable.

---

## 6. Despliegue en AWS

### 6.1. Servicios utilizados

| Servicio | Rol en la arquitectura |
| --- | --- |
| ECR | Almacenamiento de imágenes Docker |
| ECS/Fargate (o EC2) | Ejecución de contenedores |
| S3 | Almacenamiento de artefactos del modelo |
| CloudWatch | Logs de los contenedores |
| GitHub Actions | Orquestación del CI/CD |

### 6.2. Consideraciones para AWS Academy

- **Limitación de IAM**: No se pueden crear roles custom. Se usa
  `LabRole` para todos los servicios.
- **Región**: Restringido a `us-east-1` o `us-west-2`.
- **Instancias**: Máximo t2.medium/t2.large para EC2.
- **Presupuesto**: Importante detener recursos cuando no se usen.
  EC2 t2.medium cuesta ~$0.046/hora ≈ $1.10/día.

### 6.3. Alternativa simplificada

Para la demo del TFM, la opción más práctica es:

1. Levantar Docker Compose localmente.
2. Grabar un screencast de la demo (API + Dashboard + Monitoreo).
3. Mostrar el deployment en EC2 como evidencia de competencia cloud.
4. Mostrar los workflows de GitHub Actions como evidencia de CI/CD.

---

## 7. Pipeline MLOps Completo — Diagrama para la Presentación

```text
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│  Data     │───▶│ Feature   │───▶│  Model    │───▶│  Deploy   │───▶│  Monitor  │
│ Ingest    │    │  Engin.   │    │ Training  │    │  (API)    │    │  (Drift)  │
│           │    │           │    │           │    │           │    │           │
│ API       │    │ 15 feats  │    │  MLflow   │    │ FastAPI   │    │ Evidently │
│ mosqlim.  │    │ no leak   │    │ Registry  │    │  Docker   │    │ Reports   │
└───────────┘    └───────────┘    └───────────┘    └───────────┘    └───────────┘
      │                │                │                │                │
      ▼                ▼                ▼                ▼                ▼
   Parquet        15 features       Champion          Predict        Drift HTML
   raw data        validated         XGBoost          endpoint         reports
                   (97% cov)      (macro_f1=0.39)    (Swagger)        (batch)
```

**Herramientas por fase:**

| Fase | Herramientas | Artefactos |
| --- | --- | --- |
| Data Ingestion | Python, requests, mosqlimate API | Parquet files |
| Feature Engineering | pandas, numpy, pytest (26 tests) | Production dataset |
| Modeling | XGBoost, Optuna, MLflow | Champion model, experiment logs |
| Deployment | FastAPI, Docker, GitHub Actions, AWS | Container images, API docs |
| Monitoring | Evidently, Streamlit | Drift reports, dashboard |

---

## 8. Estructura de archivos generada

```text
app/
├── __init__.py
├── api.py                 # FastAPI REST API
├── schemas.py             # Pydantic models
└── streamlit_app.py       # Dashboard Streamlit

src/monitoring/
├── __init__.py
└── drift_detector.py      # Evidently drift detection

monitoring/
├── predictions_log.csv    # Log de predicciones (generado)
└── reports/               # Reportes HTML de drift (generados)

.github/workflows/
├── ci.yml                 # CI: tests + lint + docker build
└── deploy.yml             # CD: push ECR + deploy ECS

docs/deployment/
├── architecture.md        # Diagrama de arquitectura
├── aws_deployment.md      # Guía paso a paso AWS
└── deployment_technical.md # Este documento (contenido para TFM)

Dockerfile.api             # Imagen Docker para API
Dockerfile.dashboard       # Imagen Docker para Dashboard
docker-compose.yml         # Orquestación local
.dockerignore             # Exclusiones Docker
requirements-prod.txt     # Deps mínimas de producción
```
