# Arquitectura de Despliegue — MLOps Dengue Prediction

## Visión General

El sistema de producción sigue una arquitectura de microservicios containerizados, orquestados con Docker Compose en desarrollo local y desplegados en AWS (ECR + ECS/Fargate) para producción.

```text
┌───────────────────────────────────────────────────────────────────────┐
│                           GitHub Repository                           │
│  ┌────────────┐  push  ┌────────────────────┐ deploy ┌─────────────┐  │
│  │ Developer  │───────▶│   GitHub Actions   │───────▶│     AWS     │  │
│  └────────────┘        │   CI/CD Pipeline   │        │  ECR / ECS  │  │
│                        │   - tests          │        └──────┬──────┘  │
│                        │   - lint           │               │         │
│                        │   - docker build   │               │         │
│                        │   - push ECR       │               ▼         │
│                        └────────────────────┘        ┌─────────────┐  │
│                                                      │   Fargate   │  │
│                                                      │   Cluster   │  │
│                                                      └─────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
```

## Componentes del Sistema

### 1. API REST (FastAPI) — Puerto 8000

Servicio de predicción que expone el modelo champion vía HTTP.

**Endpoints:**

| Endpoint | Método | Descripción |
| --- | --- | --- |
| `/health` | GET | Health check del servicio |
| `/model/info` | GET | Metadata del modelo (versión, métricas, features) |
| `/predict` | POST | Predicción individual |
| `/predict/batch` | POST | Predicción por lotes |

**Responsabilidades:**

- Cargar modelo champion desde MLflow Model Registry
- Validar input con Pydantic schemas
- Decodificar predicciones (0-3 → nivel 1-4)
- Loguear predicciones para monitoreo posterior

### 2. Dashboard (Streamlit) — Puerto 8501

Interfaz visual para consulta de predicciones y monitoreo.

**Páginas:**

- **Mapa de Brasil**: Visualización geoespacial del nivel de alerta por municipio
- **Predicción**: Formulario interactivo para consulta individual
- **Modelo**: Métricas del champion, feature importance, matriz de confusión
- **Monitoreo**: Reportes de data drift (Evidently)

### 3. Monitoreo (Evidently) — Batch

Sistema de detección de data drift sobre las predicciones en producción.

**Estrategia:**

- **Reference dataset**: Datos de entrenamiento (2010-2021)
- **Current dataset**: Predicciones recientes (ventana deslizante)
- **Reportes**: Data Drift, Data Summary
- **Frecuencia**: Bajo demanda o programado (cron/EventBridge)
- **Demo offline**: `scripts/monitoring_demo.py` genera reportes sin API, usando datos históricos o features sintéticas con drift configurable

### 4. MLflow Tracking

Gestión del ciclo de vida del modelo.

- **Model Registry**: `dengue-alertlevel-classifier` con alias `champion`
- **Tracking local**: `mlruns/` (desarrollo)
- **Tracking remoto**: S3 + RDS (producción, opcional)

## Infraestructura

### Desarrollo Local (Docker Compose)

```yaml
services:
  api:        # FastAPI en puerto 8000
  dashboard:  # Streamlit en puerto 8501
  mlflow:     # MLflow UI en puerto 5000 (opcional)
```

### Producción (AWS)

```text
┌──────────────────────────────────────────────────────────┐
│                   AWS Cloud (us-east-1)                  │
│                                                          │
│  ┌────────────┐      ┌──────────────────────────────┐    │
│  │    ECR     │      │       ECS / Fargate          │    │
│  │ (imagenes  │─────▶│  ┌───────┐  ┌─────────────┐  │    │
│  │  Docker)   │      │  │  API  │  │  Dashboard  │  │    │
│  └────────────┘      │  └───┬───┘  └──────┬──────┘  │    │
│                      │      │             │         │    │
│  ┌────────────┐      │      ▼             ▼         │    │
│  │    S3      │      │  ┌────────────────────────┐  │    │
│  │ (artifacts │◀─────│  │    Model Artifacts     │  │    │
│  │  data)     │      │  └────────────────────────┘  │    │
│  └────────────┘      └──────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  ALB (Application Load Balancer) - opcional        │  │
│  │  api.domain:8000 / dashboard.domain:8501           │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**Servicios AWS utilizados:**

| Servicio | Propósito | Capa gratuita Academy |
| --- | --- | --- |
| ECR | Registro de imágenes Docker | Sí |
| ECS + Fargate | Orquestación de contenedores | Sí |
| S3 | Almacenamiento de artefactos y datos | Sí |
| EC2 (alternativa) | Hosting directo con docker-compose | Sí (hasta 9 instancias) |
| CloudFormation | IaC para despliegue reproducible | Sí |

### CI/CD (GitHub Actions)

```text
push to main
     │
     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Run Tests    │────▶│ Lint + Type   │────▶│ Build Docker  │
│  pytest       │     │  Check        │     │  Images       │
└───────────────┘     └───────────────┘     └───────────────┘
                                                    │
                                                    ▼
                                            ┌───────────────┐
                                            │ Push to ECR   │
                                            │  (si main)    │
                                            └───────────────┘
                                                    │
                                                    ▼
                                            ┌───────────────┐
                                            │ Deploy ECS    │
                                            │  (si tag)     │
                                            └───────────────┘
```

**Workflows:**

1. **CI** (`ci.yml`): Tests + lint en cada push/PR
2. **CD** (`deploy.yml`): Build Docker → Push ECR → Update ECS (solo en main/tags)

## Reproducibilidad Local

Para ejecutar el sistema localmente, `scripts/setup_data.py` es el punto de entrada
para descargar los artefactos necesarios:

| Modo | Comando | Qué descarga | API key |
| --- | --- | --- | --- |
| Default | `python scripts/setup_data.py` | Modelo champion + GeoJSON | No |
| Quick start | `--latest` | Modelo + GeoJSON + datos año actual | Sí |
| Rango | `--data --from 2020 --to 2024` | Datos de un rango de años | Sí |
| Completo | `--all` | Todo (histórico 2010-presente, ~1.5 GB) | Sí |

El modelo champion se obtiene desde [GitHub Releases v1.0.0](https://github.com/jmponcebe/DengueMLOps/releases/tag/v1.0.0).
En AWS, los contenedores descargan datos y modelo desde S3 vía entrypoint scripts.

## Flujo de Datos en Producción

```text
1. Usuario/sistema envía request → API /predict
2. API valida input (Pydantic) → carga features
3. Modelo XGBoost predice → decode labels (0-3 → 1-4)
4. Respuesta JSON con nivel de alerta + probabilidades
5. Predicción se loguea para monitoreo
6. Evidently compara distribución vs. referencia (batch)
7. Dashboard Streamlit muestra resultados + drift reports
```

## Decisiones de Diseño

### ¿Por qué FastAPI y no Flask?

- Validación automática con Pydantic (type safety)
- Documentación OpenAPI auto-generada (Swagger UI)
- Rendimiento async nativo
- Estándar en MLOps moderno

### ¿Por qué ECS/Fargate y no Lambda?

- El modelo XGBoost (~16MB) tiene cold start significativo en Lambda
- Fargate mantiene el contenedor caliente
- Más natural para API stateful con modelo en memoria
- Alternativa viable: EC2 con docker-compose (más simple para TFM)

### ¿Por qué Evidently batch y no real-time?

- El dominio epidemiológico opera en escala semanal (semanas epidemiológicas)
- No hay necesidad de drift detection en tiempo real
- Reportes HTML ricos para incluir en la memoria del TFM
- Menor complejidad operacional

### ¿Por qué no MLflow Serving?

- MLflow serving (mlflow models serve) es una opción válida
- FastAPI proporciona más control sobre la lógica de negocio
- Permite integrar validación, logging, y monitoreo custom
- Más representativo de un sistema de producción real

## Modelo Champion

| Propiedad | Valor |
| --- | --- |
| Algoritmo | XGBoost (XGBClassifier) |
| Nombre registro | `dengue-alertlevel-classifier` |
| Alias | `champion` |
| Features | 15 (sin target leakage) |
| Target | `nivel` (1-4, alerta epidemiológica) |
| Métrica principal | macro_f1 = 0.39 |
| Kappa | 0.33 |
| Optimización | Optuna (40 trials) + balanced sample weights |
| Split temporal | Train 2010-2021, Val 2022-2023, Test 2024 |
