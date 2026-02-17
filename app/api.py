"""
FastAPI REST API para predicción de dengue.
Sirve el modelo champion desde MLflow Model Registry.
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
import mlflow
import mlflow.xgboost  # type: ignore[attr-defined]
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Asegurar que el proyecto está en el path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import (  # noqa: E402
    PredictionInput, PredictionResult, PredictionResponse,
    BatchInput, BatchResponse,
    ModelInfo, HealthResponse,
    FEATURE_NAMES, ALERT_LABELS, ALERT_COLORS,
)
from src.models.constants import LABEL_MAP_INV  # noqa: E402

logger = logging.getLogger("dengue-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# --- Estado global ---
_state = {
    "model": None,
    "model_version": "unknown",
    "model_loaded": False,
    "load_time": None,
    "predictions_log": [],  # buffer para monitoreo
}

# Config
MODEL_ARTIFACT_PATH = os.getenv(
    "MODEL_ARTIFACT_PATH",
    str(PROJECT_ROOT / "mlflow-artifacts" / "models"
        / "m-81d328499d7d4eeaa187f9c052124f62" / "artifacts")
)
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    (PROJECT_ROOT / "mlruns").as_uri()
)
REGISTERED_MODEL_NAME = "dengue-alertlevel-classifier"
MODEL_ALIAS = "champion"
PREDICTIONS_LOG_PATH = PROJECT_ROOT / "monitoring" / "predictions_log.csv"


def load_model():
    """
    Carga el modelo champion. Intenta primero desde el registry,
    y si no está disponible, carga directamente del artefacto local.
    """
    # Opción 1: desde MLflow registry
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.xgboost.load_model(model_uri)  # type: ignore[attr-defined]
        _state["model"] = model
        _state["model_version"] = "1"  # champion es v1
        _state["model_loaded"] = True
        _state["load_time"] = datetime.utcnow().isoformat()
        logger.info(f"Modelo cargado desde registry: {model_uri}")
        return
    except Exception as e:
        logger.warning(f"No se pudo cargar desde registry: {e}")

    # Opción 2: artefacto local directo
    try:
        model = mlflow.xgboost.load_model(MODEL_ARTIFACT_PATH)  # type: ignore[attr-defined]
        _state["model"] = model
        _state["model_version"] = "1-local"
        _state["model_loaded"] = True
        _state["load_time"] = datetime.utcnow().isoformat()
        logger.info(f"Modelo cargado desde artefacto local: {MODEL_ARTIFACT_PATH}")
        return
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise RuntimeError(f"No se pudo cargar el modelo: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga modelo al iniciar y limpia al cerrar."""
    load_model()
    yield
    _state["model"] = None
    logger.info("API shutdown, modelo liberado")


# --- App ---
app = FastAPI(
    title="Dengue Alert Level Prediction API",
    description=(
        "API REST para predicción del nivel de alerta de dengue (1-4) "
        "en municipios de Brasil. Modelo XGBoost optimizado con Optuna, "
        "15 features de producción sin target leakage."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helpers ---

def _predict_single(features: dict) -> PredictionResult:
    """Ejecuta predicción para una instancia."""
    model = _state["model"]
    df = pd.DataFrame([features], columns=FEATURE_NAMES)

    # Predicción
    y_pred_raw = model.predict(df)[0]
    nivel = int(LABEL_MAP_INV.get(int(y_pred_raw), int(y_pred_raw)))

    # Probabilidades
    proba = model.predict_proba(df)[0]
    probabilities = {
        ALERT_LABELS[i + 1]: round(float(p), 4) for i, p in enumerate(proba)
    }

    return PredictionResult(
        nivel=nivel,
        label=ALERT_LABELS[nivel],
        color=ALERT_COLORS[nivel],
        probabilities=probabilities,
    )


def _log_prediction(features: dict, result: PredictionResult):
    """Almacena predicción para monitoreo (Evidently)."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "predicted_nivel": result.nivel,
        **features,
    }
    _state["predictions_log"].append(record)

    # Flush a disco cada 100 predicciones
    if len(_state["predictions_log"]) >= 100:
        _flush_predictions()


def _flush_predictions():
    """Escribe el buffer de predicciones a CSV."""
    if not _state["predictions_log"]:
        return
    PREDICTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(_state["predictions_log"])
    write_header = not PREDICTIONS_LOG_PATH.exists()
    df.to_csv(PREDICTIONS_LOG_PATH, mode="a", header=write_header, index=False)
    _state["predictions_log"] = []
    logger.info(f"Flushed {len(df)} predicciones a {PREDICTIONS_LOG_PATH}")


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Estado del servicio y disponibilidad del modelo."""
    return HealthResponse(
        status="healthy" if _state["model_loaded"] else "degraded",
        model_loaded=_state["model_loaded"],
        version=_state["model_version"],
    )


@app.get("/model/info", response_model=ModelInfo, tags=["model"])
async def model_info():
    """Metadata del modelo champion en producción."""
    if not _state["model_loaded"]:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    return ModelInfo(
        name=REGISTERED_MODEL_NAME,
        version=_state["model_version"],
        alias=MODEL_ALIAS,
        algorithm="XGBClassifier",
        n_features=len(FEATURE_NAMES),
        features=FEATURE_NAMES,
        target="nivel",
        classes={k: v for k, v in ALERT_LABELS.items()},
        metrics={
            "macro_f1": 0.39,
            "accuracy": 0.88,
            "cohen_kappa": 0.33,
        },
        training_period="2010-2021",
        description=(
            "Clasificador de nivel de alerta de dengue (1-4) para municipios "
            "de Brasil. XGBoost con balanced sample weights, optimizado con "
            "Optuna (40 trials). 15 features sin target leakage."
        ),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(input_data: PredictionInput):
    """
    Predicción individual del nivel de alerta de dengue.

    Recibe las 15 features de producción y devuelve el nivel predicho
    (1-4) con probabilidades por clase.
    """
    if not _state["model_loaded"]:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    features = input_data.to_feature_dict()
    result = _predict_single(features)
    _log_prediction(features, result)

    return PredictionResponse(
        prediction=result,
        model_version=_state["model_version"],
        features_used=len(FEATURE_NAMES),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
async def predict_batch(batch: BatchInput):
    """
    Predicción por lotes. Máximo 1000 instancias por request.
    """
    if not _state["model_loaded"]:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    results = []
    for instance in batch.instances:
        features = instance.to_feature_dict()
        result = _predict_single(features)
        _log_prediction(features, result)
        results.append(result)

    return BatchResponse(
        predictions=results,
        count=len(results),
        model_version=_state["model_version"],
    )


@app.post("/monitoring/flush", tags=["monitoring"])
async def flush_predictions():
    """Fuerza el flush del buffer de predicciones a disco."""
    count = len(_state["predictions_log"])
    _flush_predictions()
    return {"flushed": count, "path": str(PREDICTIONS_LOG_PATH)}


# --- Entrypoint ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
