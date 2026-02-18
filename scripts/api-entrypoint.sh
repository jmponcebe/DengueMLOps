#!/bin/bash
# Entrypoint para la API en ECS/Fargate
# Descarga modelo desde S3 si S3_DATA_BUCKET está configurado
set -e

if [ -n "$S3_DATA_BUCKET" ]; then
    echo "[entrypoint] Descargando modelo desde S3..."
    mkdir -p /app/mlflow-artifacts/models/champion/artifacts

    aws s3 sync "s3://$S3_DATA_BUCKET/models/champion/" \
        /app/mlflow-artifacts/models/champion/ --quiet

    # Apuntar MODEL_ARTIFACT_PATH al modelo descargado
    export MODEL_ARTIFACT_PATH=/app/mlflow-artifacts/models/champion/artifacts
    echo "[entrypoint] Modelo descargado en $MODEL_ARTIFACT_PATH"
fi

exec uvicorn app.api:app --host 0.0.0.0 --port 8000
