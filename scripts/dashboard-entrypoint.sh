#!/bin/bash
# Entrypoint para el dashboard en ECS/Fargate
# Descarga datos desde S3 si S3_DATA_BUCKET está configurado
set -e

if [ -n "$S3_DATA_BUCKET" ]; then
    echo "[entrypoint] Descargando datos desde S3..."
    mkdir -p /app/data/raw/historical_api_data /app/data/external

    aws s3 sync "s3://$S3_DATA_BUCKET/raw/historical_api_data/" \
        /app/data/raw/historical_api_data/ --quiet

    aws s3 cp "s3://$S3_DATA_BUCKET/external/brazil_uf.geojson" \
        /app/data/external/brazil_uf.geojson --quiet

    echo "[entrypoint] Datos descargados"
fi

exec streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
