"""
Script para generar predicciones de prueba y reporte Evidently.
Ejecutar con la API corriendo: python scripts/generate_drift_report.py
"""

import sys
import json
import random
import logging
from pathlib import Path

import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"

# Rangos realistas para generar predicciones variadas
FEATURE_RANGES = {
    "month_sin": (-1.0, 1.0),
    "month_cos": (-1.0, 1.0),
    "se_sin": (-1.0, 1.0),
    "se_cos": (-1.0, 1.0),
    "is_peak_season": (0, 1),
    "quarter": (1, 4),
    "tempmed_lag8w": (18.0, 32.0),
    "tempmed_roll12w": (18.0, 30.0),
    "umidmed_roll4w": (55.0, 95.0),
    "temp_x_humid_lag4w": (1200.0, 2800.0),
    "pop_log": (8.0, 16.5),
    "region_Nordeste": (0, 1),
    "region_Centro-Oeste": (0, 1),
    "region_Sudeste": (0, 1),
    "region_Sul": (0, 1),
}


def generate_random_prediction():
    """Genera una instancia con valores realistas."""
    features = {}
    for name, (low, high) in FEATURE_RANGES.items():
        if name in ("is_peak_season", "quarter") or name.startswith("region_"):
            features[name] = random.randint(int(low), int(high))
        else:
            features[name] = round(random.uniform(low, high), 4)

    # Asegurar que solo una region sea 1 (o ninguna = Norte)
    regions = ["region_Nordeste", "region_Centro-Oeste", "region_Sudeste", "region_Sul"]
    for r in regions:
        features[r] = 0
    if random.random() > 0.2:  # 80% de las veces, asignar una region
        features[random.choice(regions)] = 1

    return features


def check_api():
    """Verifica que la API esté accesible."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        data = r.json()
        if data.get("model_loaded"):
            logger.info(f"API healthy, modelo v{data.get('version')}")
            return True
        else:
            logger.warning("API en modo degraded (sin modelo)")
            return False
    except Exception as e:
        logger.error(f"API no accesible: {e}")
        return False


def generate_predictions(n=150):
    """Genera N predicciones variadas."""
    logger.info(f"Generando {n} predicciones...")
    results = []
    for i in range(n):
        features = generate_random_prediction()
        try:
            r = requests.post(f"{API_URL}/predict", json=features, timeout=10)
            if r.status_code == 200:
                results.append(r.json())
            else:
                logger.warning(f"Pred {i}: status {r.status_code}")
        except Exception as e:
            logger.warning(f"Pred {i}: {e}")

    logger.info(f"Generadas {len(results)} predicciones exitosas")
    return results


def flush_predictions():
    """Fuerza flush del buffer de la API a disco."""
    try:
        r = requests.post(f"{API_URL}/monitoring/flush", timeout=10)
        data = r.json()
        logger.info(f"Flush: {data}")
        return data
    except Exception as e:
        logger.error(f"Error en flush: {e}")
        return None


def generate_drift_report():
    """Genera el reporte de drift con Evidently."""
    from src.monitoring.drift_detector import DriftDetector
    from src.data.mosqlimate_loader import MosqlimateDataLoader
    from src.features.feature_engineer import DengueFeatureEngineer

    log_path = PROJECT_ROOT / "monitoring" / "predictions_log.csv"
    if not log_path.exists():
        logger.error(f"No existe {log_path}")
        return None

    df_log = pd.read_csv(log_path)
    logger.info(f"Predicciones en log: {len(df_log)}")

    # Dataset de referencia con ruta absoluta
    data_path = str(PROJECT_ROOT / "data" / "raw" / "historical_api_data")
    logger.info("Construyendo dataset de referencia (esto puede tardar ~60s)...")

    loader = MosqlimateDataLoader(data_path=data_path)
    df_raw = loader.load_all_states_data()
    # NO llamar standardize_columns — el feature engineer espera nombres originales
    fe = DengueFeatureEngineer()
    df_features = fe.transform(df_raw)
    ref_df = fe.get_production_dataset(df_features)
    ref_df = ref_df[ref_df["year"] <= 2021].copy()

    logger.info(f"Referencia: {len(ref_df)} registros")

    # Detector
    detector = DriftDetector(ref_df)

    # Generar reporte guardado en monitoring/reports/
    result = detector.detect(df_log, report_name="drift_report")

    # Copiar el reporte a la ubicacion que espera la captura
    report_src = Path(result["report_path"])
    report_dst = PROJECT_ROOT / "monitoring" / "drift_report.html"
    if report_src != report_dst:
        import shutil
        shutil.copy2(report_src, report_dst)

    logger.info(f"Reporte HTML: {result['report_path']}")
    logger.info(f"Features analizadas: {result['features_analyzed']}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=int, default=150, help="Numero de predicciones")
    parser.add_argument("--skip-predictions", action="store_true", help="Solo generar reporte")
    args = parser.parse_args()

    if not args.skip_predictions:
        if not check_api():
            print("\n❌ API no disponible. Arrancala con:")
            print("   docker compose up -d")
            print("   # o directamente:")
            print("   python -m uvicorn app.api:app --port 8000")
            sys.exit(1)

        generate_predictions(args.predictions)
        flush_predictions()

    print("\n--- Generando reporte de drift ---")
    result = generate_drift_report()
    if result:
        print(f"\n✅ Reporte generado: {result['report_path']}")
        print("Abre el HTML en un navegador y toma la captura para figs/evidently_drift_report.png")
