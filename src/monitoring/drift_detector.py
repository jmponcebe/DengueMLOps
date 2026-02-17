"""
Detector de data drift para el modelo de dengue.
Usa Evidently para comparar la distribución de features
en producción vs. los datos de referencia (entrenamiento).

Genera reportes HTML para el dashboard y logs para alertas.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Features numéricas y categóricas del modelo
NUMERICAL_FEATURES = [
    "month_sin", "month_cos", "se_sin", "se_cos",
    "tempmed_lag8w", "tempmed_roll12w", "umidmed_roll4w", "temp_x_humid_lag4w",
    "pop_log",
]
CATEGORICAL_FEATURES = [
    "is_peak_season", "quarter",
    "region_Nordeste", "region_Centro-Oeste", "region_Sudeste", "region_Sul",
]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
PREDICTION_COL = "predicted_nivel"


class DriftDetector:
    """
    Detecta data drift comparando datos de producción contra referencia.
    
    Genera reportes Evidently en HTML para incluir en el dashboard
    Streamlit y para la documentación del TFM.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        reports_dir: Optional[str] = None,
    ):
        self.reports_dir = Path(reports_dir or PROJECT_ROOT / "monitoring" / "reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Solo columnas disponibles en referencia y en producción
        available = [c for c in ALL_FEATURES if c in reference_data.columns]
        self.features = available

        num_cols = [c for c in NUMERICAL_FEATURES if c in available]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in available]

        self.data_def = DataDefinition(
            numerical_columns=num_cols,
            categorical_columns=cat_cols,
        )
        self._n_reference = len(reference_data)
        self.reference_ds = Dataset.from_pandas(
            reference_data[available].copy(),
            data_definition=self.data_def,
        )
        logger.info(
            f"DriftDetector inicializado con {len(available)} features, "
            f"{self._n_reference} registros de referencia"
        )

    def detect(
        self,
        current_data: pd.DataFrame,
        report_name: Optional[str] = None,
    ) -> dict:
        """
        Ejecuta detección de drift y genera reporte HTML.

        Args:
            current_data: DataFrame con datos de producción (mismas columnas)
            report_name: nombre del archivo HTML (default: timestamp)

        Returns:
            dict con resultados: drift detectado, path del reporte, stats
        """
        available = [c for c in self.features if c in current_data.columns]
        current_ds = Dataset.from_pandas(
            current_data[available].copy(),
            data_definition=self.data_def,
        )

        # Reporte de drift
        report = Report([
            DataDriftPreset(),
            DataSummaryPreset(),
        ])
        snapshot = report.run(
            reference_data=self.reference_ds,
            current_data=current_ds,
        )

        # Guardar HTML
        if report_name is None:
            report_name = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_path = self.reports_dir / f"{report_name}.html"
        snapshot.save_html(str(report_path))

        logger.info(f"Reporte de drift guardado en: {report_path}")

        return {
            "report_path": str(report_path),
            "n_reference": self._n_reference,
            "n_current": len(current_data),
            "features_analyzed": len(available),
            "timestamp": datetime.now().isoformat(),
        }

    def detect_from_log(
        self,
        log_path: Optional[str] = None,
        last_n: int = 500,
    ) -> dict:
        """
        Detecta drift desde el CSV de predicciones logueadas por la API.

        Args:
            log_path: ruta al CSV de predicciones
            last_n: usar las últimas N predicciones
        """
        if log_path is None:
            log_path = str(PROJECT_ROOT / "monitoring" / "predictions_log.csv")

        if not Path(log_path).exists():
            raise FileNotFoundError(
                f"No existe log de predicciones en {log_path}. "
                "Ejecuta la API y haz algunas predicciones primero."
            )

        df = pd.read_csv(log_path)
        if len(df) > last_n:
            df = df.tail(last_n)

        return self.detect(df, report_name=f"drift_from_api_log_{len(df)}")


def build_reference_dataset(
    data_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Construye el dataset de referencia a partir de los datos de entrenamiento.
    Aplica el feature engineering para obtener las 15 features.
    """
    if data_path and Path(data_path).exists():
        return pd.read_parquet(data_path)

    # Intentar construir desde raw
    from src.data.mosqlimate_loader import MosqlimateDataLoader, DataProcessor
    from src.features.feature_engineer import DengueFeatureEngineer

    loader = MosqlimateDataLoader()
    df_raw = loader.load_all_states_data()
    df_raw = DataProcessor.standardize_columns(df_raw)

    fe = DengueFeatureEngineer()
    df_features = fe.transform(df_raw)
    df_prod = fe.get_production_dataset(df_features)

    # Solo período de entrenamiento
    df_ref = df_prod[df_prod["year"] <= 2021].copy()
    return df_ref


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detectar data drift en predicciones")
    parser.add_argument("--reference", type=str, default=None, help="Path al dataset de referencia (parquet)")
    parser.add_argument("--current", type=str, default=None, help="Path al CSV de predicciones actuales")
    parser.add_argument("--last-n", type=int, default=500, help="Últimas N predicciones a analizar")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print("Construyendo dataset de referencia...")
    ref_df = build_reference_dataset(args.reference)
    print(f"Referencia: {len(ref_df)} registros, {ref_df.shape[1]} columnas")

    detector = DriftDetector(ref_df)

    if args.current:
        current_df = pd.read_csv(args.current)
        results = detector.detect(current_df)
    else:
        results = detector.detect_from_log(last_n=args.last_n)

    print(f"\nReporte: {results['report_path']}")
    print(f"Features analizadas: {results['features_analyzed']}")
