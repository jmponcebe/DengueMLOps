"""
Feature engineering pipeline for dengue prediction.
Extracts production-safe features from raw data without target leakage.

Key principles:
- Climate variables only available with 4-8 week lag (vector lifecycle)
- shift() before rolling to exclude current observation
- Region encoding by domain knowledge, not target encoding
- All epidemiological/current-period variables excluded from production
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict

from .constants import (
    TARGET, TEMPORAL_VARS, GEO_VARS, CLIMATE_VARS, LEAKAGE_VARS,
    REGION_MAP, REGIONS, ALL_ENGINEERED_FEATURES,
    TEMPORAL_FEATURES, CLIMATE_FEATURES, GEO_FEATURES,
    CLIMATE_BASE_VARS, LAG_PERIODS, ROLLING_WINDOWS,
    PRODUCTION_ID_COLS, PRODUCTION_COLUMNS,
)

logger = logging.getLogger(__name__)


class DengueFeatureEngineer:
    """
    Pipeline de feature engineering para predicción de dengue.
    
    Genera features temporales (encoding cíclico), climáticas (lags + rolling)
    y geográficas (log población + macro-región) a partir del dataset crudo.
    
    Usage:
        fe = DengueFeatureEngineer()
        df_features = fe.transform(df_raw)
        df_production = fe.get_production_dataset(df_features)
    """

    def __init__(self):
        self.region_map = REGION_MAP
        self.regions = REGIONS
        self.engineered_features = ALL_ENGINEERED_FEATURES

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el pipeline completo de feature engineering.
        
        Args:
            df: DataFrame con columnas originales del dataset dengue.
                Requiere: month, SE, pop, uf, municipio_geocodigo, data_iniSE,
                          tempmed, umidmed, tempmin, tempmax
        
        Returns:
            DataFrame con todas las columnas originales + features engineered.
        """
        df_out = df.copy()

        self._validate_input(df_out)

        df_out = self._add_temporal_features(df_out)
        df_out = self._add_climate_features(df_out)
        df_out = self._add_geographic_features(df_out)

        logger.info(
            f"Feature engineering completado: {len(self.engineered_features)} features, "
            f"{df_out.shape[0]:,} registros"
        )
        return df_out

    def get_production_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra a columnas seguras para producción (sin leakage).
        
        Args:
            df: DataFrame con features engineered (output de transform()).
        
        Returns:
            DataFrame con solo columnas de producción.
        """
        available = [c for c in PRODUCTION_COLUMNS if c in df.columns]
        missing = [c for c in PRODUCTION_COLUMNS if c not in df.columns]
        if missing:
            logger.warning(f"Columnas faltantes en producción: {missing}")

        result = df[available].copy()

        # Verificación de seguridad
        leakage = [c for c in result.columns if c in LEAKAGE_VARS + CLIMATE_VARS]
        if leakage:
            raise ValueError(f"Target leakage detectado en producción: {leakage}")

        return result

    def get_feature_names(self) -> List[str]:
        """Features engineered disponibles."""
        return list(self.engineered_features)

    # --- Features temporales ---

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encoding cíclico de mes/SE + indicador de temporada pico + trimestre."""
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['se_sin'] = np.sin(2 * np.pi * df['SE'] / 52)
        df['se_cos'] = np.cos(2 * np.pi * df['SE'] / 52)
        df['is_peak_season'] = (df['month'].isin([1, 2, 3, 4])).astype(int)
        df['quarter'] = ((df['month'] - 1) // 3) + 1
        return df

    # --- Features climáticas con lag ---

    def _add_climate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lags climáticos, rolling means con shift, e interacciones."""
        # Ordenar por municipio y fecha para lags correctos
        df['date'] = pd.to_datetime(df['data_iniSE'], errors='coerce')
        df = df.sort_values(['municipio_geocodigo', 'date']).reset_index(drop=True)

        # Lags
        for var in CLIMATE_BASE_VARS:
            for lag in LAG_PERIODS:
                df[f'{var}_lag{lag}w'] = df.groupby('municipio_geocodigo')[var].shift(lag)

        # Rolling con shift(1) para excluir observación actual
        for var in CLIMATE_BASE_VARS:
            for window in ROLLING_WINDOWS:
                df[f'{var}_roll{window}w'] = df.groupby('municipio_geocodigo')[var].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=2).mean()
                )

        # Interacciones
        df['temp_x_humid_lag4w'] = df['tempmed_lag4w'] * df['umidmed_lag4w']
        df['temp_range_lag4w'] = (
            df.groupby('municipio_geocodigo')['tempmax'].shift(4) -
            df.groupby('municipio_geocodigo')['tempmin'].shift(4)
        )

        df = df.drop(columns=['date'], errors='ignore')
        return df

    # --- Features geográficas ---

    def _add_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log de población + one-hot encoding de macro-región."""
        df['pop_log'] = np.log1p(df['pop'])

        df['region'] = pd.Categorical(
            df['uf'].map(self.region_map), categories=self.regions
        )
        region_dummies = pd.get_dummies(df['region'], prefix='region', dtype=int)
        df = pd.concat([df, region_dummies], axis=1)
        df = df.drop(columns=['region'], errors='ignore')

        return df

    # --- Validación ---

    def _validate_input(self, df: pd.DataFrame):
        """Verifica que las columnas necesarias existen."""
        required = ['month', 'SE', 'pop', 'uf', 'municipio_geocodigo',
                     'data_iniSE', 'tempmed', 'umidmed', 'tempmin', 'tempmax']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Columnas requeridas faltantes: {missing}")

    # --- Utilidades ---

    def validate_no_leakage(self, df: pd.DataFrame) -> bool:
        """Verifica que un DataFrame no contiene variables con leakage."""
        leakage_found = [c for c in df.columns if c in LEAKAGE_VARS + CLIMATE_VARS]
        if leakage_found:
            logger.error(f"Variables con leakage: {leakage_found}")
            return False
        return True

    def get_feature_summary(self) -> Dict:
        """Resumen de features para logging/tracking."""
        return {
            'temporal': TEMPORAL_FEATURES,
            'climate': CLIMATE_FEATURES,
            'geographic': GEO_FEATURES,
            'total': len(self.engineered_features),
            'production_columns': PRODUCTION_COLUMNS,
        }
