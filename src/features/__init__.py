"""
Feature engineering module for dengue prediction.

Main components:
- DengueFeatureEngineer: Pipeline completo de feature engineering
- Constants: Clasificación de variables y features seleccionadas

Usage:
    from features import DengueFeatureEngineer

    fe = DengueFeatureEngineer()
    df_features = fe.transform(df_raw)
    df_production = fe.get_production_dataset(df_features)
"""

from .feature_engineer import DengueFeatureEngineer
from .constants import (
    TARGET, TEMPORAL_VARS, GEO_VARS, CLIMATE_VARS, LEAKAGE_VARS,
    REGION_MAP, REGIONS, ALL_ENGINEERED_FEATURES,
    TEMPORAL_FEATURES, CLIMATE_FEATURES, GEO_FEATURES,
    PRODUCTION_COLUMNS,
)

__all__ = [
    'DengueFeatureEngineer',
    'TARGET',
    'TEMPORAL_VARS',
    'GEO_VARS',
    'CLIMATE_VARS',
    'LEAKAGE_VARS',
    'REGION_MAP',
    'REGIONS',
    'ALL_ENGINEERED_FEATURES',
    'TEMPORAL_FEATURES',
    'CLIMATE_FEATURES',
    'GEO_FEATURES',
    'PRODUCTION_COLUMNS',
]
