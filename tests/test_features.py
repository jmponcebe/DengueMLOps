"""
Tests para el módulo de feature engineering.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.feature_engineer import DengueFeatureEngineer
from features.constants import (
    TARGET, LEAKAGE_VARS, CLIMATE_VARS, ALL_ENGINEERED_FEATURES,
    REGION_MAP, REGIONS, TEMPORAL_FEATURES, CLIMATE_FEATURES, GEO_FEATURES,
    PRODUCTION_COLUMNS,
)


@pytest.fixture
def sample_df():
    """DataFrame mínimo que simula la estructura del dataset dengue."""
    np.random.seed(42)
    n = 520  # ~10 años de semanas para un municipio
    dates = pd.date_range('2010-01-03', periods=n, freq='7D')

    df = pd.DataFrame({
        'data_iniSE': dates.strftime('%Y-%m-%d'),
        'SE': [d.isocalendar()[1] for d in dates],
        'year': dates.year,
        'month': dates.month,
        'municipio_geocodigo': 3304557,
        'municipio_nome': 'Rio de Janeiro',
        'pop': 6748000,
        'uf': 'RJ',
        'state_name': 'Rio de Janeiro',
        'tempmin': np.random.uniform(18, 24, n),
        'tempmed': np.random.uniform(22, 30, n),
        'tempmax': np.random.uniform(28, 36, n),
        'umidmin': np.random.uniform(50, 70, n),
        'umidmed': np.random.uniform(65, 85, n),
        'umidmax': np.random.uniform(80, 99, n),
        'nivel': np.random.choice([1, 2, 3, 4], n, p=[0.93, 0.04, 0.005, 0.025]),
        # Variables de leakage (deben eliminarse en producción)
        'casos': np.random.randint(0, 500, n),
        'casos_est': np.random.uniform(0, 500, n),
        'p_rt1': np.random.uniform(0, 1, n),
    })
    return df


@pytest.fixture
def multi_municipio_df(sample_df):
    """DataFrame con múltiples municipios para validar groupby."""
    df2 = sample_df.copy()
    df2['municipio_geocodigo'] = 3550308
    df2['municipio_nome'] = 'São Paulo'
    df2['uf'] = 'SP'
    df2['state_name'] = 'São Paulo'
    return pd.concat([sample_df, df2], ignore_index=True)


@pytest.fixture
def fe():
    return DengueFeatureEngineer()


class TestDengueFeatureEngineer:
    """Tests del pipeline de feature engineering."""

    def test_transform_adds_expected_features(self, fe, sample_df):
        """Verifica que transform() crea todas las features esperadas."""
        result = fe.transform(sample_df)
        for feat in ALL_ENGINEERED_FEATURES:
            assert feat in result.columns, f"Feature faltante: {feat}"

    def test_transform_preserves_original_columns(self, fe, sample_df):
        """Las columnas originales no se pierden."""
        original_cols = set(sample_df.columns)
        result = fe.transform(sample_df)
        for col in original_cols:
            assert col in result.columns, f"Columna original {col} perdida"

    def test_transform_row_count_preserved(self, fe, sample_df):
        """El número de filas no cambia."""
        result = fe.transform(sample_df)
        assert len(result) == len(sample_df)

    def test_temporal_features_range(self, fe, sample_df):
        """Features cíclicas deben estar en [-1, 1]."""
        result = fe.transform(sample_df)
        for feat in ['month_sin', 'month_cos', 'se_sin', 'se_cos']:
            assert result[feat].min() >= -1.0
            assert result[feat].max() <= 1.0

    def test_is_peak_season_binary(self, fe, sample_df):
        """is_peak_season debe ser 0 o 1."""
        result = fe.transform(sample_df)
        assert set(result['is_peak_season'].unique()).issubset({0, 1})

    def test_quarter_range(self, fe, sample_df):
        """quarter debe ser 1-4."""
        result = fe.transform(sample_df)
        assert result['quarter'].min() >= 1
        assert result['quarter'].max() <= 4

    def test_pop_log_positive(self, fe, sample_df):
        """pop_log debe ser positivo."""
        result = fe.transform(sample_df)
        assert (result['pop_log'] > 0).all()

    def test_region_dummies_exist(self, fe, sample_df):
        """Las dummies de región deben existir."""
        result = fe.transform(sample_df)
        # RJ → Sudeste, así que region_Sudeste = 1
        assert (result['region_Sudeste'] == 1).all()
        assert (result['region_Norte'] == 0).all()

    def test_climate_lags_have_nulls(self, fe, sample_df):
        """Los lags climáticos generan NaN en las primeras observaciones."""
        result = fe.transform(sample_df)
        # lag4w necesita al menos 4 semanas previas
        assert result['tempmed_lag4w'].isnull().any()

    def test_climate_rolling_uses_shift(self, fe, sample_df):
        """Rolling debe usar shift(1) — la primera obs de rolling también es NaN."""
        result = fe.transform(sample_df)
        assert result['tempmed_roll4w'].isnull().any()

    def test_multi_municipio_independent_lags(self, fe, multi_municipio_df):
        """Lags se calculan por municipio, no se mezclan."""
        result = fe.transform(multi_municipio_df)
        # Primer registro de cada municipio debe tener lag NaN
        for geo in result['municipio_geocodigo'].unique():
            subset = result[result['municipio_geocodigo'] == geo]
            first_idx = subset.index[0]
            assert pd.isna(result.loc[first_idx, 'tempmed_lag4w'])


class TestProductionDataset:
    """Tests del dataset de producción (sin leakage)."""

    def test_get_production_no_leakage(self, fe, sample_df):
        """El dataset de producción no contiene variables con leakage."""
        full = fe.transform(sample_df)
        prod = fe.get_production_dataset(full)
        leakage_found = [c for c in prod.columns if c in LEAKAGE_VARS + CLIMATE_VARS]
        assert len(leakage_found) == 0, f"Leakage detectado: {leakage_found}"

    def test_production_has_target(self, fe, sample_df):
        """El dataset de producción incluye el target."""
        full = fe.transform(sample_df)
        prod = fe.get_production_dataset(full)
        assert TARGET in prod.columns

    def test_production_has_engineered_features(self, fe, sample_df):
        """El dataset de producción incluye las features engineered."""
        full = fe.transform(sample_df)
        prod = fe.get_production_dataset(full)
        for feat in ALL_ENGINEERED_FEATURES:
            assert feat in prod.columns, f"Feature {feat} faltante en producción"

    def test_production_column_count(self, fe, sample_df):
        """El dataset de producción tiene el nº esperado de columnas."""
        full = fe.transform(sample_df)
        prod = fe.get_production_dataset(full)
        assert prod.shape[1] == len(PRODUCTION_COLUMNS)

    def test_validate_no_leakage_clean(self, fe, sample_df):
        """validate_no_leakage pasa en dataset limpio."""
        full = fe.transform(sample_df)
        prod = fe.get_production_dataset(full)
        assert fe.validate_no_leakage(prod) is True

    def test_validate_no_leakage_dirty(self, fe, sample_df):
        """validate_no_leakage detecta leakage."""
        full = fe.transform(sample_df)
        assert fe.validate_no_leakage(full) is False


class TestConstants:
    """Tests de consistencia de constantes."""

    def test_region_map_covers_all_states(self):
        """REGION_MAP debe cubrir los 27 estados de Brasil."""
        assert len(REGION_MAP) == 27

    def test_regions_list_correct(self):
        """5 macro-regiones de Brasil."""
        assert len(REGIONS) == 5
        assert set(REGIONS) == {'Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul'}

    def test_all_engineered_features_consistent(self):
        """ALL_ENGINEERED_FEATURES = temporal + climate + geo."""
        expected = TEMPORAL_FEATURES + CLIMATE_FEATURES + GEO_FEATURES
        assert ALL_ENGINEERED_FEATURES == expected

    def test_production_columns_includes_target(self):
        """PRODUCTION_COLUMNS incluye el target."""
        assert TARGET in PRODUCTION_COLUMNS

    def test_no_overlap_leakage_production(self):
        """No debe haber overlap entre leakage y columnas de producción."""
        overlap = set(LEAKAGE_VARS) & set(PRODUCTION_COLUMNS)
        assert len(overlap) == 0, f"Overlap: {overlap}"

    def test_no_climate_in_production(self):
        """Variables climáticas originales no están en producción."""
        overlap = set(CLIMATE_VARS) & set(PRODUCTION_COLUMNS)
        assert len(overlap) == 0, f"Clima en producción: {overlap}"


class TestInputValidation:
    """Tests de validación de entrada."""

    def test_missing_column_raises(self, fe):
        """Columnas faltantes dan ValueError."""
        df_bad = pd.DataFrame({'month': [1], 'SE': [1]})
        with pytest.raises(ValueError, match="Columnas requeridas faltantes"):
            fe.transform(df_bad)

    def test_get_feature_names(self, fe):
        """get_feature_names devuelve la lista correcta."""
        names = fe.get_feature_names()
        assert len(names) == len(ALL_ENGINEERED_FEATURES)

    def test_get_feature_summary(self, fe):
        """get_feature_summary devuelve dict con claves esperadas."""
        summary = fe.get_feature_summary()
        assert 'temporal' in summary
        assert 'climate' in summary
        assert 'geographic' in summary
        assert summary['total'] == len(ALL_ENGINEERED_FEATURES)
