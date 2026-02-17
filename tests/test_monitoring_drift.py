"""
Tests para src/monitoring/drift_detector.py
Detección de data drift con Evidently.
"""

import os
import csv
import pytest
import numpy as np
import pandas as pd

from src.monitoring.drift_detector import (
    DriftDetector, ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    PREDICTION_COL, build_reference_dataset,
)


# ---------- Fixtures ----------

@pytest.fixture
def reference_df():
    """DataFrame de referencia con las 15 features del modelo."""
    np.random.seed(42)
    n = 200
    data = {
        'month_sin': np.sin(2 * np.pi * np.random.randint(1, 13, n) / 12),
        'month_cos': np.cos(2 * np.pi * np.random.randint(1, 13, n) / 12),
        'se_sin': np.sin(2 * np.pi * np.random.randint(1, 53, n) / 52),
        'se_cos': np.cos(2 * np.pi * np.random.randint(1, 53, n) / 52),
        'tempmed_lag8w': np.random.uniform(20, 32, n),
        'tempmed_roll12w': np.random.uniform(22, 30, n),
        'umidmed_roll4w': np.random.uniform(60, 85, n),
        'temp_x_humid_lag4w': np.random.uniform(1200, 2800, n),
        'pop_log': np.random.uniform(8, 16, n),
        'is_peak_season': np.random.choice([0, 1], n),
        'quarter': np.random.choice([1, 2, 3, 4], n),
        'region_Nordeste': np.random.choice([0, 1], n),
        'region_Centro-Oeste': np.random.choice([0, 1], n),
        'region_Sudeste': np.random.choice([0, 1], n),
        'region_Sul': np.random.choice([0, 1], n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def current_df(reference_df):
    """DataFrame de 'producción' (misma estructura, datos diferentes)."""
    np.random.seed(99)
    n = 50
    data = {}
    for col in reference_df.columns:
        if reference_df[col].dtype == float:
            data[col] = np.random.uniform(
                reference_df[col].min(), reference_df[col].max(), n
            )
        else:
            data[col] = np.random.choice(reference_df[col].unique(), n)
    return pd.DataFrame(data)


@pytest.fixture
def detector(reference_df, tmp_path):
    """DriftDetector inicializado con datos de referencia."""
    return DriftDetector(reference_df, reports_dir=str(tmp_path))


# ---------- Tests ----------

class TestDriftDetectorInit:
    """Verificar inicialización del DriftDetector."""

    def test_creates_reports_dir(self, reference_df, tmp_path):
        reports_dir = tmp_path / "new_reports"
        DriftDetector(reference_df, reports_dir=str(reports_dir))
        assert reports_dir.exists()

    def test_filters_available_features(self, tmp_path):
        # Solo pasa un subconjunto de features
        df = pd.DataFrame({
            'month_sin': [0.5, 0.3],
            'pop_log': [10.0, 12.0],
            'columna_extra': [1, 2],
        })
        det = DriftDetector(df, reports_dir=str(tmp_path))
        assert 'month_sin' in det.features
        assert 'pop_log' in det.features
        assert 'columna_extra' not in det.features

    def test_stores_reference_count(self, reference_df, tmp_path):
        det = DriftDetector(reference_df, reports_dir=str(tmp_path))
        assert det._n_reference == len(reference_df)


class TestDriftDetect:
    """Verificar detección de drift."""

    def test_returns_dict(self, detector, current_df):
        result = detector.detect(current_df)
        assert isinstance(result, dict)

    def test_result_keys(self, detector, current_df):
        result = detector.detect(current_df)
        expected = {'report_path', 'n_reference', 'n_current', 'features_analyzed', 'timestamp'}
        assert expected.issubset(result.keys())

    def test_report_html_created(self, detector, current_df):
        result = detector.detect(current_df, report_name='test_report')
        assert os.path.exists(result['report_path'])
        assert result['report_path'].endswith('.html')

    def test_n_current_matches(self, detector, current_df):
        result = detector.detect(current_df)
        assert result['n_current'] == len(current_df)

    def test_features_analyzed_count(self, detector, current_df):
        result = detector.detect(current_df)
        assert result['features_analyzed'] > 0
        assert result['features_analyzed'] <= len(ALL_FEATURES)


class TestDetectFromLog:
    """Verificar detección desde CSV de predicciones."""

    def test_file_not_found_raises(self, detector):
        with pytest.raises(FileNotFoundError, match="No existe log"):
            detector.detect_from_log(log_path="/nonexistent/path.csv")

    def test_reads_csv_and_detects(self, detector, current_df, tmp_path):
        csv_path = tmp_path / "predictions.csv"
        current_df.to_csv(csv_path, index=False)
        result = detector.detect_from_log(log_path=str(csv_path))
        assert 'report_path' in result

    def test_last_n_limits_rows(self, detector, reference_df, tmp_path):
        csv_path = tmp_path / "big_log.csv"
        reference_df.to_csv(csv_path, index=False)  # 200 filas
        result = detector.detect_from_log(log_path=str(csv_path), last_n=50)
        assert result['n_current'] == 50


class TestBuildReferenceDataset:
    """Verificar build_reference_dataset con parquet existente."""

    def test_loads_from_parquet(self, reference_df, tmp_path):
        parquet_path = tmp_path / "reference.parquet"
        reference_df.to_parquet(parquet_path)
        result = build_reference_dataset(str(parquet_path))
        assert len(result) == len(reference_df)
        assert list(result.columns) == list(reference_df.columns)

    def test_returns_dataframe(self, reference_df, tmp_path):
        parquet_path = tmp_path / "ref.parquet"
        reference_df.to_parquet(parquet_path)
        result = build_reference_dataset(str(parquet_path))
        assert isinstance(result, pd.DataFrame)


class TestFeatureLists:
    """Verificar consistencia de las listas de features."""

    def test_all_features_is_union(self):
        assert set(ALL_FEATURES) == set(NUMERICAL_FEATURES) | set(CATEGORICAL_FEATURES)

    def test_no_overlap(self):
        overlap = set(NUMERICAL_FEATURES) & set(CATEGORICAL_FEATURES)
        assert len(overlap) == 0, f"Overlap: {overlap}"

    def test_prediction_col_name(self):
        assert PREDICTION_COL == "predicted_nivel"
