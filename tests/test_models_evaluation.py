"""
Tests para src/models/evaluation.py
Métricas de clasificación, visualizaciones y reporting.
"""

import json
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # backend sin GUI para tests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.models.evaluation import (
    compute_metrics, plot_confusion_matrix,
    plot_feature_importance, save_classification_report,
)
from src.models.constants import CLASSES


# ---------- Fixtures ----------

@pytest.fixture
def classification_data():
    """Datos de clasificación sintéticos (labels originales 1-4)."""
    np.random.seed(42)
    y_true = np.array([1]*20 + [2]*5 + [3]*3 + [4]*2)
    y_pred = np.array([1]*18 + [2]*2 + [2]*3 + [1]*2 + [3]*2 + [1] + [4]*1 + [1]*1)
    return y_true, y_pred


@pytest.fixture
def classification_data_with_proba(classification_data):
    """Añade probabilidades ficticias."""
    y_true, y_pred = classification_data
    np.random.seed(42)
    n = len(y_true)
    # Probabilidades aleatorias normalizadas
    proba = np.random.dirichlet([1, 1, 1, 1], size=n)
    return y_true, y_pred, proba


class TestComputeMetrics:
    """Verificar métricas de clasificación multiclase."""

    def test_returns_dict(self, classification_data):
        y_true, y_pred = classification_data
        metrics = compute_metrics(y_true, y_pred)
        assert isinstance(metrics, dict)

    def test_required_keys_present(self, classification_data):
        y_true, y_pred = classification_data
        metrics = compute_metrics(y_true, y_pred)
        for key in ['accuracy', 'macro_f1', 'weighted_f1', 'cohen_kappa']:
            assert key in metrics, f"Falta clave: {key}"

    def test_per_class_f1(self, classification_data):
        y_true, y_pred = classification_data
        metrics = compute_metrics(y_true, y_pred)
        for cls in CLASSES:
            assert f'f1_class_{cls}' in metrics

    def test_metrics_range(self, classification_data):
        y_true, y_pred = classification_data
        metrics = compute_metrics(y_true, y_pred)
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['macro_f1'] <= 1
        assert 0 <= metrics['weighted_f1'] <= 1

    def test_perfect_predictions(self):
        y = np.array([1, 2, 3, 4, 1, 2])
        metrics = compute_metrics(y, y)
        assert metrics['accuracy'] == 1.0
        assert metrics['macro_f1'] == 1.0

    def test_with_probabilities(self, classification_data_with_proba):
        y_true, y_pred, y_proba = classification_data_with_proba
        metrics = compute_metrics(y_true, y_pred, y_proba)
        assert 'log_loss' in metrics
        assert metrics['log_loss'] > 0

    def test_without_probabilities(self, classification_data):
        y_true, y_pred = classification_data
        metrics = compute_metrics(y_true, y_pred)
        assert 'log_loss' not in metrics
        assert 'roc_auc_ovr' not in metrics


class TestPlotConfusionMatrix:
    """Verificar generación de confusion matrix."""

    def test_returns_figure(self, classification_data):
        y_true, y_pred = classification_data
        fig = plot_confusion_matrix(y_true, y_pred)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_two_subplots(self, classification_data):
        y_true, y_pred = classification_data
        fig = plot_confusion_matrix(y_true, y_pred, title='Test')
        # 2 plots principales + colorbars
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_custom_title(self, classification_data):
        y_true, y_pred = classification_data
        fig = plot_confusion_matrix(y_true, y_pred, title='Custom Title')
        titles = [ax.get_title() for ax in fig.axes]
        assert any('Custom Title' in t for t in titles)
        plt.close(fig)


class TestPlotFeatureImportance:
    """Verificar feature importance plot."""

    def test_returns_figure_for_tree_model(self):
        class MockModel:
            feature_importances_ = np.array([0.3, 0.5, 0.1, 0.1])

        fig = plot_feature_importance(MockModel(), ['a', 'b', 'c', 'd'])
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_returns_none_without_importances(self):
        class MockModel:
            pass

        result = plot_feature_importance(MockModel(), ['a', 'b'])
        assert result is None

    def test_top_n_limits_bars(self):
        class MockModel:
            feature_importances_ = np.array([0.1] * 20)

        features = [f'feat_{i}' for i in range(20)]
        fig = plot_feature_importance(MockModel(), features, top_n=5)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestSaveClassificationReport:
    """Verificar guardado de classification report."""

    def test_saves_json(self, classification_data, tmp_path):
        y_true, y_pred = classification_data
        filepath = tmp_path / 'report.json'
        report = save_classification_report(y_true, y_pred, str(filepath))
        assert filepath.exists()
        assert isinstance(report, dict)

    def test_json_valid(self, classification_data, tmp_path):
        y_true, y_pred = classification_data
        filepath = tmp_path / 'report.json'
        save_classification_report(y_true, y_pred, str(filepath))
        with open(filepath) as f:
            data = json.load(f)
        assert 'accuracy' in data

    def test_report_contains_classes(self, classification_data, tmp_path):
        y_true, y_pred = classification_data
        filepath = tmp_path / 'report.json'
        report = save_classification_report(y_true, y_pred, str(filepath))
        # sklearn report convierte labels a string
        assert '1' in report
