"""
Tests para src/models/trainer.py
Predicción con DengueModelTrainer (predict/predict_proba).
train_and_log se testea con mocks de MLflow.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, ANY

from src.models.trainer import DengueModelTrainer
from src.models.constants import CLASSES, NEEDS_ZERO_INDEX


# ---------- Fixtures ----------

@pytest.fixture
def features():
    return ['month_sin', 'month_cos', 'pop_log', 'tempmed_lag8w']


@pytest.fixture
def trainer(features):
    return DengueModelTrainer(features=features)


@pytest.fixture
def X_sample():
    """DataFrame de features para predicción."""
    np.random.seed(42)
    return pd.DataFrame({
        'month_sin': np.random.randn(10),
        'month_cos': np.random.randn(10),
        'pop_log': np.random.uniform(8, 16, 10),
        'tempmed_lag8w': np.random.uniform(20, 32, 10),
    })


# ---------- Tests ----------

class TestTrainerInit:
    """Verificar inicialización."""

    def test_stores_features(self, trainer, features):
        assert trainer.features == features

    def test_default_seed(self, trainer):
        assert trainer.seed == 42

    def test_custom_seed(self, features):
        t = DengueModelTrainer(features=features, seed=123)
        assert t.seed == 123


class TestPredict:
    """predict: decode automático para modelos 0-indexed."""

    def test_non_xgb_model_returns_raw(self, X_sample):
        """Modelo genérico devuelve predicciones sin decodificar."""
        # Mock sin spec de XGBoost
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2])

        result = DengueModelTrainer.predict(mock_model, X_sample)
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 1, 2, 3, 4, 1, 2])

    def test_xgb_model_decodes_labels(self, X_sample):
        """XGBClassifier devuelve labels 0-3, predict los decodifica a 1-4."""
        from xgboost import XGBClassifier
        mock_model = MagicMock(spec=XGBClassifier)
        mock_model.predict.return_value = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])

        result = DengueModelTrainer.predict(mock_model, X_sample)
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 1, 2, 3, 4, 1, 2])


class TestPredictProba:
    """predict_proba: delegación directa al modelo."""

    def test_returns_model_proba(self, X_sample):
        mock_model = MagicMock()
        expected = np.random.dirichlet([1, 1, 1, 1], size=10)
        mock_model.predict_proba.return_value = expected

        result = DengueModelTrainer.predict_proba(mock_model, X_sample)
        np.testing.assert_array_equal(result, expected)


class TestTrainAndLog:
    """train_and_log con mocks de MLflow — verifica flujo sin servidor."""

    @pytest.fixture
    def train_data(self, X_sample):
        """Datos de train/val sintéticos."""
        np.random.seed(42)
        X_val = X_sample.copy()
        y_train = pd.Series(np.random.choice(CLASSES, len(X_sample)))
        y_val = pd.Series(np.random.choice(CLASSES, len(X_val)))
        return X_sample, y_train, X_val, y_val

    @patch('src.models.trainer.mlflow')
    @patch('src.models.trainer.save_classification_report')
    @patch('src.models.trainer.plot_confusion_matrix')
    @patch('src.models.trainer.plot_feature_importance')
    @patch('os.makedirs')
    def test_train_and_log_runs(
        self, mock_makedirs, mock_fi, mock_cm, mock_report,
        mock_mlflow, trainer, train_data,
    ):
        X_train, y_train, X_val, y_val = train_data

        # Mock del modelo
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.choice(CLASSES, len(X_val))
        mock_model.predict_proba.return_value = np.random.dirichlet([1]*4, len(X_val))
        mock_model.get_params.return_value = {'n_estimators': 100}

        # Mock MLflow context
        mock_run = MagicMock()
        mock_run.info.run_id = 'test-run-id'
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Mock plots
        mock_fig = MagicMock()
        mock_cm.return_value = mock_fig
        mock_fi.return_value = mock_fig

        run_id, metrics = trainer.train_and_log(
            mock_model, 'test-run', X_train, y_train, X_val, y_val,
            log_model=False,
        )

        assert run_id == 'test-run-id'
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'macro_f1' in metrics
        mock_model.fit.assert_called_once()

    @patch('src.models.trainer.mlflow')
    @patch('src.models.trainer.save_classification_report')
    @patch('src.models.trainer.plot_confusion_matrix')
    @patch('src.models.trainer.plot_feature_importance')
    @patch('os.makedirs')
    def test_train_with_sample_weights(
        self, mock_makedirs, mock_fi, mock_cm, mock_report,
        mock_mlflow, trainer, train_data,
    ):
        X_train, y_train, X_val, y_val = train_data
        sw = np.ones(len(y_train))

        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.choice(CLASSES, len(X_val))
        mock_model.predict_proba.return_value = np.random.dirichlet([1]*4, len(X_val))
        mock_model.get_params.return_value = {}

        mock_run = MagicMock()
        mock_run.info.run_id = 'test-sw'
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        mock_fig = MagicMock()
        mock_cm.return_value = mock_fig
        mock_fi.return_value = mock_fig

        trainer.train_and_log(
            mock_model, 'sw-run', X_train, y_train, X_val, y_val,
            sample_weight=sw, log_model=False,
        )

        # Verificar que se pasaron sample_weight al fit
        call_kwargs = mock_model.fit.call_args
        assert 'sample_weight' in call_kwargs.kwargs
