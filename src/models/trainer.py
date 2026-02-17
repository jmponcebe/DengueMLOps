"""
Pipeline de entrenamiento del clasificador de dengue con MLflow tracking.
Encapsula train/predict/log para producción y reentrenamiento.
"""

import os
import time
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src.models.constants import (
    NEEDS_ZERO_INDEX, SEED, encode_labels, decode_labels,
)
from src.models.evaluation import (
    compute_metrics, plot_confusion_matrix,
    plot_feature_importance, save_classification_report,
)
from configs.mlflow_config import DEFAULT_TAGS, ARTIFACT_TEMP_DIR


def _tmp(name):
    return os.path.join(ARTIFACT_TEMP_DIR, name)


class DengueModelTrainer:
    """
    Entrena, evalúa y loguea modelos de clasificación de dengue en MLflow.

    Maneja automáticamente:
    - Encoding 1-4 → 0-3 para XGBoost/LightGBM
    - Sample weights balanceados
    - Logging completo en MLflow (params, metrics, artifacts, model)
    """

    def __init__(self, features, seed=SEED):
        self.features = features
        self.seed = seed

    def train_and_log(self, model, run_name, X_train, y_train, X_val, y_val,
                      sample_weight=None, extra_params=None, extra_tags=None,
                      log_model=True):
        """
        Entrena modelo, evalúa en validación, loguea todo en MLflow.

        Returns:
            (run_id, metrics_dict)
        """
        zero_idx = isinstance(model, NEEDS_ZERO_INDEX)

        with mlflow.start_run(run_name=run_name) as run:
            tags = {**DEFAULT_TAGS, 'algorithm': type(model).__name__}
            if extra_tags:
                tags.update(extra_tags)
            mlflow.set_tags(tags)

            mlflow.log_params(model.get_params())
            if extra_params:
                mlflow.log_params(extra_params)
            mlflow.log_param('n_features', X_train.shape[1])
            mlflow.log_param('n_train_samples', X_train.shape[0])
            mlflow.log_param('seed', self.seed)

            y_fit = encode_labels(y_train) if zero_idx else y_train

            start = time.time()
            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight
            model.fit(X_train, y_fit, **fit_kwargs)
            train_time = time.time() - start
            mlflow.log_metric('training_time_s', round(train_time, 2))

            y_pred_raw = model.predict(X_val)
            y_pred = decode_labels(y_pred_raw) if zero_idx else y_pred_raw

            y_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_val)
                except Exception:
                    pass

            metrics = compute_metrics(y_val, y_pred, y_proba)
            mlflow.log_metrics(metrics)

            # Artifacts
            os.makedirs(ARTIFACT_TEMP_DIR, exist_ok=True)
            fig_cm = plot_confusion_matrix(y_val, y_pred, title=run_name)
            fig_cm.savefig(_tmp('confusion_matrix.png'), dpi=100, bbox_inches='tight')
            mlflow.log_artifact(_tmp('confusion_matrix.png'))
            plt.close(fig_cm)

            fig_fi = plot_feature_importance(model, self.features)
            if fig_fi:
                fig_fi.savefig(_tmp('feature_importance.png'), dpi=100, bbox_inches='tight')
                mlflow.log_artifact(_tmp('feature_importance.png'))
                plt.close(fig_fi)

            save_classification_report(y_val, y_pred, _tmp('classification_report.json'))
            mlflow.log_artifact(_tmp('classification_report.json'))

            if log_model:
                y_tr_pred = model.predict(X_train)
                signature = infer_signature(X_train, y_tr_pred)
                input_example = X_val.iloc[:3]
                mlflow.sklearn.log_model(  # type: ignore[attr-defined]
                    model, name='model',
                    signature=signature,
                    input_example=input_example,
                )

            run_id = run.info.run_id
            print(f'{run_name}: macro_f1={metrics["macro_f1"]:.4f}, '
                  f'accuracy={metrics["accuracy"]:.4f}, '
                  f'kappa={metrics["cohen_kappa"]:.4f}, '
                  f'time={train_time:.1f}s')

        return run_id, metrics

    @staticmethod
    def predict(model, X):
        """Predicción con decode automático para 0-indexed models."""
        y_pred_raw = model.predict(X)
        if isinstance(model, NEEDS_ZERO_INDEX):
            return decode_labels(y_pred_raw)
        return y_pred_raw

    @staticmethod
    def predict_proba(model, X):
        """Probabilidades de predicción."""
        return model.predict_proba(X)
