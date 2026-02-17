"""
Funciones de evaluación para el clasificador de dengue.
Métricas, visualizaciones y reporting.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, log_loss,
    classification_report, roc_auc_score, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

from src.models.constants import CLASSES


def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Calcula métricas de clasificación multiclase.
    Espera labels originales (1-4).

    Returns:
        dict con accuracy, macro_f1, weighted_f1, cohen_kappa,
        y opcionalmente log_loss, roc_auc_ovr, f1 por clase.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
    }

    if y_proba is not None:
        try:
            metrics['log_loss'] = log_loss(y_true, y_proba, labels=CLASSES)
        except Exception:
            pass
        try:
            y_bin = label_binarize(y_true, classes=CLASSES)
            metrics['roc_auc_ovr'] = roc_auc_score(
                y_bin, y_proba, multi_class='ovr', average='macro'
            )
        except Exception:
            pass

    f1_per_class = np.asarray(
        f1_score(y_true, y_pred, average=None, labels=CLASSES, zero_division=0)
    )
    for cls, f1_val in zip(CLASSES, f1_per_class):
        metrics[f'f1_class_{cls}'] = f1_val

    return metrics


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Genera confusion matrix con counts y normalizada."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=CLASSES, ax=axes[0], cmap='Blues'
    )
    axes[0].set_title(f'{title} (counts)')

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=CLASSES, normalize='true',
        ax=axes[1], cmap='Blues', values_format='.2f'
    )
    axes[1].set_title(f'{title} (normalized)')

    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=15):
    """Feature importance horizontal bar chart para tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        return None

    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), importances[idx], color='steelblue')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    plt.tight_layout()
    return fig


def save_classification_report(y_true, y_pred, filepath):
    """Guarda classification report como JSON."""
    report = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True)
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    return report
