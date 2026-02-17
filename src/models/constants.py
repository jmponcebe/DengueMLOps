"""
Constantes de modelado para el clasificador de dengue.
Mapeos de labels, clases, y configuración de modelos.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None  # type: ignore[misc,assignment]

# Nivel de alerta 1-4 → 0-3 (XGBoost/LightGBM requieren 0-indexed)
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
CLASSES = [1, 2, 3, 4]

# Modelos que necesitan labels 0-indexed
NEEDS_ZERO_INDEX = tuple(
    cls for cls in (XGBClassifier, LGBMClassifier) if cls is not None
)

SEED = 42


def encode_labels(y):
    """nivel 1-4 -> 0-3 para modelos que requieren 0-indexed."""
    return y.map(LABEL_MAP)


def decode_labels(y_pred):
    """0-3 -> nivel 1-4 original."""
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        return y_pred.map(LABEL_MAP_INV)  # type: ignore[arg-type]
    return np.array([LABEL_MAP_INV[v] for v in y_pred])


def compute_balanced_weights(y):
    """
    Calcula sample weights balanceados inversamente proporcionales
    a la frecuencia de cada clase. Estándar para desbalance severo.

    Returns:
        class_weights: dict {class: weight}
        sample_weights: np.array con peso por observación
    """
    counts = y.value_counts().sort_index()
    n_classes = len(counts)
    class_weights = {c: len(y) / (n_classes * count) for c, count in counts.items()}
    sample_weights = y.map(class_weights).values
    return class_weights, sample_weights
