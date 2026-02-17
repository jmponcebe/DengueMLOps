"""
Tests para src/models/constants.py
Mapeo de labels, constantes y cálculo de pesos balanceados.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.constants import (
    LABEL_MAP, LABEL_MAP_INV, CLASSES, NEEDS_ZERO_INDEX, SEED,
    encode_labels, decode_labels, compute_balanced_weights,
)


class TestConstants:
    """Verificar que las constantes de modelado son consistentes."""

    def test_label_map_covers_all_classes(self):
        assert set(LABEL_MAP.keys()) == {1, 2, 3, 4}

    def test_label_map_inv_is_inverse(self):
        for k, v in LABEL_MAP.items():
            assert LABEL_MAP_INV[v] == k

    def test_classes_list(self):
        assert CLASSES == [1, 2, 3, 4]

    def test_needs_zero_index_is_tuple(self):
        assert isinstance(NEEDS_ZERO_INDEX, tuple)
        assert len(NEEDS_ZERO_INDEX) >= 1  # al menos XGBClassifier

    def test_seed_value(self):
        assert SEED == 42


class TestEncodeLabels:
    """encode_labels: nivel 1-4 → 0-3."""

    def test_encode_series(self):
        y = pd.Series([1, 2, 3, 4])
        encoded = encode_labels(y)
        assert list(encoded) == [0, 1, 2, 3]

    def test_encode_repeated(self):
        y = pd.Series([1, 1, 1, 2, 4])
        encoded = encode_labels(y)
        assert list(encoded) == [0, 0, 0, 1, 3]

    def test_encode_preserves_index(self):
        y = pd.Series([3, 4], index=[10, 20])
        encoded = encode_labels(y)
        assert list(encoded.index) == [10, 20]


class TestDecodeLabels:
    """decode_labels: 0-3 → nivel 1-4."""

    def test_decode_series(self):
        y = pd.Series([0, 1, 2, 3])
        decoded = decode_labels(y)
        assert list(decoded) == [1, 2, 3, 4]

    def test_decode_numpy(self):
        y = np.array([0, 1, 2, 3])
        decoded = decode_labels(y)
        assert isinstance(decoded, np.ndarray)
        np.testing.assert_array_equal(decoded, [1, 2, 3, 4])

    def test_decode_list(self):
        decoded = decode_labels([0, 2, 3])
        np.testing.assert_array_equal(decoded, [1, 3, 4])

    def test_roundtrip_encode_decode(self):
        original = pd.Series([1, 2, 3, 4, 1, 4])
        roundtrip = decode_labels(encode_labels(original))
        assert list(roundtrip) == list(original)


class TestComputeBalancedWeights:
    """compute_balanced_weights: pesos inversamente proporcionales a frecuencia."""

    def test_returns_dict_and_array(self):
        y = pd.Series([1, 1, 1, 2, 3, 4])
        cw, sw = compute_balanced_weights(y)
        assert isinstance(cw, dict)
        assert isinstance(sw, np.ndarray)
        assert len(sw) == len(y)

    def test_balanced_dataset_equal_weights(self):
        y = pd.Series([1, 2, 3, 4])  # perfectamente balanceado
        cw, sw = compute_balanced_weights(y)
        # Todos los pesos deben ser iguales
        assert all(abs(w - 1.0) < 1e-10 for w in cw.values())

    def test_imbalanced_rare_class_higher_weight(self):
        # Clase 1 dominante como en datos reales
        y = pd.Series([1]*93 + [2]*4 + [3]*1 + [4]*2)
        cw, sw = compute_balanced_weights(y)
        # Clase rara (3) debe tener peso mucho mayor que dominante (1)
        assert cw[3] > cw[1]
        assert cw[3] > cw[2]

    def test_sample_weights_match_class_weights(self):
        y = pd.Series([1, 1, 2, 3])
        cw, sw = compute_balanced_weights(y)
        # El peso de cada sample debe coincidir con su clase
        assert sw[0] == cw[1]
        assert sw[2] == cw[2]
        assert sw[3] == cw[3]

    def test_weights_sum_to_n_samples(self):
        y = pd.Series([1]*10 + [2]*5 + [3]*3 + [4]*2)
        _, sw = compute_balanced_weights(y)
        # La suma de pesos = n_samples (propiedad de balanced weights)
        np.testing.assert_almost_equal(float(sw.sum()), float(len(y)), decimal=10)  # type: ignore[attr-defined]
