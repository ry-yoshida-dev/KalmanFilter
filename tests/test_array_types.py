"""Tests for float array type aliases and container acceptance."""

from __future__ import annotations

import numpy as np

from kalman_filter import FloatArray
from kalman_filter.parameter_containers import (
    ErrorCovarianceMatrix,
    MeasurementNoiseCovariance,
    ProcessNoiseCovariance,
    StateVector,
)


def test_float_array_alias_matches_floating_ndarray() -> None:
    """FloatArray values must use a floating NumPy dtype."""
    matrix: FloatArray = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    assert np.issubdtype(matrix.dtype, np.floating)


def test_state_vector_accepts_float64_column_vector() -> None:
    """StateVector.register_zero_vec_from_format stores float64 values."""
    state_format = {
        "static": ["x"],
        "dynamic": [],
        "acceleration": [],
    }
    state = StateVector.register_zero_vec_from_format(format=state_format)
    assert state.value.dtype == np.float64
    assert state.value.shape == (1, 1)


def test_covariance_matrices_accept_float_weights() -> None:
    """Diagonal covariance builders accept list[float] weights."""
    r_matrix = MeasurementNoiseCovariance.create_matrix(weights=[1.0, 4.0], scaling_factor=1.0)
    p_matrix = ErrorCovarianceMatrix.create_matrix(weights=[10.0, 20.0])
    q_matrix = ProcessNoiseCovariance.create_matrix(weights=[0.1, 0.2])

    assert np.allclose(np.diag(r_matrix.value), [1.0, 4.0])
    assert np.allclose(np.diag(p_matrix.value), [10.0, 20.0])
    assert np.allclose(np.diag(q_matrix.value), [0.1, 0.2])


def test_covariance_matrices_accept_float_ndarray_weights() -> None:
    """Diagonal covariance builders accept FloatArray weight vectors."""
    weights: FloatArray = np.array([2.0, 3.0], dtype=np.float64)
    matrix = MeasurementNoiseCovariance.create_matrix(weights=weights)
    assert np.allclose(np.diag(matrix.value), [2.0, 3.0])
