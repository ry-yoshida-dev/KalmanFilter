"""Tests for measurement and state container behavior."""

from __future__ import annotations

import numpy as np

from kalman_filter.parameter_containers import (
    MeasurementMatrix,
    MeasurementVector,
    StateRole,
    StateVector,
)


def test_measurement_vector_indexing_by_name() -> None:
    """MeasurementVector supports string-key access."""
    vector = MeasurementVector.register_zero_vec_from_format(format=["u", "v"])
    vector["u"] = 3.5
    vector["v"] = 7.0

    assert vector["u"] == 3.5
    assert vector["v"] == 7.0


def test_state_vector_group_indexing() -> None:
    """StateVector supports StateRole-based indexing."""
    state_format = {
        "static": ["u", "v"],
        "dynamic": ["u"],
        "acceleration": [],
    }
    state = StateVector.register_zero_vec_from_format(format=state_format)
    state[StateRole.STATIC, "u"] = 10.0
    state[StateRole.STATIC, "v"] = 20.0
    state[StateRole.DYNAMIC, "u"] = 1.5

    assert state[StateRole.STATIC, "u"] == 10.0
    assert state[StateRole.STATIC, "v"] == 20.0
    assert state[StateRole.DYNAMIC, "u"] == 1.5


def test_measurement_matrix_default_is_rectangular_identity() -> None:
    """Default H maps the first measurement dimensions from state space."""
    state_format = {
        "static": ["u", "v", "s", "r"],
        "dynamic": ["u", "v", "s"],
        "acceleration": [],
    }
    state = StateVector.register_zero_vec_from_format(format=state_format)
    measurement = MeasurementVector.register_zero_vec_from_format(format=["u", "v", "s", "r"])

    matrix = MeasurementMatrix.create_matrix(
        state_vector=state,
        measurement_vector=measurement,
        matrix_value=None,
    )

    expected = np.eye(measurement.dim, state.dim)
    assert np.allclose(matrix.value, expected)


def test_compute_innovation_is_measurement_minus_prediction() -> None:
    """Innovation equals z minus H @ x."""
    state_format = {"static": ["x"], "dynamic": [], "acceleration": []}
    state = StateVector.register_zero_vec_from_format(format=state_format)
    state.value[0, 0] = 2.0

    measurement = MeasurementVector.register_zero_vec_from_format(format=["x"])
    measurement.value[0, 0] = 5.0

    matrix = MeasurementMatrix.create_matrix(
        state_vector=state,
        measurement_vector=measurement,
    )
    innovation = measurement.compute_innovation(x=state, H=matrix)

    assert innovation.shape == (1, 1)
    assert np.allclose(innovation, np.array([[3.0]]))
