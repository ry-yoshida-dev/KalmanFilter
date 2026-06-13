"""Tests for KalmanFilter predict/update workflow."""

from __future__ import annotations

import numpy as np

from kalman_filter import KalmanFilter, KFParameters
from kalman_filter.parameter_containers import ErrorCovarianceMatrix, StateVector

from helpers import make_measurement_vector


def test_predict_returns_state_and_covariance(default_params: KFParameters) -> None:
    """predict() returns StateVector and ErrorCovarianceMatrix instances."""
    kalman_filter = KalmanFilter(params=default_params)
    initial_measurement = make_measurement_vector([100.0, 80.0, 400.0, 1.2])
    kalman_filter.set_initial_measurement(initial_measurement)

    predicted_state, predicted_covariance = kalman_filter.predict(is_update_x=False, is_update_P=False)

    assert isinstance(predicted_state, StateVector)
    assert isinstance(predicted_covariance, ErrorCovarianceMatrix)
    assert predicted_state.value.shape == (default_params.x.dim, 1)
    assert predicted_covariance.value.shape == (default_params.x.dim, default_params.x.dim)


def test_predict_and_update_mutate_internal_state(default_params: KFParameters) -> None:
    """A full predict/update cycle updates x and P inside the filter."""
    kalman_filter = KalmanFilter(params=default_params)
    initial_measurement = make_measurement_vector([100.0, 80.0, 400.0, 1.2])
    kalman_filter.set_initial_measurement(initial_measurement)

    state_before = kalman_filter.x.value.copy()
    covariance_before = kalman_filter.P.value.copy()

    kalman_filter.predict()
    new_measurement = make_measurement_vector([102.0, 81.0, 398.0, 1.2])
    kalman_filter.update(z=new_measurement)

    assert not np.allclose(kalman_filter.x.value, state_before)
    assert not np.allclose(kalman_filter.P.value, covariance_before)


def test_clone_produces_independent_filter(default_params: KFParameters) -> None:
    """clone() must not share mutable parameter arrays with the source filter."""
    source = KalmanFilter(params=default_params)
    source.set_initial_measurement(make_measurement_vector([1.0, 2.0, 3.0, 4.0]))
    source.predict()
    source.update(z=make_measurement_vector([1.1, 2.1, 3.1, 4.1]))

    cloned = source.clone()
    cloned.x.value[0, 0] = -123.0

    assert source.x.value[0, 0] != -123.0
