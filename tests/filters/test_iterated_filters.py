"""Tests for the iterated extended and unscented Kalman filters."""

from __future__ import annotations

import numpy as np
import pytest

from kalman_filter import (
    ExtendedKalmanFilter,
    FloatArray,
    FunctionMeasurementModel,
    IteratedExtendedKalmanFilter,
    IteratedUnscentedKalmanFilter,
    KalmanFilter,
    LinearKalmanFilter,
    LinearMotionModel,
    UnscentedKalmanFilter,
)

type IteratedType = type[IteratedExtendedKalmanFilter] | type[IteratedUnscentedKalmanFilter]


def _cubic_sensor() -> FunctionMeasurementModel:
    return FunctionMeasurementModel(
        measure=lambda states: states[..., :1] ** 3, noise_covariance=[0.1], dimension=1
    )


def _range_sensor() -> FunctionMeasurementModel:
    return FunctionMeasurementModel(
        measure=lambda states: np.linalg.norm(states[..., :2], axis=-1, keepdims=True),
        noise_covariance=[1e-4],
        dimension=1,
    )


def _exact_posterior_mean(prior_x: float, prior_P: float, measurement: float) -> float:
    """Posterior mean of the scalar cubic model by numerical integration."""
    grid = np.linspace(-6.0, 8.0, 200_001)
    log_density: FloatArray = (
        -0.5 * (grid - prior_x) ** 2 / prior_P - 0.5 * (measurement - grid**3) ** 2 / 0.1
    )
    density: FloatArray = np.exp(log_density - np.max(log_density))
    mean: float = float(np.sum(grid * density) / np.sum(density))
    return mean


@pytest.mark.parametrize(
    "iterated_type", [IteratedExtendedKalmanFilter, IteratedUnscentedKalmanFilter]
)
def test_iterated_filters_match_plain_filters_on_linear_models(
    linear_filter: LinearKalmanFilter, iterated_type: IteratedType
) -> None:
    iterated = iterated_type(
        linear_filter.motion_model,
        linear_filter.measurement_model,
        linear_filter.x,
        linear_filter.P,
    )
    linear_filter.update([0.3, -0.2])
    iterated.update([0.3, -0.2])
    assert np.allclose(iterated.x, linear_filter.x)
    assert np.allclose(iterated.P, linear_filter.P)
    assert iterated.log_likelihood == pytest.approx(linear_filter.log_likelihood)


def test_iterated_extended_update_is_the_maximum_a_posteriori_state() -> None:
    motion = LinearMotionModel(np.eye(2), [0.1, 0.1])
    sensor = _range_sensor()
    prior_x = np.array([3.0, 1.0])
    prior_P = np.array([[1.0, 0.3], [0.3, 0.5]])
    measurement = np.array([2.0])

    # Stationarity of the MAP objective: P^-1 (x - x0) = H^T R^-1 (z - h(x)).
    def gradient(state: FloatArray) -> FloatArray:
        jacobian = sensor.jacobian(state)
        prior_term = np.linalg.solve(prior_P, state - prior_x)
        residual = (measurement - sensor.measure(state)) / 1e-4
        result: FloatArray = prior_term - jacobian.T @ residual
        return result

    iterated = IteratedExtendedKalmanFilter(motion, sensor, prior_x, prior_P)
    iterated.update(measurement)
    assert np.linalg.norm(gradient(iterated.x.value)) < 1e-4
    plain = ExtendedKalmanFilter(motion, sensor, prior_x, prior_P)
    plain.update(measurement)
    assert np.linalg.norm(gradient(plain.x.value)) > 1.0


@pytest.mark.parametrize(
    ("plain_type", "iterated_type"),
    [
        (ExtendedKalmanFilter, IteratedExtendedKalmanFilter),
        (UnscentedKalmanFilter, IteratedUnscentedKalmanFilter),
    ],
)
def test_iterated_filters_reduce_error_on_nonlinear_measurement(
    plain_type: type[ExtendedKalmanFilter] | type[UnscentedKalmanFilter],
    iterated_type: IteratedType,
) -> None:
    motion = LinearMotionModel([[1.0]], [0.1])
    sensor = _cubic_sensor()
    exact = _exact_posterior_mean(1.0, 1.0, 8.0)
    plain = plain_type(motion, sensor, [1.0], [1.0])
    iterated = iterated_type(motion, sensor, [1.0], [1.0])
    plain.update([8.0])
    iterated.update([8.0])
    assert abs(iterated.x[0] - exact) < 0.05
    assert abs(iterated.x[0] - exact) < 0.2 * abs(plain.x[0] - exact)


def test_iterated_unscented_keeps_linearization_error_in_the_innovation_covariance() -> None:
    motion = LinearMotionModel([[1.0]], [0.1])
    prior_variance = 1.0
    kf = IteratedUnscentedKalmanFilter(motion, _cubic_sensor(), [1.0], [prior_variance])
    kf.update([8.0])
    assert kf.P[0, 0] > 0.0
    assert kf.log_likelihood is not None
    assert np.isfinite(kf.log_likelihood)
    assert kf.R is not None
    assert kf.H is not None
    assert kf.S is not None
    assert kf.R.value[0, 0] == pytest.approx(0.1)
    linearized = kf.H.value[0, 0] ** 2 * prior_variance + kf.R.value[0, 0]
    assert kf.S.value[0, 0] > linearized


@pytest.mark.parametrize(
    "iterated_type", [IteratedExtendedKalmanFilter, IteratedUnscentedKalmanFilter]
)
def test_single_iteration_is_the_plain_update(iterated_type: IteratedType) -> None:
    motion = LinearMotionModel([[1.0]], [0.1])
    sensor = _cubic_sensor()
    plain: KalmanFilter = (
        ExtendedKalmanFilter(motion, sensor, [1.0], [1.0])
        if iterated_type is IteratedExtendedKalmanFilter
        else UnscentedKalmanFilter(motion, sensor, [1.0], [1.0])
    )
    single = iterated_type(motion, sensor, [1.0], [1.0], max_iterations=1)
    plain.update([8.0])
    single.update([8.0])
    assert np.allclose(single.x, plain.x)


@pytest.mark.parametrize(
    "iterated_type", [IteratedExtendedKalmanFilter, IteratedUnscentedKalmanFilter]
)
def test_iterated_filters_reject_invalid_settings(iterated_type: IteratedType) -> None:
    motion = LinearMotionModel([[1.0]], [0.1])
    with pytest.raises(ValueError, match="max_iterations"):
        iterated_type(motion, _cubic_sensor(), [1.0], [1.0], max_iterations=0)
    with pytest.raises(ValueError, match="tolerance"):
        iterated_type(motion, _cubic_sensor(), [1.0], [1.0], tolerance=0.0)
