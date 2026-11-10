"""Tests for the measurement models."""

from __future__ import annotations

import numpy as np
import pytest

from kalman_filter import (
    FunctionMeasurementModel,
    KinematicMotionModel,
    LinearMeasurementModel,
    MeasurementModel,
    RangeBearingMeasurementModel,
)


def test_linear_measurement_from_observed_states() -> None:
    model = KinematicMotionModel.constant_velocity(["x", "y"], noise_intensity=1.0)
    sensor = LinearMeasurementModel.from_observed_states(model.layout, ["y", "x"], [1.0, 2.0])
    assert np.allclose(sensor.measure(np.array([1.0, 2.0, 3.0, 4.0])), [2.0, 1.0])
    assert sensor.dimension == 2


def test_range_bearing_measurement_and_jacobian() -> None:
    sensor = RangeBearingMeasurementModel((0, 1), [0.1, 0.01], sensor_position=(1.0, 1.0))
    state = np.array([4.0, 5.0, 0.0])
    assert np.allclose(sensor.measure(state), [5.0, np.arctan2(4.0, 3.0)])
    numerical = FunctionMeasurementModel(sensor.measure, [0.1, 0.01], 2).jacobian(state)
    assert np.allclose(sensor.jacobian(state), numerical, atol=1e-7)


def test_range_bearing_residual_and_mean_wrap() -> None:
    sensor = RangeBearingMeasurementModel((0, 1), [0.1, 0.01])
    residual = sensor.residual(np.array([1.0, np.pi - 0.1]), np.array([1.0, -np.pi + 0.1]))
    assert residual[1] == pytest.approx(-0.2)
    mean = sensor.mean(np.array([[1.0, np.pi - 0.1], [1.0, -np.pi + 0.1]]), np.array([0.5, 0.5]))
    assert abs(mean[1]) == pytest.approx(np.pi)


def test_range_bearing_jacobian_matches_numerical() -> None:
    model = RangeBearingMeasurementModel([0, 1], [0.1, 0.01], sensor_position=(1.0, -2.0))
    for mean in np.random.default_rng(0).normal(size=(4, 3)) * 5.0:
        numerical = MeasurementModel.jacobian(model, mean)
        assert np.allclose(model.jacobian(mean), numerical, rtol=0.0, atol=1e-9)
    with pytest.raises(ValueError, match="sensor position"):
        model.jacobian(np.array([1.0, -2.0, 0.0]))
    with pytest.raises(ValueError, match="distinct"):
        RangeBearingMeasurementModel([0, 0], [0.1, 0.01])


def test_measurement_jacobian_follows_the_given_state_addition() -> None:
    sensor = FunctionMeasurementModel(
        measure=lambda states: states, noise_covariance=[1.0], dimension=1
    )
    mean = np.array([5.0])
    assert sensor.jacobian(mean) == pytest.approx(np.eye(1))
    # Under a multiplicative error x * exp(d), dh/dd = x.
    jacobian = sensor.jacobian(mean, lambda x, increment: x * np.exp(increment))
    assert jacobian == pytest.approx(np.array([[5.0]]))
