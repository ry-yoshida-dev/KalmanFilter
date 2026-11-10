"""Tests for the continuous-time motion model."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from kalman_filter import (
    ContinuousTimeMotionModel,
    ExtendedKalmanFilter,
    FloatArray,
    KinematicMotionModel,
    LinearMeasurementModel,
    UnscentedKalmanFilter,
)

type Simulate = Callable[..., tuple[FloatArray, FloatArray]]


def _constant_velocity_drift(states: FloatArray, _: FloatArray | None) -> FloatArray:
    return np.stack([states[..., 1], np.zeros_like(states[..., 1])], axis=-1)


def _van_der_pol_drift(states: FloatArray, _: FloatArray | None) -> FloatArray:
    position, velocity = states[..., 0], states[..., 1]
    return np.stack([velocity, (1.0 - position**2) * velocity - position], axis=-1)


def test_linear_drift_reproduces_the_kinematic_model() -> None:
    continuous = ContinuousTimeMotionModel(
        drift=_constant_velocity_drift,
        noise_spectral_density=[[0.7]],
        noise_gain=[[0.0], [1.0]],
        dimension=2,
    )
    kinematic = KinematicMotionModel.constant_velocity(["x"], noise_intensity=0.7)
    state = np.array([1.0, -2.0])
    assert np.allclose(continuous.transition(state, 0.3), kinematic.transition(state, 0.3))
    assert np.allclose(continuous.jacobian(state, 0.3), kinematic.jacobian(state, 0.3))
    assert np.allclose(
        continuous.process_noise(state, 0.3), kinematic.process_noise(state, 0.3), atol=1e-12
    )


def test_exponential_decay_matches_closed_form() -> None:
    model = ContinuousTimeMotionModel(
        drift=lambda states, control: -2.0 * states,
        noise_spectral_density=[0.5],
        dimension=1,
        drift_jacobian=lambda mean: np.array([[-2.0]]),
    )
    states = np.array([[1.0], [3.0]])
    assert np.allclose(model.transition(states, 0.7), states * np.exp(-1.4), rtol=1e-9)
    expected_variance = 0.5 * (1.0 - np.exp(-2.8)) / 4.0
    assert model.process_noise(np.array([1.0]), 0.7)[0, 0] == pytest.approx(expected_variance)
    assert np.allclose(model.transition(states, 0.0), states)


@pytest.mark.parametrize("filter_type", [ExtendedKalmanFilter, UnscentedKalmanFilter])
def test_filters_track_a_van_der_pol_oscillator(
    filter_type: type[ExtendedKalmanFilter] | type[UnscentedKalmanFilter],
    simulate: Simulate,
) -> None:
    motion = ContinuousTimeMotionModel(
        drift=_van_der_pol_drift,
        noise_spectral_density=[[0.05]],
        noise_gain=[[0.0], [1.0]],
        dimension=2,
        names=["position", "velocity"],
    )
    sensor = LinearMeasurementModel.from_observed_states(motion.layout, ["position"], [0.01])
    states, measurements = simulate(motion, sensor, np.array([2.0, 0.0]), 200, 0.05, 4)
    kf = filter_type(motion, sensor, [1.5, 0.5], [0.5, 0.5])
    velocity_errors: list[float] = []
    for step, z in enumerate(measurements):
        if step > 0:
            kf.predict(0.05)
        kf.update(z)
        if step >= 50:
            velocity_errors.append(float(kf.x[1] - states[step, 1]))
    assert np.sqrt(np.mean(np.square(velocity_errors))) < 0.3


def test_non_vectorized_drift_matches_vectorized() -> None:
    vectorized = ContinuousTimeMotionModel(_van_der_pol_drift, [0.1, 0.1], dimension=2)
    looped = ContinuousTimeMotionModel(
        _van_der_pol_drift, [0.1, 0.1], dimension=2, is_vectorized=False
    )
    states = np.array([[1.0, 0.0], [0.5, -1.0]])
    assert np.allclose(vectorized.transition(states, 0.1), looped.transition(states, 0.1))


def test_continuous_time_model_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="max_step"):
        ContinuousTimeMotionModel(_van_der_pol_drift, [0.1, 0.1], dimension=2, max_step=0.0)
    with pytest.raises(ValueError, match="noise_gain"):
        ContinuousTimeMotionModel(
            _van_der_pol_drift, [[0.1]], dimension=2, noise_gain=[[1.0], [0.0], [0.0]]
        )
    with pytest.raises(ValueError, match="noise_spectral_density"):
        ContinuousTimeMotionModel(_van_der_pol_drift, [0.1, 0.1, 0.1], dimension=2)
