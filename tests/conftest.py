"""Shared fixtures for the test suite."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from kalman_filter import (
    FloatArray,
    KinematicMotionModel,
    LinearKalmanFilter,
    LinearMeasurementModel,
    MeasurementModel,
    MotionModel,
)

type Simulate = Callable[
    [MotionModel, MeasurementModel, FloatArray, int, float, int], tuple[FloatArray, FloatArray]
]


@pytest.fixture
def constant_velocity_model() -> KinematicMotionModel:
    """Planar constant-velocity model with state ``[x, y, x_velocity, y_velocity]``."""
    return KinematicMotionModel.constant_velocity(["x", "y"], noise_intensity=0.5)


@pytest.fixture
def position_sensor(constant_velocity_model: KinematicMotionModel) -> LinearMeasurementModel:
    """Sensor observing ``x`` and ``y`` of the constant-velocity state."""
    return LinearMeasurementModel.from_observed_states(
        constant_velocity_model.layout, ["x", "y"], [0.25, 0.25]
    )


@pytest.fixture
def initial_x() -> FloatArray:
    """Initial constant-velocity state."""
    return np.array([0.0, 0.0, 1.0, 0.5])


@pytest.fixture
def initial_covariance() -> FloatArray:
    """Broad initial covariance of the constant-velocity state."""
    return np.eye(4)


@pytest.fixture
def linear_filter(
    constant_velocity_model: KinematicMotionModel,
    position_sensor: LinearMeasurementModel,
    initial_x: FloatArray,
    initial_covariance: FloatArray,
) -> LinearKalmanFilter:
    """Standard Kalman filter on a broad prior."""
    return LinearKalmanFilter(
        constant_velocity_model, position_sensor, initial_x, initial_covariance
    )


def _simulate(
    motion: MotionModel,
    sensor: MeasurementModel,
    initial: FloatArray,
    steps: int,
    dt: float,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """True states ``(steps, n)`` from ``initial`` and noisy measurements ``(steps, m)``."""
    generator = np.random.default_rng(seed)
    states = np.empty((steps, motion.dimension))
    states[0] = initial
    for step in range(1, steps):
        noise = generator.multivariate_normal(
            np.zeros(motion.dimension), motion.process_noise(states[step - 1], dt)
        )
        states[step] = motion.add(motion.transition(states[step - 1], dt), noise)
    standard_normal = generator.standard_normal((steps, sensor.dimension))
    measurements = np.empty((steps, sensor.dimension))
    for step, state in enumerate(states):
        # Eigendecomposition instead of Cholesky so singular noise is allowed.
        eigenvalues, eigenvectors = np.linalg.eigh(sensor.noise_covariance(state))
        root = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
        measurements[step] = sensor.measure(state) + root @ standard_normal[step]
    return states, measurements


@pytest.fixture
def simulate() -> Simulate:
    """Ground truth and measurement sampler ``simulate(motion, sensor, x0, steps, dt, seed)``."""
    return _simulate
