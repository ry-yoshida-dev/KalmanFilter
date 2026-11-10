"""Accuracy and validity of the covariance forms on ill-conditioned problems."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from kalman_filter import (
    CovarianceUpdate,
    ExtendedKalmanFilter,
    FloatArray,
    KalmanFilter,
    KinematicMotionModel,
    LinearMeasurementModel,
    LinearMotionModel,
    UnscentedKalmanFilter,
)

_FilterFactory = Callable[
    [KinematicMotionModel | LinearMotionModel, LinearMeasurementModel, FloatArray, FloatArray],
    KalmanFilter,
]

_FILTER_FACTORIES: dict[str, _FilterFactory] = {
    "extended_joseph": lambda motion, sensor, x, P: ExtendedKalmanFilter(motion, sensor, x, P),
    "extended_standard": lambda motion, sensor, x, P: ExtendedKalmanFilter(
        motion, sensor, x, P, covariance_update=CovarianceUpdate.STANDARD
    ),
    "unscented": lambda motion, sensor, x, P: UnscentedKalmanFilter(motion, sensor, x, P),
}
_STABLE_FORMS = ("extended_joseph",)


def _posteriors(
    kf: KalmanFilter, measurements: FloatArray, dt: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Posterior means and covariances of every step, and the last prior covariance."""
    means: list[FloatArray] = []
    covariances: list[FloatArray] = []
    prior_covariance = kf.P.value
    for step, z in enumerate(measurements):
        if step > 0:
            kf.predict(dt)
        prior_covariance = kf.P.value.copy()
        kf.update(z)
        means.append(kf.x.value.copy())
        covariances.append(kf.P.value.copy())
    return np.array(means), np.array(covariances), prior_covariance


@pytest.mark.parametrize("name", _STABLE_FORMS)
@pytest.mark.parametrize("ratio_exponent", [4, 8, 12])
def test_precise_measurement_on_a_vague_prior_keeps_its_variance(
    name: str, ratio_exponent: int
) -> None:
    # The posterior variance of the observed combination is s R / (s + R); the covariance
    # forms that avoid P - K S K^T keep it to the expected rounding of P.
    prior_variance = 10.0 ** (ratio_exponent / 2)
    noise_variance = 10.0 ** (-ratio_exponent / 2)
    observation = np.array([[1.0, 0.5]])
    motion = LinearMotionModel(np.eye(2), np.zeros((2, 2)))
    sensor = LinearMeasurementModel(observation, [noise_variance])
    kf = _FILTER_FACTORIES[name](
        motion, sensor, np.zeros(2), np.diag([prior_variance, prior_variance])
    )
    kf.update([1.0])
    posterior_covariance = kf.P.value

    projected_variance = 1.25 * prior_variance
    expected = projected_variance * noise_variance / (projected_variance + noise_variance)
    actual = (observation @ posterior_covariance @ observation.T).item()
    assert actual == pytest.approx(expected, rel=1e-4)
    assert np.min(np.linalg.eigvalsh(posterior_covariance)) > 0.0


def _tiny_noise_models() -> tuple[KinematicMotionModel, LinearMeasurementModel]:
    motion = KinematicMotionModel.constant_velocity(["x", "y"], noise_intensity=1e-10)
    sensor = LinearMeasurementModel.from_observed_states(motion.layout, ["x", "y"], [1e-6, 1e-6])
    return motion, sensor


@pytest.mark.parametrize("name", list(_FILTER_FACTORIES))
def test_long_run_with_tiny_noise_stays_positive_definite_and_converges(name: str) -> None:
    motion, sensor = _tiny_noise_models()
    kf = _FILTER_FACTORIES[name](motion, sensor, np.zeros(4), np.diag(np.full(4, 1e8)))
    _, covariances, last_prior_covariance = _posteriors(kf, np.zeros((3000, 2)), 1.0)

    assert np.allclose(covariances, np.swapaxes(covariances, 1, 2), rtol=0.0, atol=0.0)
    assert np.min(np.linalg.eigvalsh(covariances)) > 0.0
    steady_prior = solve_discrete_are(
        motion.transition_matrix(1.0).T,
        sensor.jacobian(np.zeros(4)).T,
        motion.process_noise(np.zeros(4), 1.0),
        sensor.noise_covariance(np.zeros(4)),
    )
    scale = float(np.max(np.abs(steady_prior)))
    assert np.allclose(last_prior_covariance, steady_prior, rtol=0.0, atol=1e-6 * scale)


@pytest.mark.parametrize("name", list(_FILTER_FACTORIES))
def test_large_coordinates_do_not_change_the_estimate(name: str) -> None:
    # A shift of every position by 1e7 must shift the means and leave the covariances alone,
    # up to the rounding of the shifted numbers themselves.
    motion = KinematicMotionModel.constant_velocity(["x", "y"], noise_intensity=0.5)
    sensor = LinearMeasurementModel.from_observed_states(motion.layout, ["x", "y"], [0.25, 0.25])
    measurements = np.random.default_rng(3).normal(size=(50, 2))
    offset: FloatArray = np.array([1e7, -1e7, 0.0, 0.0])
    initial_mean = np.array([0.0, 0.0, 1.0, 1.0])

    near_filter = _FILTER_FACTORIES[name](motion, sensor, initial_mean, np.eye(4))
    far_filter = _FILTER_FACTORIES[name](motion, sensor, initial_mean + offset, np.eye(4))
    near_means, near_covariances, _ = _posteriors(near_filter, measurements, 0.1)
    far_means, far_covariances, _ = _posteriors(far_filter, measurements + offset[:2], 0.1)
    assert np.allclose(far_means - offset, near_means, rtol=0.0, atol=1e-7)
    assert np.allclose(far_covariances, near_covariances, rtol=0.0, atol=1e-8)
