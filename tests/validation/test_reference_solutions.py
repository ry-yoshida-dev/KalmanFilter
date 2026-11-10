"""
Comparisons against reference solutions derived without the library.

Every reference here is written from the textbook equations with plain NumPy
and SciPy, so a mistake shared by several filters of the library (or by the
models) cannot cancel out the way it can in filter-to-filter comparisons.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are
from scipy.stats import multivariate_normal

from kalman_filter import (
    CovarianceUpdate,
    CubatureSigmaPointScheme,
    ExtendedKalmanFilter,
    FloatArray,
    FunctionMeasurementModel,
    FunctionMotionModel,
    IntArray,
    IteratedExtendedKalmanFilter,
    IteratedUnscentedKalmanFilter,
    KalmanFilter,
    KinematicMotionModel,
    LinearMeasurementModel,
    MerweScaledSigmaPointScheme,
    UnscentedKalmanFilter,
)

_NOISE_INTENSITY: float = 0.7
_MEASUREMENT_VARIANCES: FloatArray = np.array([0.3, 0.2])
_OBSERVATION_MATRIX: FloatArray = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
_INITIAL_MEAN: FloatArray = np.array([0.5, -1.0, 1.0, 0.3])
_INITIAL_COVARIANCE: FloatArray = np.diag([2.0, 1.5, 1.0, 0.8])
_STATE_DIMENSION: int = 4


@dataclass(frozen=True)
class _Run:
    """Priors, posteriors, and log-likelihoods recorded while running a filter."""

    prior_means: FloatArray
    prior_covariances: FloatArray
    posterior_means: FloatArray
    posterior_covariances: FloatArray
    log_likelihoods: FloatArray


def _run(kf: KalmanFilter, measurements: FloatArray, time_steps: FloatArray | float) -> _Run:
    """Update with ``measurements[0]``, then predict and update for every later step."""
    steps = measurements.shape[0]
    time_step_array = np.broadcast_to(np.asarray(time_steps, dtype=np.float64), (steps - 1,))
    prior_means: list[FloatArray] = []
    prior_covariances: list[FloatArray] = []
    posterior_means: list[FloatArray] = []
    posterior_covariances: list[FloatArray] = []
    log_likelihoods: list[float] = []
    for step, z in enumerate(measurements):
        if step > 0:
            kf.predict(float(time_step_array[step - 1]))
        prior_means.append(kf.x.value.copy())
        prior_covariances.append(kf.P.value.copy())
        kf.update(z)
        posterior_means.append(kf.x.value.copy())
        posterior_covariances.append(kf.P.value.copy())
        log_likelihoods.append(kf.log_likelihood or 0.0)
    return _Run(
        np.array(prior_means),
        np.array(prior_covariances),
        np.array(posterior_means),
        np.array(posterior_covariances),
        np.array(log_likelihoods),
    )


def _constant_velocity_matrices(dt: float, noise_intensity: float) -> tuple[FloatArray, FloatArray]:
    """Textbook ``F(dt)`` and ``Q(dt)`` of the planar white-noise-acceleration model."""
    identity = np.eye(2)
    transition = np.block([[identity, dt * identity], [np.zeros((2, 2)), identity]])
    noise = noise_intensity * np.block(
        [
            [dt**3 / 3.0 * identity, dt**2 / 2.0 * identity],
            [dt**2 / 2.0 * identity, dt * identity],
        ]
    )
    return transition, noise


def _state_indices(steps: Sequence[int]) -> IntArray:
    return np.concatenate(
        [np.arange(step * _STATE_DIMENSION, (step + 1) * _STATE_DIMENSION) for step in steps]
    )


@dataclass(frozen=True)
class _JointGaussian:
    """All states and measurements of a linear-Gaussian run as one Gaussian vector."""

    state_mean: FloatArray
    state_covariance: FloatArray
    measurement_mean: FloatArray
    measurement_covariance: FloatArray
    state_measurement_covariance: FloatArray
    measurement_steps: IntArray
    measurement_values: FloatArray

    @classmethod
    def build(cls, time_steps: FloatArray, measurements: FloatArray) -> _JointGaussian:
        steps = measurements.shape[0]
        means = [_INITIAL_MEAN]
        # blocks[k][j] = Cov(x_k, x_j) for j <= k, from x_{k+1} = F_k x_k + w_k.
        blocks: list[list[FloatArray]] = [[_INITIAL_COVARIANCE]]
        for k, dt in enumerate(time_steps):
            transition, noise = _constant_velocity_matrices(float(dt), _NOISE_INTENSITY)
            means.append(transition @ means[-1])
            covariance_row = [transition @ blocks[k][j] for j in range(k + 1)]
            covariance_row.append(transition @ blocks[k][k] @ transition.T + noise)
            blocks.append(covariance_row)
        state_covariance = np.zeros((steps * _STATE_DIMENSION, steps * _STATE_DIMENSION))
        for k in range(steps):
            for j in range(k + 1):
                rows = _state_indices([k])
                columns = _state_indices([j])
                state_covariance[np.ix_(rows, columns)] = blocks[k][j]
                state_covariance[np.ix_(columns, rows)] = blocks[k][j].T

        is_observed = ~np.isnan(measurements)
        observation_rows: list[FloatArray] = []
        variances: list[float] = []
        measurement_steps: list[int] = []
        for k in range(steps):
            for component in np.flatnonzero(is_observed[k]):
                observation_row = np.zeros(steps * _STATE_DIMENSION)
                observation_row[_state_indices([k])] = _OBSERVATION_MATRIX[component]
                observation_rows.append(observation_row)
                variances.append(float(_MEASUREMENT_VARIANCES[component]))
                measurement_steps.append(k)
        observation = np.array(observation_rows)
        state_mean = np.concatenate(means)
        return cls(
            state_mean=state_mean,
            state_covariance=state_covariance,
            measurement_mean=observation @ state_mean,
            measurement_covariance=observation @ state_covariance @ observation.T
            + np.diag(variances),
            state_measurement_covariance=state_covariance @ observation.T,
            measurement_steps=np.array(measurement_steps),
            measurement_values=measurements[is_observed],
        )

    def conditional(
        self, steps: Sequence[int], last_measurement_step: int
    ) -> tuple[FloatArray, FloatArray]:
        """Mean and covariance of the stacked states at ``steps`` given measurements so far."""
        indices = _state_indices(steps)
        used = self.measurement_steps <= last_measurement_step
        mean = self.state_mean[indices]
        covariance = self.state_covariance[np.ix_(indices, indices)]
        if not np.any(used):
            return mean, covariance
        cross = self.state_measurement_covariance[np.ix_(indices, np.flatnonzero(used))]
        innovation_covariance = self.measurement_covariance[np.ix_(used, used)]
        gain = np.linalg.solve(innovation_covariance, cross.T).T
        innovation = self.measurement_values[used] - self.measurement_mean[used]
        return mean + gain @ innovation, covariance - gain @ cross.T

    def log_evidence(self, last_measurement_step: int) -> float:
        """Log-density of all measurements up to ``last_measurement_step``."""
        used = self.measurement_steps <= last_measurement_step
        if not np.any(used):
            return 0.0
        return float(
            multivariate_normal(
                self.measurement_mean[used], self.measurement_covariance[np.ix_(used, used)]
            ).logpdf(self.measurement_values[used])
        )


def _linear_problem() -> tuple[FloatArray, FloatArray]:
    random_generator = np.random.default_rng(7)
    steps = 12
    time_steps = random_generator.uniform(0.05, 0.6, size=steps - 1)
    measurements = random_generator.normal(size=(steps, 2)) + np.arange(steps)[:, None] * 0.4
    measurements[3, 1] = np.nan
    measurements[6] = np.nan
    measurements[9, 0] = np.nan
    return time_steps, measurements


def _linear_models() -> tuple[KinematicMotionModel, LinearMeasurementModel]:
    motion = KinematicMotionModel.constant_velocity(["x", "y"], noise_intensity=_NOISE_INTENSITY)
    sensor = LinearMeasurementModel(_OBSERVATION_MATRIX, _MEASUREMENT_VARIANCES)
    return motion, sensor


_INITIAL = (_INITIAL_MEAN, _INITIAL_COVARIANCE)

_LINEAR_FILTER_FACTORIES: dict[str, Callable[[], KalmanFilter]] = {
    "extended_joseph": lambda: ExtendedKalmanFilter(*_linear_models(), *_INITIAL),
    "extended_standard": lambda: ExtendedKalmanFilter(
        *_linear_models(), *_INITIAL, covariance_update=CovarianceUpdate.STANDARD
    ),
    "iterated_extended": lambda: IteratedExtendedKalmanFilter(*_linear_models(), *_INITIAL),
    "unscented": lambda: UnscentedKalmanFilter(*_linear_models(), *_INITIAL),
    "cubature": lambda: UnscentedKalmanFilter(
        *_linear_models(), *_INITIAL, sigma_point_scheme=CubatureSigmaPointScheme()
    ),
    "iterated_unscented": lambda: IteratedUnscentedKalmanFilter(*_linear_models(), *_INITIAL),
}


@pytest.mark.parametrize("name", list(_LINEAR_FILTER_FACTORIES))
def test_filter_matches_exact_conditioning_of_the_joint_gaussian(name: str) -> None:
    time_steps, measurements = _linear_problem()
    joint = _JointGaussian.build(time_steps, measurements)
    run = _run(_LINEAR_FILTER_FACTORIES[name](), measurements, time_steps)

    for step in range(measurements.shape[0]):
        prior_mean, prior_covariance = joint.conditional([step], step - 1)
        posterior_mean, posterior_covariance = joint.conditional([step], step)
        assert np.allclose(run.prior_means[step], prior_mean, atol=1e-9)
        assert np.allclose(run.prior_covariances[step], prior_covariance, atol=1e-9)
        assert np.allclose(run.posterior_means[step], posterior_mean, atol=1e-9)
        assert np.allclose(run.posterior_covariances[step], posterior_covariance, atol=1e-9)
        step_log_likelihood = joint.log_evidence(step) - joint.log_evidence(step - 1)
        assert run.log_likelihoods[step] == pytest.approx(step_log_likelihood, abs=1e-9)
    assert np.sum(run.log_likelihoods) == pytest.approx(
        joint.log_evidence(measurements.shape[0] - 1)
    )


@pytest.mark.parametrize(
    "covariance_update",
    [CovarianceUpdate.JOSEPH, CovarianceUpdate.STANDARD],
)
def test_covariance_converges_to_the_riccati_steady_state(
    covariance_update: CovarianceUpdate,
) -> None:
    dt = 0.1
    motion, sensor = _linear_models()
    kf = ExtendedKalmanFilter(
        motion,
        sensor,
        np.zeros(_STATE_DIMENSION),
        np.eye(_STATE_DIMENSION) * 100.0,
        covariance_update=covariance_update,
    )
    run = _run(kf, np.zeros((400, 2)), dt)

    transition, noise = _constant_velocity_matrices(dt, _NOISE_INTENSITY)
    measurement_noise = np.diag(_MEASUREMENT_VARIANCES)
    prior = solve_discrete_are(transition.T, _OBSERVATION_MATRIX.T, noise, measurement_noise)
    innovation_covariance = _OBSERVATION_MATRIX @ prior @ _OBSERVATION_MATRIX.T + measurement_noise
    gain = np.linalg.solve(innovation_covariance, _OBSERVATION_MATRIX @ prior).T
    posterior = prior - gain @ innovation_covariance @ gain.T
    assert np.allclose(run.prior_covariances[-1], prior, rtol=1e-8, atol=1e-12)
    assert np.allclose(run.posterior_covariances[-1], posterior, rtol=1e-8, atol=1e-12)


# Planar pendulum: state [angle, angular velocity], bob position observed.
_PENDULUM_DT: float = 0.05
_PENDULUM_PROCESS_NOISE: FloatArray = np.diag([1e-4, 4e-3])
_PENDULUM_MEASUREMENT_NOISE: FloatArray = np.diag([0.02, 0.03])


def _pendulum_transition(states: FloatArray) -> FloatArray:
    angle = states[..., 0]
    velocity = states[..., 1]
    return np.stack(
        [angle + _PENDULUM_DT * velocity, velocity - _PENDULUM_DT * 9.81 * np.sin(angle)], axis=-1
    )


def _pendulum_transition_jacobian(state: FloatArray) -> FloatArray:
    return np.array([[1.0, _PENDULUM_DT], [-_PENDULUM_DT * 9.81 * np.cos(state[0]), 1.0]])


def _pendulum_measure(states: FloatArray) -> FloatArray:
    return np.stack([np.sin(states[..., 0]), -np.cos(states[..., 0])], axis=-1)


def _pendulum_measure_jacobian(state: FloatArray) -> FloatArray:
    return np.array([[np.cos(state[0]), 0.0], [np.sin(state[0]), 0.0]])


def _pendulum_models() -> tuple[FunctionMotionModel, FunctionMeasurementModel]:
    motion = FunctionMotionModel(
        transition=lambda states, dt, control: _pendulum_transition(states),
        process_noise=_PENDULUM_PROCESS_NOISE,
        dimension=2,
        jacobian=lambda mean, dt, control: _pendulum_transition_jacobian(mean),
    )
    sensor = FunctionMeasurementModel(
        measure=_pendulum_measure,
        noise_covariance=_PENDULUM_MEASUREMENT_NOISE,
        dimension=2,
        jacobian=_pendulum_measure_jacobian,
    )
    return motion, sensor


def _pendulum_measurements() -> FloatArray:
    random_generator = np.random.default_rng(11)
    state = np.array([1.2, 0.0])
    measurements: list[FloatArray] = []
    for _ in range(60):
        measurements.append(
            random_generator.multivariate_normal(
                _pendulum_measure(state), _PENDULUM_MEASUREMENT_NOISE
            )
        )
        state = random_generator.multivariate_normal(
            _pendulum_transition(state), _PENDULUM_PROCESS_NOISE
        )
    return np.array(measurements)


_PENDULUM_INITIAL_MEAN: FloatArray = np.array([0.8, 0.5])
_PENDULUM_INITIAL_COVARIANCE: FloatArray = np.diag([0.3, 1.0])


def _reference_unscented_run(
    measurements: FloatArray, alpha: float, beta: float, kappa: float
) -> tuple[FloatArray, FloatArray]:
    """Scaled unscented filter written directly from Wan and van der Merwe (2000)."""
    dimension = 2
    scale = alpha**2 * (dimension + kappa) - dimension
    mean_weights = np.full(2 * dimension + 1, 0.5 / (dimension + scale))
    mean_weights[0] = scale / (dimension + scale)
    covariance_weights = mean_weights.copy()
    covariance_weights[0] += 1.0 - alpha**2 + beta

    def sigma_points(mean: FloatArray, covariance: FloatArray) -> FloatArray:
        root = np.linalg.cholesky((dimension + scale) * covariance)
        return np.vstack([mean, mean + root.T, mean - root.T])

    mean = _PENDULUM_INITIAL_MEAN
    covariance = _PENDULUM_INITIAL_COVARIANCE
    means: list[FloatArray] = []
    covariances: list[FloatArray] = []
    for step, measurement in enumerate(measurements):
        if step > 0:
            propagated = _pendulum_transition(sigma_points(mean, covariance))
            mean = mean_weights @ propagated
            deviation = propagated - mean
            covariance = (covariance_weights * deviation.T) @ deviation + _PENDULUM_PROCESS_NOISE
        points = sigma_points(mean, covariance)
        predicted = _pendulum_measure(points)
        predicted_mean = mean_weights @ predicted
        measurement_deviation = predicted - predicted_mean
        innovation_covariance = (
            covariance_weights * measurement_deviation.T
        ) @ measurement_deviation + _PENDULUM_MEASUREMENT_NOISE
        cross = (covariance_weights * (points - mean).T) @ measurement_deviation
        gain = cross @ np.linalg.inv(innovation_covariance)
        mean = mean + gain @ (measurement - predicted_mean)
        covariance = covariance - gain @ innovation_covariance @ gain.T
        means.append(mean)
        covariances.append(covariance)
    return np.array(means), np.array(covariances)


@pytest.mark.parametrize(
    ("scheme", "parameters"),
    [
        (MerweScaledSigmaPointScheme(), (1.0, 2.0, 0.0)),
        (MerweScaledSigmaPointScheme(0.5, 2.0, 1.0), (0.5, 2.0, 1.0)),
        # The cubature rule is the scaled transform with alpha = 1, beta = 0, kappa = 0.
        (CubatureSigmaPointScheme(), (1.0, 0.0, 0.0)),
    ],
)
def test_unscented_filter_matches_textbook_equations(
    scheme: MerweScaledSigmaPointScheme | CubatureSigmaPointScheme,
    parameters: tuple[float, float, float],
) -> None:
    measurements = _pendulum_measurements()
    expected_means, expected_covariances = _reference_unscented_run(measurements, *parameters)
    kf = UnscentedKalmanFilter(
        *_pendulum_models(),
        _PENDULUM_INITIAL_MEAN,
        _PENDULUM_INITIAL_COVARIANCE,
        sigma_point_scheme=scheme,
    )
    run = _run(kf, measurements, 1.0)
    assert np.allclose(run.posterior_means, expected_means, atol=1e-9)
    assert np.allclose(run.posterior_covariances, expected_covariances, atol=1e-9)


@pytest.mark.parametrize(
    "covariance_update",
    [CovarianceUpdate.JOSEPH, CovarianceUpdate.STANDARD],
)
def test_extended_filter_matches_textbook_equations(covariance_update: CovarianceUpdate) -> None:
    measurements = _pendulum_measurements()
    mean = _PENDULUM_INITIAL_MEAN
    covariance = _PENDULUM_INITIAL_COVARIANCE
    expected_means: list[FloatArray] = []
    expected_covariances: list[FloatArray] = []
    for step, measurement in enumerate(measurements):
        if step > 0:
            transition_jacobian = _pendulum_transition_jacobian(mean)
            mean = _pendulum_transition(mean)
            covariance = (
                transition_jacobian @ covariance @ transition_jacobian.T + _PENDULUM_PROCESS_NOISE
            )
        measurement_jacobian = _pendulum_measure_jacobian(mean)
        innovation_covariance = (
            measurement_jacobian @ covariance @ measurement_jacobian.T + _PENDULUM_MEASUREMENT_NOISE
        )
        gain = covariance @ measurement_jacobian.T @ np.linalg.inv(innovation_covariance)
        mean = mean + gain @ (measurement - _pendulum_measure(mean))
        covariance = (np.eye(2) - gain @ measurement_jacobian) @ covariance
        expected_means.append(mean)
        expected_covariances.append(covariance)

    kf = ExtendedKalmanFilter(
        *_pendulum_models(),
        _PENDULUM_INITIAL_MEAN,
        _PENDULUM_INITIAL_COVARIANCE,
        covariance_update=covariance_update,
    )
    run = _run(kf, measurements, 1.0)
    assert np.allclose(run.posterior_means, np.array(expected_means), atol=1e-9)
    assert np.allclose(run.posterior_covariances, np.array(expected_covariances), atol=1e-9)
