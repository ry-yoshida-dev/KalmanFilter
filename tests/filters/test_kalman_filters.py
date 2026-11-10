"""Tests for the linear, extended, and unscented Kalman filters."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from kalman_filter import (
    ConstantTurnRateVelocityMotionModel,
    CovarianceUpdate,
    CubatureSigmaPointScheme,
    ExtendedKalmanFilter,
    FifthDegreeCubatureSigmaPointScheme,
    FloatArray,
    FunctionMeasurementModel,
    FunctionMotionModel,
    GaussHermiteSigmaPointScheme,
    IteratedExtendedKalmanFilter,
    IteratedUnscentedKalmanFilter,
    KalmanFilter,
    KinematicMotionModel,
    LinearKalmanFilter,
    LinearMeasurementModel,
    LinearMotionModel,
    MerweScaledSigmaPointScheme,
    RangeBearingMeasurementModel,
    SigmaPointScheme,
    UnscentedKalmanFilter,
)

type Simulate = Callable[..., tuple[FloatArray, FloatArray]]


def test_scalar_filter_matches_closed_form() -> None:
    motion = LinearMotionModel([[1.0]], [0.5])
    sensor = LinearMeasurementModel([[1.0]], [2.0])
    kf = ExtendedKalmanFilter(motion, sensor, x=[0.0], P=[1.0])
    kf.predict()
    assert kf.P[0, 0] == pytest.approx(1.5)
    assert kf.F is not None
    assert kf.F.value.tolist() == [[1.0]]
    assert kf.Q is not None
    assert kf.Q.value.tolist() == [[0.5]]

    kf.update([3.0])
    gain = 1.5 / 3.5
    assert kf.K is not None
    assert kf.S is not None
    assert kf.y is not None
    assert kf.K.value[0, 0] == pytest.approx(gain)
    assert kf.S.value[0, 0] == pytest.approx(3.5)
    assert kf.y.value[0] == pytest.approx(3.0)
    assert kf.H is not None
    assert kf.H.value.tolist() == [[1.0]]
    assert kf.R is not None
    assert kf.R.value.tolist() == [[2.0]]
    assert kf.x[0] == pytest.approx(3.0 * gain)
    assert kf.P[0, 0] == pytest.approx((1.0 - gain) * 1.5)
    expected_log_likelihood = -0.5 * (9.0 / 3.5 + np.log(2.0 * np.pi * 3.5))
    assert kf.log_likelihood == pytest.approx(expected_log_likelihood)


def test_parameters_are_none_before_the_first_step(linear_filter: LinearKalmanFilter) -> None:
    assert linear_filter.F is None
    assert linear_filter.Q is None
    assert linear_filter.H is None
    assert linear_filter.K is None
    assert linear_filter.S is None


def test_estimate_can_be_replaced(linear_filter: LinearKalmanFilter) -> None:
    linear_filter.x = [1.0, 2.0, 3.0, 4.0]
    linear_filter.P = [2.0, 2.0, 2.0, 2.0]
    assert linear_filter.x.value.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert np.array_equal(linear_filter.P, 2.0 * np.eye(4))
    with pytest.raises(ValueError, match="length 4"):
        linear_filter.x = [1.0]
    with pytest.raises(ValueError, match="symmetric"):
        linear_filter.P = np.triu(np.ones((4, 4)))


@pytest.mark.parametrize(
    "scheme",
    [
        MerweScaledSigmaPointScheme(),
        MerweScaledSigmaPointScheme(0.3, 2.0, 1.0),
        CubatureSigmaPointScheme(),
        FifthDegreeCubatureSigmaPointScheme(),
        GaussHermiteSigmaPointScheme(),
    ],
)
def test_unscented_equals_extended_on_linear_models(
    linear_filter: LinearKalmanFilter,
    scheme: SigmaPointScheme,
) -> None:
    unscented = UnscentedKalmanFilter(
        linear_filter.motion_model,
        linear_filter.measurement_model,
        linear_filter.x,
        linear_filter.P,
        sigma_point_scheme=scheme,
    )
    linear_filter.predict(0.1)
    unscented.predict(0.1)
    assert np.allclose(linear_filter.x, unscented.x)
    assert np.allclose(linear_filter.P, unscented.P)
    # The statistical linearization of a linear model is the model's matrix.
    assert linear_filter.F is not None
    assert unscented.F is not None
    assert np.allclose(linear_filter.F, unscented.F)

    linear_filter.update([0.3, -0.2])
    unscented.update([0.3, -0.2])
    assert np.allclose(linear_filter.x, unscented.x)
    assert np.allclose(linear_filter.P, unscented.P)
    assert linear_filter.K is not None
    assert unscented.K is not None
    assert linear_filter.S is not None
    assert unscented.S is not None
    assert np.allclose(linear_filter.K.value, unscented.K.value)
    assert np.allclose(linear_filter.S.value, unscented.S.value)
    assert linear_filter.log_likelihood == pytest.approx(unscented.log_likelihood)
    assert linear_filter.H is not None
    assert unscented.H is not None
    assert np.allclose(linear_filter.H, unscented.H)


def test_joseph_and_standard_forms_agree(linear_filter: LinearKalmanFilter) -> None:
    standard = ExtendedKalmanFilter(
        linear_filter.motion_model,
        linear_filter.measurement_model,
        linear_filter.x,
        linear_filter.P,
        covariance_update=CovarianceUpdate.STANDARD,
    )
    linear_filter.update([0.1, 0.2])
    standard.update([0.1, 0.2])
    assert np.allclose(linear_filter.P, standard.P)


def test_nan_components_are_skipped(
    linear_filter: LinearKalmanFilter,
    constant_velocity_model: KinematicMotionModel,
) -> None:
    reduced_sensor = LinearMeasurementModel.from_observed_states(
        constant_velocity_model.layout, ["x"], [0.25]
    )
    reduced = ExtendedKalmanFilter(
        constant_velocity_model, reduced_sensor, linear_filter.x, linear_filter.P
    )
    linear_filter.update([0.4, np.nan])
    reduced.update([0.4])
    assert linear_filter.is_observed is not None
    assert linear_filter.is_observed.tolist() == [True, False]
    assert np.allclose(linear_filter.x, reduced.x)
    assert np.allclose(linear_filter.P, reduced.P)
    assert linear_filter.H is not None
    assert reduced.H is not None
    assert np.allclose(linear_filter.H.value, reduced.H.value)
    assert linear_filter.log_likelihood == pytest.approx(reduced.log_likelihood)


def test_missing_measurement_keeps_the_prior(linear_filter: LinearKalmanFilter) -> None:
    x, P = linear_filter.x.value.copy(), linear_filter.P.value.copy()
    linear_filter.update([np.nan, np.nan])
    assert np.array_equal(linear_filter.x, x)
    assert np.array_equal(linear_filter.P, P)
    assert linear_filter.K is not None
    assert linear_filter.K.shape == (4, 0)
    assert linear_filter.log_likelihood == 0.0


def test_failed_update_leaves_the_filter_unchanged() -> None:
    motion = LinearMotionModel(np.eye(2), [0.0, 0.0])
    sensor = LinearMeasurementModel([[1.0, 0.0], [0.0, 1.0]], [0.0, 1.0])
    kf = LinearKalmanFilter(motion, sensor, x=[1.0, 2.0], P=[0.0, 1.0])
    kf.update([np.nan, 2.5])
    x, P = kf.x.value.copy(), kf.P.value.copy()
    assert kf.is_observed is not None
    observed_before = kf.is_observed.copy()
    # The first component is exactly known and exactly measured, so S is singular.
    with pytest.raises(np.linalg.LinAlgError):
        kf.update([1.0, 2.0])
    assert np.array_equal(kf.x, x)
    assert np.array_equal(kf.P, P)
    assert np.array_equal(kf.is_observed, observed_before)
    assert kf.K is not None
    assert kf.K.shape == (2, 1)


def test_fading_memory_inflates_prior(linear_filter: LinearKalmanFilter) -> None:
    fading = ExtendedKalmanFilter(
        linear_filter.motion_model,
        linear_filter.measurement_model,
        linear_filter.x,
        linear_filter.P,
        fading_memory_factor=1.1,
    )
    linear_filter.predict(0.1)
    fading.predict(0.1)
    assert np.trace(fading.P) > np.trace(linear_filter.P)
    with pytest.raises(ValueError, match="fading_memory_factor"):
        ExtendedKalmanFilter(
            linear_filter.motion_model,
            linear_filter.measurement_model,
            linear_filter.x,
            linear_filter.P,
            fading_memory_factor=0.9,
        )


def test_dimension_mismatch_is_rejected(linear_filter: LinearKalmanFilter) -> None:
    with pytest.raises(ValueError, match="length 4"):
        ExtendedKalmanFilter(
            linear_filter.motion_model, linear_filter.measurement_model, [0.0], np.eye(4)
        )
    with pytest.raises(ValueError, match="length 2"):
        linear_filter.update([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="dt"):
        linear_filter.predict(float("nan"))
    with pytest.raises(ValueError, match="vector"):
        linear_filter.update(np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        linear_filter.x = [0.0, np.nan, 0.0, 0.0]
    # A column vector is a vector.
    linear_filter.update(np.array([[0.1], [0.2]]))


def test_invalid_steps_are_rejected(linear_filter: LinearKalmanFilter) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        linear_filter.predict(-0.1)
    with pytest.raises(ValueError, match="infinite"):
        linear_filter.update([np.inf, 0.0])
    with pytest.raises(ValueError, match="positive semi-definite"):
        linear_filter.P = [
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]


def test_linear_filter_accepts_only_linear_models(
    linear_filter: LinearKalmanFilter,
    constant_velocity_model: KinematicMotionModel,
    initial_x: FloatArray,
) -> None:
    range_sensor = RangeBearingMeasurementModel([0, 1], [0.1, 0.01])
    with pytest.raises(TypeError, match="LinearMeasurementModel"):
        # The cast hides the wrong type from the type checkers, as for an untyped caller.
        LinearKalmanFilter(
            constant_velocity_model,
            cast(LinearMeasurementModel, cast(object, range_sensor)),
            initial_x,
            np.eye(4),
        )
    with pytest.raises(TypeError, match="LinearMeasurementModel"):
        linear_filter.update([1.0, 0.1], range_sensor)
    three_columns = LinearMeasurementModel([[1.0, 0.0, 0.0]], [1.0])
    with pytest.raises(ValueError, match="4 columns"):
        linear_filter.update([1.0], three_columns)


def test_prediction_does_not_share_memory_with_the_model() -> None:
    target = np.array([1.0, 2.0])
    motion = FunctionMotionModel(
        lambda states, dt, u: target,
        [0.1, 0.1],
        dimension=2,
        jacobian=lambda mean, dt, u: np.zeros((2, 2)),
    )
    sensor = LinearMeasurementModel([[1.0, 0.0]], [1.0])
    kf = ExtendedKalmanFilter(motion, sensor, [0.0, 0.0], [1.0, 1.0])
    kf.predict()
    kf.x[0] = 5.0
    assert target[0] == 1.0


def test_function_model_that_is_not_vectorized_gets_a_hint() -> None:
    motion = FunctionMotionModel(
        lambda x, dt, u: np.array([x[0] + dt * x[1], x[1]]), [0.1, 0.1], dimension=2
    )
    sensor = LinearMeasurementModel([[1.0, 0.0]], [1.0])
    kf = UnscentedKalmanFilter(motion, sensor, [0.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="is_vectorized=False"):
        kf.predict(0.1)


def test_measurement_of_the_wrong_size_is_rejected(
    constant_velocity_model: KinematicMotionModel, initial_x: FloatArray
) -> None:
    wrong_size = FunctionMeasurementModel(
        measure=lambda states: states[..., :1], noise_covariance=[1.0, 1.0], dimension=2
    )
    kf = ExtendedKalmanFilter(constant_velocity_model, wrong_size, initial_x, np.eye(4))
    with pytest.raises(ValueError, match="measure returned shape"):
        kf.update([0.0, 0.0])


def test_control_input_must_be_finite_and_sized() -> None:
    motion = LinearMotionModel(np.eye(1), [0.01], control_matrix=[[1.0]])
    kf = ExtendedKalmanFilter(motion, LinearMeasurementModel([[1.0]], [1.0]), [1.0], [1.0])
    with pytest.raises(ValueError, match="u must be finite"):
        kf.predict(u=[np.nan])
    with pytest.raises(ValueError, match="control must have length 1"):
        kf.predict(u=[1.0, 2.0])


def test_filter_is_consistent_on_simulated_data(
    linear_filter: LinearKalmanFilter, simulate: Simulate
) -> None:
    initial_x, initial_P = linear_filter.x.value.copy(), linear_filter.P.value.copy()
    nees_values: list[float] = []
    nis_values: list[float] = []
    for seed in range(30):
        states, measurements = simulate(
            linear_filter.motion_model, linear_filter.measurement_model, initial_x, 60, 0.1, seed
        )
        linear_filter.x, linear_filter.P = initial_x, initial_P
        for step, z in enumerate(measurements):
            if step > 0:
                linear_filter.predict(0.1)
            linear_filter.update(z)
            assert linear_filter.y is not None
            assert linear_filter.S is not None
            error = states[step] - linear_filter.x
            nees_values.append(float(error @ np.linalg.solve(linear_filter.P, error)))
            nis_values.append(
                float(linear_filter.y @ np.linalg.solve(linear_filter.S, linear_filter.y))
            )
    # Expected NEES is n = 4 and expected NIS is m = 2 for a consistent filter.
    assert np.mean(nees_values) == pytest.approx(4.0, rel=0.1)
    assert np.mean(nis_values) == pytest.approx(2.0, rel=0.1)


def test_control_input_is_forwarded() -> None:
    motion = LinearMotionModel(np.eye(1), [0.01], control_matrix=[[1.0]])
    kf = ExtendedKalmanFilter(motion, LinearMeasurementModel([[1.0]], [1.0]), [1.0], [1.0])
    kf.predict(u=[2.5])
    assert kf.x[0] == pytest.approx(3.5)


@pytest.mark.parametrize(
    "filter_type",
    ["extended", "unscented", "cubature", "iterated_extended", "iterated_unscented"],
)
def test_nonlinear_filters_track_turning_target(filter_type: str, simulate: Simulate) -> None:
    motion = ConstantTurnRateVelocityMotionModel(
        acceleration_variance=0.05, yaw_acceleration_variance=0.001
    )
    radar = RangeBearingMeasurementModel((0, 1), [0.25, 0.0004], sensor_position=(-20.0, 0.0))
    truth_start = np.array([0.0, 0.0, 5.0, 0.0, 0.3])
    states, measurements = simulate(motion, radar, truth_start, 100, 0.1, 3)
    x = truth_start + np.array([0.5, -0.5, 0.5, 0.1, -0.1])
    P = np.diag([1.0, 1.0, 1.0, 0.1, 0.05])
    kf: KalmanFilter
    match filter_type:
        case "extended":
            kf = ExtendedKalmanFilter(motion, radar, x, P)
        case "unscented":
            kf = UnscentedKalmanFilter(motion, radar, x, P)
        case "iterated_extended":
            kf = IteratedExtendedKalmanFilter(motion, radar, x, P)
        case "iterated_unscented":
            kf = IteratedUnscentedKalmanFilter(motion, radar, x, P)
        case _:
            kf = UnscentedKalmanFilter(
                motion, radar, x, P, sigma_point_scheme=CubatureSigmaPointScheme()
            )
    errors: list[FloatArray] = []
    for step, z in enumerate(measurements):
        if step > 0:
            kf.predict(0.1)
        kf.update(z)
        if step >= 50:
            errors.append(kf.x[:2] - states[step, :2])
    assert np.sqrt(np.mean(np.square(errors))) < 0.3


@pytest.mark.parametrize("filter_type", ["extended", "unscented"])
def test_update_keeps_yaw_wrapped(filter_type: str) -> None:
    motion = ConstantTurnRateVelocityMotionModel(0.1, 0.1)
    sensor = LinearMeasurementModel.from_observed_states(motion.layout, ["x", "y"], [0.1, 0.1])
    P = np.eye(5)
    P[0, 3] = P[3, 0] = 0.9
    x = [0.0, 0.0, 1.0, 3.1, 0.0]
    kf: KalmanFilter = (
        ExtendedKalmanFilter(motion, sensor, x, P)
        if filter_type == "extended"
        else UnscentedKalmanFilter(motion, sensor, x, P)
    )
    kf.update([1.0, 0.0])
    assert -np.pi <= kf.x[3] < np.pi


@pytest.mark.parametrize(
    "scheme",
    [
        MerweScaledSigmaPointScheme(),
        CubatureSigmaPointScheme(),
        FifthDegreeCubatureSigmaPointScheme(),
        GaussHermiteSigmaPointScheme(),
    ],
    ids=["merwe", "cubature", "fifth_degree_cubature", "gauss_hermite"],
)
def test_unscented_accepts_exactly_known_components(
    linear_filter: LinearKalmanFilter,
    scheme: SigmaPointScheme,
) -> None:
    x, P = [0.0, 0.0, 1.0, 0.5], [1.0, 1.0, 0.0, 0.0]
    linear_filter.x, linear_filter.P = x, P
    unscented = UnscentedKalmanFilter(
        linear_filter.motion_model,
        linear_filter.measurement_model,
        x,
        P,
        sigma_point_scheme=scheme,
    )
    for kf in (linear_filter, unscented):
        kf.predict(0.1)
        kf.update([0.1, 0.0])
    assert np.allclose(unscented.x, linear_filter.x)
    assert np.allclose(unscented.P, linear_filter.P)


def test_unscented_filter_uses_the_measurement_mean_of_a_function_model() -> None:
    # The bearing to a target behind the sensor straddles +/- pi, so sigma point bearings must
    # be averaged on the circle; a function model with the same residual and mean must agree
    # with the built-in range-bearing model.
    motion = LinearMotionModel(np.eye(2), [0.01, 0.01])
    radar = RangeBearingMeasurementModel((0, 1), [0.01, 0.001])
    wrapped = FunctionMeasurementModel(
        radar.measure, [0.01, 0.001], dimension=2, residual=radar.residual, mean=radar.mean
    )
    unwrapped = FunctionMeasurementModel(
        radar.measure, [0.01, 0.001], dimension=2, residual=radar.residual
    )
    x, P, z = [-10.0, 0.0], [1.0, 1.0], [10.0, np.pi - 0.01]
    reference = UnscentedKalmanFilter(motion, radar, x, P)
    reference.update(z)
    for sensor, is_expected_equal in ((wrapped, True), (unwrapped, False)):
        kf = UnscentedKalmanFilter(motion, sensor, x, P)
        kf.update(z)
        assert np.allclose(kf.x, reference.x) is is_expected_equal


def test_unscented_linearization_approaches_the_jacobians_for_small_covariance() -> None:
    motion = ConstantTurnRateVelocityMotionModel(1.0, 0.1)
    sensor = RangeBearingMeasurementModel([0, 1], [0.1, 0.01])
    x = np.array([10.0, 5.0, 2.0, 0.3, 0.2])
    small = 1e-6 * np.eye(5)
    unscented = UnscentedKalmanFilter(motion, sensor, x, small)
    unscented.predict(0.5)
    assert unscented.F is not None
    assert np.allclose(unscented.F, motion.jacobian(x, 0.5), atol=1e-5)
    prior = unscented.x.value
    unscented.update([11.0, 0.5])
    assert unscented.H is not None
    assert np.allclose(unscented.H, sensor.jacobian(prior), atol=1e-5)
