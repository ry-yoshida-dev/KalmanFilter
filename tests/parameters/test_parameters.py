"""Tests for the parameter containers and their access by role, name, and index."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from kalman_filter import (
    ErrorCovarianceMatrix,
    ExtendedKalmanFilter,
    KalmanGain,
    KinematicMotionModel,
    LinearMeasurementModel,
    LinearMotionModel,
    StateRole,
    StateTransitionMatrix,
    StateVector,
)


def _bounding_box_filter() -> ExtendedKalmanFilter:
    motion = KinematicMotionModel.from_state_roles(
        {StateRole.STATIC: ["u", "v", "s", "r"], StateRole.DYNAMIC: ["u", "v", "s"]},
        noise_intensities=1.0,
    )
    sensor = LinearMeasurementModel.from_observed_states(
        motion.layout, ["u", "v", "s", "r"], [1.0, 1.0, 10.0, 0.01]
    )
    return ExtendedKalmanFilter(motion, sensor, np.zeros(7), np.eye(7))


def test_state_roles_build_the_original_layout() -> None:
    kf = _bounding_box_filter()
    assert kf.motion_model.layout.names == (
        "u",
        "v",
        "s",
        "r",
        "u_velocity",
        "v_velocity",
        "s_velocity",
    )
    with pytest.raises(ValueError, match="lower role"):
        KinematicMotionModel.from_state_roles({StateRole.DYNAMIC: ["u"]}, 1.0)


def test_state_vector_access_by_role_name_and_index() -> None:
    kf = _bounding_box_filter()
    kf.x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert isinstance(kf.x, StateVector)
    assert kf.x[StateRole.STATIC, "s"] == 3.0
    assert kf.x[StateRole.DYNAMIC, "u"] == 5.0
    assert kf.x[StateRole.STATIC].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert kf.x[StateRole.DYNAMIC].tolist() == [5.0, 6.0, 7.0]
    assert kf.x[StateRole.ACCELERATION].tolist() == []
    assert kf.x["v_velocity"] == 6.0
    assert kf.x[0] == 1.0
    assert kf.x[-1] == 7.0
    assert kf.x[:2].tolist() == [1.0, 2.0]
    assert len(kf.x) == kf.x.dimension == 7
    with pytest.raises(KeyError):
        kf.x[StateRole.DYNAMIC, "r"]
    with pytest.raises(IndexError):
        kf.x[7]


def test_writes_through_the_containers_change_the_filter() -> None:
    kf = _bounding_box_filter()
    kf.x[StateRole.DYNAMIC, "s"] = 0.5
    kf.x[StateRole.STATIC] = [1.0, 2.0, 3.0, 4.0]
    kf.x["u_velocity"] = 2.0
    assert kf.x.value.tolist() == [1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.5]
    kf.P["u", "u_velocity"] = 0.3
    kf.P[1, 1] = 9.0
    assert kf.P.value[0, 4] == kf.P.value[4, 0] == 0.3
    assert kf.P.standard_deviation("v") == 3.0
    kf.predict(1.0)
    assert kf.x["u"] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="shape"):
        kf.x[StateRole.DYNAMIC] = [1.0, 2.0]
    with pytest.raises(ValueError, match="finite"):
        kf.x["u"] = float("nan")
    with pytest.raises(ValueError, match="non-negative"):
        kf.P["u", "u"] = -1.0


def test_containers_behave_like_their_arrays() -> None:
    kf = _bounding_box_filter()
    kf.predict(0.5)
    kf.update([1.0, 2.0, 3.0, 4.0])
    assert isinstance(kf.P, ErrorCovarianceMatrix)
    assert isinstance(kf.F, StateTransitionMatrix)
    assert isinstance(kf.K, KalmanGain)
    assert kf.F is not None
    assert kf.K is not None
    assert np.allclose(kf.F @ kf.P, kf.F.value @ kf.P.value)
    assert np.allclose(kf.P.value @ kf.F.T, np.asarray(kf.P) @ kf.F.value.T)
    assert kf.K.shape == (7, 4)
    assert np.trace(kf.P) == pytest.approx(np.trace(kf.P.value))


def test_model_matrices_cannot_be_changed_through_the_filter() -> None:
    motion = LinearMotionModel([[1.0, 1.0], [0.0, 1.0]], [0.1, 0.1])
    sensor = LinearMeasurementModel([[1.0, 0.0]], [1.0])
    kf = ExtendedKalmanFilter(motion, sensor, [0.0, 1.0], [1.0, 1.0])
    kf.predict()
    assert kf.F is not None
    assert kf.Q is not None
    with pytest.raises(ValueError, match="read-only"):
        kf.F.value[0, 1] = 2.0
    with pytest.raises(ValueError, match="read-only"):
        kf.Q.value[0, 0] = 2.0
    assert motion.transition_matrix[0, 1] == 1.0


def test_a_container_read_before_a_step_does_not_follow_the_new_estimate() -> None:
    kf = _bounding_box_filter()
    before = kf.x
    kf.predict(1.0)
    before["u"] = 5.0
    assert kf.x["u"] == 0.0
    kf.x["u"] = 5.0
    assert kf.x["u"] == 5.0


def test_containers_support_numpy_operators() -> None:
    kf = _bounding_box_filter()
    kf.predict(0.5)
    kf.update([1.0, 2.0, 3.0, 4.0])
    truth = np.ones(7)
    assert np.array_equal(kf.x - truth, kf.x.value - truth)
    assert np.array_equal(2.0 * kf.P, 2.0 * kf.P.value)
    assert np.array_equal(kf.x == kf.x.value, np.full(7, True))
    assert kf.y is not None
    assert len(kf.y) == 4
    assert len(kf.x) == 7


def test_in_place_operators_go_through_the_validated_setters() -> None:
    kf = _bounding_box_filter()
    kf.x += 1.0
    kf.P *= 4.0
    assert np.array_equal(kf.x.value, np.ones(7))
    assert kf.P["u", "u"] == 4.0
    with pytest.raises(ValueError, match="finite"):
        kf.x += np.nan
    assert np.array_equal(kf.x.value, np.ones(7))


def test_group_reads_are_copies() -> None:
    kf = _bounding_box_filter()
    kf.x[:2][0] = 5.0
    kf.x[StateRole.STATIC][0] = 5.0
    assert kf.x["u"] == 0.0


def test_covariance_needs_a_pair_of_keys() -> None:
    kf = _bounding_box_filter()
    with pytest.raises(KeyError, match="pair"):
        kf.P[cast(tuple[str, str], cast(object, "u"))]
