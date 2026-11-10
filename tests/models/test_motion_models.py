"""Tests for the motion models."""

from __future__ import annotations

import numpy as np
import pytest

from kalman_filter import (
    ConstantTurnRateVelocityMotionModel,
    FloatArray,
    FunctionMotionModel,
    KinematicMotionModel,
    LinearMotionModel,
    MotionModel,
    StateRole,
)


def test_constant_velocity_matrices_match_closed_form() -> None:
    model = KinematicMotionModel.constant_velocity(["x"], noise_intensity=2.0)
    dt = 0.5
    assert np.allclose(model.transition_matrix(dt), [[1.0, dt], [0.0, 1.0]])
    expected_noise = 2.0 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    assert np.allclose(model.process_noise(np.zeros(2), dt), expected_noise)


def test_constant_acceleration_matrices_match_closed_form() -> None:
    model = KinematicMotionModel.constant_acceleration(["x"], noise_intensity=1.0)
    dt = 0.3
    expected_transition = [[1.0, dt, dt**2 / 2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]]
    expected_noise = [
        [dt**5 / 20, dt**4 / 8, dt**3 / 6],
        [dt**4 / 8, dt**3 / 3, dt**2 / 2],
        [dt**3 / 6, dt**2 / 2, dt],
    ]
    assert np.allclose(model.transition_matrix(dt), expected_transition)
    assert np.allclose(model.process_noise(np.zeros(3), dt), expected_noise)


def test_kinematic_layout_supports_bounding_box_state() -> None:
    model = KinematicMotionModel({"u": 1, "v": 1, "s": 1, "r": 0}, noise_intensities=1.0)
    assert model.layout.names == ("u", "v", "s", "r", "u_velocity", "v_velocity", "s_velocity")
    transition = model.transition_matrix(1.0)
    assert transition[model.layout.index("u"), model.layout.index("u_velocity")] == 1.0
    assert transition[model.layout.index("r"), :].tolist() == [0, 0, 0, 1, 0, 0, 0]


def test_kinematic_batch_transition_matches_single() -> None:
    model = KinematicMotionModel.constant_velocity(["x", "y"], noise_intensity=1.0)
    states = np.random.default_rng(0).normal(size=(5, 4))
    batch = model.transition(states, 0.2)
    single = np.stack([model.transition(row, 0.2) for row in states])
    assert np.allclose(batch, single)


def test_kinematic_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="derivative order"):
        KinematicMotionModel({"x": 4}, noise_intensities=1.0)
    with pytest.raises(KeyError):
        KinematicMotionModel({"x": 1, "y": 1}, noise_intensities={"x": 1.0})
    with pytest.raises(ValueError, match="control"):
        KinematicMotionModel({"x": 1}, 1.0).transition(np.zeros(2), 1.0, np.ones(1))
    with pytest.raises(KeyError, match="unknown"):
        KinematicMotionModel({"x": 1}, noise_intensities={"x": 1.0, "typo": 1.0})
    with pytest.raises(ValueError, match="unique"):
        KinematicMotionModel.from_state_roles(
            {StateRole.STATIC: ["x"], StateRole.DYNAMIC: ["x", "x"]}, 1.0
        )


def test_linear_motion_model_applies_control() -> None:
    model = LinearMotionModel(
        transition_matrix=[[1.0, 1.0], [0.0, 1.0]],
        process_noise_covariance=[0.1, 0.1],
        control_matrix=[[0.5], [1.0]],
    )
    propagated = model.transition(np.array([0.0, 1.0]), 1.0, np.array([2.0]))
    assert np.allclose(propagated, [2.0, 3.0])
    with pytest.raises(ValueError, match="square"):
        LinearMotionModel([[1.0, 0.0]], [1.0])


@pytest.mark.parametrize("yaw_rate", [0.8, -0.3, 1e-6, 0.0])
def test_constant_turn_rate_velocity_analytic_jacobian(yaw_rate: float) -> None:
    model = ConstantTurnRateVelocityMotionModel(1.0, 0.1)
    for mean in ([1.0, -2.0, 3.0, 0.7, yaw_rate], [100.0, -50.0, 12.0, -2.5, yaw_rate]):
        analytic = model.jacobian(np.array(mean), 0.5)
        numerical = MotionModel.jacobian(model, np.array(mean), 0.5)
        assert np.allclose(analytic, numerical, rtol=0.0, atol=1e-8)


def test_constant_turn_rate_velocity_straight_branch_matches_arc() -> None:
    model = ConstantTurnRateVelocityMotionModel(1.0, 0.1)
    speed, yaw, yaw_rate, dt = 10.0, 0.3, 0.99e-4, 1.0
    propagated = model.transition(np.array([0.0, 0.0, speed, yaw, yaw_rate]), dt)
    arc_x = speed / yaw_rate * (np.sin(yaw + yaw_rate * dt) - np.sin(yaw))
    arc_y = speed / yaw_rate * (np.cos(yaw) - np.cos(yaw + yaw_rate * dt))
    assert np.allclose(propagated[:2], [arc_x, arc_y], atol=1e-8)


def test_constant_turn_rate_velocity_full_circle() -> None:
    model = ConstantTurnRateVelocityMotionModel(0.0, 0.0)
    state = np.array([0.0, 0.0, 1.0, 0.0, 2.0 * np.pi])
    assert np.allclose(model.transition(state, 1.0), state, atol=1e-12)
    residual = model.residual(np.array([0, 0, 0, 3.1, 0.0]), np.array([0, 0, 0, -3.1, 0.0]))
    assert residual[3] == pytest.approx(6.2 - 2 * np.pi)


def test_function_motion_model_numerical_jacobian() -> None:
    def transition(states: FloatArray, dt: float, _control: FloatArray | None) -> FloatArray:
        return np.stack([states[..., 0] + dt * np.sin(states[..., 1]), states[..., 1]], axis=-1)

    model = FunctionMotionModel(transition=transition, process_noise=[0.1, 0.1], dimension=2)
    jacobian = model.jacobian(np.array([1.0, 0.4]), 0.5)
    assert np.allclose(jacobian, [[1.0, 0.5 * np.cos(0.4)], [0.0, 1.0]], atol=1e-7)


def test_function_motion_model_non_vectorized_and_noise_function() -> None:
    model = FunctionMotionModel(
        transition=lambda x, dt, u: x * 2.0,
        process_noise=lambda mean, dt: np.eye(1) * dt,
        dimension=1,
        is_vectorized=False,
    )
    assert np.allclose(model.transition(np.ones((3, 1)), 1.0), 2.0)
    assert np.allclose(model.process_noise(np.zeros(1), 0.25), [[0.25]])


def test_kinematic_accepts_numpy_scalar_intensity() -> None:
    model = KinematicMotionModel.constant_velocity(["x"], noise_intensity=np.float32(0.5))
    assert np.allclose(
        model.process_noise(np.zeros(2), 1.0), 0.5 * np.array([[1 / 3, 0.5], [0.5, 1.0]])
    )


def test_constant_turn_rate_velocity_add_inverts_residual() -> None:
    model = ConstantTurnRateVelocityMotionModel(1.0, 1.0)
    state = np.array([0.0, 0.0, 1.0, 3.0, 0.0])
    increment = np.array([0.1, 0.2, 0.3, 0.5, 0.1])
    moved = model.add(state, increment)
    assert -np.pi <= moved[3] < np.pi
    assert np.allclose(model.residual(moved, state), increment)


def test_numerical_jacobian_follows_state_arithmetic() -> None:
    # Multiplicative (log) error state: the identity transition has an identity Jacobian
    # with respect to that error, while plain perturbation would give 1 / x.
    model = FunctionMotionModel(
        transition=lambda x, dt, u: x,
        process_noise=[0.0],
        dimension=1,
        residual=lambda x, reference: np.log(x / reference),
        add=lambda x, increment: x * np.exp(increment),
    )
    assert model.jacobian(np.array([5.0]), 1.0) == pytest.approx(np.eye(1))


def test_function_motion_model_needs_residual_and_add_together() -> None:
    with pytest.raises(ValueError, match="together"):
        FunctionMotionModel(
            transition=lambda x, dt, u: x,
            process_noise=[0.1],
            dimension=1,
            add=lambda x, increment: x + increment,
        )


def test_function_motion_model_uses_the_given_mean() -> None:
    model = FunctionMotionModel(
        transition=lambda x, dt, u: x,
        process_noise=[0.1],
        dimension=1,
        mean=lambda states, weights: np.array([np.max(states)]),
    )
    assert model.mean(np.array([[1.0], [3.0]]), np.array([0.5, 0.5])).tolist() == [3.0]
    linear = FunctionMotionModel(transition=lambda x, dt, u: x, process_noise=[0.1], dimension=1)
    assert linear.mean(np.array([[1.0], [3.0]]), np.array([0.5, 0.5])).tolist() == [2.0]
