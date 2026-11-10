"""Constant turn rate and velocity (CTRV) motion model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np

from ..._numerics import circular_mean, wrap_angle
from ...array_types import FloatArray
from ..motion_model import MotionModel
from ..state_layout import StateLayout

_STRAIGHT_LINE_TURN_RATE: float = 1e-4
_YAW_INDEX: int = 3
_LAYOUT: StateLayout = StateLayout(["x", "y", "speed", "yaw", "yaw_rate"])


@dataclass(frozen=True)
class ConstantTurnRateVelocityMotionModel(MotionModel):
    """
    Nonlinear planar CTRV model with state ``[x, y, speed, yaw, yaw_rate]``.

    Speed and yaw rate are held constant over a step; the noise is a
    longitudinal acceleration and a yaw acceleration, both piecewise constant.
    Yaw is kept in ``[-pi, pi)`` and averaged on the circle, so the model is
    safe to use with the unscented filter.

    Attributes
    ----------
    acceleration_variance : float
        Variance of the longitudinal acceleration noise.
    yaw_acceleration_variance : float
        Variance of the yaw acceleration noise.

    Raises
    ------
    ValueError
        If a variance is negative.
    """

    acceleration_variance: float
    yaw_acceleration_variance: float

    def __post_init__(self) -> None:
        if self.acceleration_variance < 0.0 or self.yaw_acceleration_variance < 0.0:
            raise ValueError("noise variances must be non-negative")

    @property
    @override
    def layout(self) -> StateLayout:
        """Names of the state variables."""
        return _LAYOUT

    @override
    def transition(
        self, states: FloatArray, dt: float, control: FloatArray | None = None
    ) -> FloatArray:
        """
        Propagate along a circular arc (or a straight line for tiny yaw rates).

        Raises
        ------
        ValueError
            If ``control`` is given, since the model has no control input.
        """
        if control is not None:
            raise ValueError("ConstantTurnRateVelocityMotionModel does not accept a control input")
        x: FloatArray = states[..., 0]
        y: FloatArray = states[..., 1]
        speed: FloatArray = states[..., 2]
        yaw: FloatArray = states[..., _YAW_INDEX]
        yaw_rate: FloatArray = states[..., 4]
        is_straight = np.abs(yaw_rate) < _STRAIGHT_LINE_TURN_RATE
        safe_rate = np.where(is_straight, 1.0, yaw_rate)
        next_yaw = yaw + yaw_rate * dt
        arc_x = speed / safe_rate * (np.sin(next_yaw) - np.sin(yaw))
        arc_y = speed / safe_rate * (np.cos(yaw) - np.cos(next_yaw))
        # Second-order expansion keeps the two branches continuous at the threshold.
        line_x = speed * dt * np.cos(yaw) - 0.5 * speed * dt**2 * yaw_rate * np.sin(yaw)
        line_y = speed * dt * np.sin(yaw) + 0.5 * speed * dt**2 * yaw_rate * np.cos(yaw)
        propagated: FloatArray = np.stack(
            [
                x + np.where(is_straight, line_x, arc_x),
                y + np.where(is_straight, line_y, arc_y),
                speed,
                wrap_angle(next_yaw),
                yaw_rate,
            ],
            axis=-1,
        )
        return propagated

    @override
    def jacobian(
        self, mean: FloatArray, dt: float, control: FloatArray | None = None
    ) -> FloatArray:
        """Analytic Jacobian of :meth:`transition`."""
        _, _, speed, yaw, yaw_rate = (float(value) for value in mean)
        sin_yaw, cos_yaw = np.sin(yaw), np.cos(yaw)
        jacobian = np.eye(self.dimension)
        jacobian[_YAW_INDEX, 4] = dt
        if abs(yaw_rate) < _STRAIGHT_LINE_TURN_RATE:
            half_dt_squared = 0.5 * dt**2
            jacobian[0, 2] = dt * cos_yaw - half_dt_squared * yaw_rate * sin_yaw
            jacobian[0, 3] = -speed * dt * sin_yaw - half_dt_squared * speed * yaw_rate * cos_yaw
            jacobian[0, 4] = -half_dt_squared * speed * sin_yaw
            jacobian[1, 2] = dt * sin_yaw + half_dt_squared * yaw_rate * cos_yaw
            jacobian[1, 3] = speed * dt * cos_yaw - half_dt_squared * speed * yaw_rate * sin_yaw
            jacobian[1, 4] = half_dt_squared * speed * cos_yaw
            return jacobian
        next_yaw = yaw + yaw_rate * dt
        sin_next, cos_next = np.sin(next_yaw), np.cos(next_yaw)
        jacobian[0, 2] = (sin_next - sin_yaw) / yaw_rate
        jacobian[0, 3] = speed / yaw_rate * (cos_next - cos_yaw)
        jacobian[0, 4] = (
            speed * dt * cos_next / yaw_rate - speed * (sin_next - sin_yaw) / yaw_rate**2
        )
        jacobian[1, 2] = (cos_yaw - cos_next) / yaw_rate
        jacobian[1, 3] = speed / yaw_rate * (sin_next - sin_yaw)
        jacobian[1, 4] = (
            speed * dt * sin_next / yaw_rate - speed * (cos_yaw - cos_next) / yaw_rate**2
        )
        return jacobian

    @override
    def process_noise(self, mean: FloatArray, dt: float) -> FloatArray:
        """Covariance ``G diag(acceleration, yaw acceleration) G^T`` at the current yaw."""
        yaw = float(mean[_YAW_INDEX])
        half_dt_squared = 0.5 * dt**2
        gain = np.array(
            [
                [half_dt_squared * np.cos(yaw), 0.0],
                [half_dt_squared * np.sin(yaw), 0.0],
                [dt, 0.0],
                [0.0, half_dt_squared],
                [0.0, dt],
            ]
        )
        variances = np.array([self.acceleration_variance, self.yaw_acceleration_variance])
        covariance: FloatArray = (gain * variances) @ gain.T
        return covariance

    @override
    def residual(self, states: FloatArray, reference: FloatArray) -> FloatArray:
        """Difference with the yaw component wrapped into ``[-pi, pi)``."""
        difference = states - reference
        difference[..., _YAW_INDEX] = wrap_angle(difference[..., _YAW_INDEX])
        return difference

    @override
    def add(self, states: FloatArray, increments: FloatArray) -> FloatArray:
        """Sum with the yaw component wrapped into ``[-pi, pi)``."""
        result = states + increments
        result[..., _YAW_INDEX] = wrap_angle(result[..., _YAW_INDEX])
        return result

    @override
    def mean(self, states: FloatArray, weights: FloatArray) -> FloatArray:
        """Weighted mean with the yaw component averaged on the circle."""
        result: FloatArray = weights @ states
        result[_YAW_INDEX] = circular_mean(states[:, _YAW_INDEX], weights)
        return result
