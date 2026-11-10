"""Range and bearing measurement model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ..._arrays import as_covariance, as_vector, read_only
from ..._numerics import addition, circular_mean, wrap_angle
from ...array_types import FloatArray, IntArray
from ..measurement_model import MeasurementModel, StateAddition

_BEARING_INDEX: int = 1


class RangeBearingMeasurementModel(MeasurementModel):
    """
    Polar measurement ``[range, bearing]`` of a planar position, e.g. radar or lidar.

    Bearing is measured counter-clockwise from the x axis in radians and is
    wrapped into ``[-pi, pi)`` in residuals and means.

    Parameters
    ----------
    position_indices : Sequence[int] | IntArray
        Indices of the ``x`` and ``y`` position inside the state vector, e.g.
        ``layout.indices(["x", "y"])``.
    noise_covariance : ArrayLike
        Noise ``R`` of ``[range, bearing]`` with shape ``(2, 2)``, or its diagonal.
    sensor_position : ArrayLike
        Sensor position ``[x, y]`` in the state frame.

    Raises
    ------
    ValueError
        If ``position_indices`` does not hold two distinct non-negative indices.
    """

    def __init__(
        self,
        position_indices: Sequence[int] | IntArray,
        noise_covariance: ArrayLike,
        sensor_position: ArrayLike = (0.0, 0.0),
    ) -> None:
        indices = [int(index) for index in position_indices]
        if len(indices) != 2 or indices[0] == indices[1] or min(indices) < 0:
            raise ValueError(
                f"position_indices must hold two distinct non-negative indices, got {indices}"
            )
        self._position_indices: list[int] = indices
        self._noise_covariance: FloatArray = read_only(
            as_covariance(noise_covariance, "noise_covariance", 2)
        )
        self._sensor_position: FloatArray = as_vector(sensor_position, "sensor_position", 2)

    @property
    @override
    def dimension(self) -> int:
        """Number of measurement components (``2``)."""
        return 2

    @override
    def measure(self, states: FloatArray) -> FloatArray:
        """Return ``[range, bearing]`` from the sensor to each state's position."""
        offset = states[..., self._position_indices] - self._sensor_position
        polar: FloatArray = np.stack(
            [np.hypot(offset[..., 0], offset[..., 1]), np.arctan2(offset[..., 1], offset[..., 0])],
            axis=-1,
        )
        return polar

    @override
    def jacobian(self, mean: FloatArray, add: StateAddition = addition) -> FloatArray:
        """
        Analytic Jacobian of :meth:`measure`; assumes plain addition on the position components.

        Raises
        ------
        ValueError
            If the position coincides with the sensor, where bearing is undefined.
        """
        offset = mean[self._position_indices] - self._sensor_position
        squared_range = float(np.sum(offset**2))
        if squared_range == 0.0:
            raise ValueError("bearing Jacobian is undefined at the sensor position")
        jacobian = np.zeros((2, mean.shape[0]))
        jacobian[0, self._position_indices] = offset / np.sqrt(squared_range)
        jacobian[1, self._position_indices] = np.array([-offset[1], offset[0]]) / squared_range
        return jacobian

    @override
    def noise_covariance(self, mean: FloatArray) -> FloatArray:
        """Return the constant ``R``."""
        return self._noise_covariance

    @override
    def residual(self, measurements: FloatArray, reference: FloatArray) -> FloatArray:
        """Difference with the bearing component wrapped into ``[-pi, pi)``."""
        difference = measurements - reference
        difference[..., _BEARING_INDEX] = wrap_angle(difference[..., _BEARING_INDEX])
        return difference

    @override
    def mean(self, measurements: FloatArray, weights: FloatArray) -> FloatArray:
        """Weighted mean with the bearing averaged on the circle."""
        result: FloatArray = weights @ measurements
        result[_BEARING_INDEX] = circular_mean(measurements[:, _BEARING_INDEX], weights)
        return result
