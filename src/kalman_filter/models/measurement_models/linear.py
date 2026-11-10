"""Linear measurement model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ..._arrays import as_covariance, as_matrix, read_only
from ..._numerics import addition
from ...array_types import FloatArray
from ..measurement_model import MeasurementModel, StateAddition
from ..state_layout import StateLayout


class LinearMeasurementModel(MeasurementModel):
    """
    Linear measurement model ``z = H x + v``.

    The stored matrices are read-only.

    Parameters
    ----------
    measurement_matrix : ArrayLike
        Measurement matrix ``H`` with shape ``(m, n)``.
    noise_covariance : ArrayLike
        Measurement noise ``R`` with shape ``(m, m)``, or its diagonal.

    Raises
    ------
    ValueError
        If the shapes are inconsistent.

    Examples
    --------
    Observe ``x`` and ``y`` of a constant-velocity state by name:

    >>> layout = StateLayout(["x", "y", "x_velocity", "y_velocity"])
    >>> model = LinearMeasurementModel.from_observed_states(layout, ["x", "y"], [0.5, 0.5])
    """

    def __init__(self, measurement_matrix: ArrayLike, noise_covariance: ArrayLike) -> None:
        matrix = as_matrix(measurement_matrix, "measurement_matrix")
        self._measurement_matrix: FloatArray = read_only(matrix)
        self._noise_covariance: FloatArray = read_only(
            as_covariance(noise_covariance, "noise_covariance", matrix.shape[0])
        )

    @classmethod
    def from_observed_states(
        cls, layout: StateLayout, observed: Sequence[str], noise_covariance: ArrayLike
    ) -> LinearMeasurementModel:
        """
        Model that directly observes the named state variables.

        Parameters
        ----------
        layout : StateLayout
            Layout of the state vector.
        observed : Sequence[str]
            Observed variable names, in measurement order.
        noise_covariance : ArrayLike
            Measurement noise ``R``, or its diagonal.

        Raises
        ------
        KeyError
            If a name is not part of ``layout``.
        """
        matrix = np.zeros((len(observed), layout.dimension))
        matrix[np.arange(len(observed)), layout.indices(observed)] = 1.0
        return cls(matrix, noise_covariance)

    @property
    @override
    def dimension(self) -> int:
        """Number of measurement components ``m``."""
        return int(self._measurement_matrix.shape[0])

    @property
    def measurement_matrix(self) -> FloatArray:
        """Measurement matrix ``H``."""
        return self._measurement_matrix

    @override
    def measure(self, states: FloatArray) -> FloatArray:
        """Apply ``H x``."""
        measured: FloatArray = states @ self._measurement_matrix.T
        return measured

    @override
    def jacobian(self, mean: FloatArray, add: StateAddition = addition) -> FloatArray:
        """Return ``H``; exact for plain state addition, so ``add`` is not used."""
        return self._measurement_matrix

    @override
    def noise_covariance(self, mean: FloatArray) -> FloatArray:
        """Return the constant ``R``."""
        return self._noise_covariance
