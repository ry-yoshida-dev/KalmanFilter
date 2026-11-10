"""Measurement model defined by user-supplied functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ..._arrays import as_covariance, as_returned, read_only
from ..._numerics import addition
from ...array_types import FloatArray
from ..measurement_model import MeasurementModel, StateAddition

type MeasurementFunction = Callable[[FloatArray], FloatArray]
type MeasurementNoiseFunction = Callable[[FloatArray], FloatArray]
type MeasurementResidualFunction = Callable[[FloatArray, FloatArray], FloatArray]
type MeasurementMeanFunction = Callable[[FloatArray, FloatArray], FloatArray]


class FunctionMeasurementModel(MeasurementModel):
    """
    Arbitrary (nonlinear) measurement model built from callables.

    Parameters
    ----------
    measure : MeasurementFunction
        ``h(states) -> measurements``.
    noise_covariance : ArrayLike | MeasurementNoiseFunction
        Constant ``R`` (matrix or diagonal), or ``R(mean) -> (m, m)``.
    dimension : int
        Number of measurement components ``m``.
    jacobian : MeasurementFunction | None
        Analytic ``dh/dx(mean) -> (m, n)``, or ``None`` to use central
        finite differences.
    residual : MeasurementResidualFunction | None
        Measurement difference, or ``None`` for plain subtraction.
    mean : MeasurementMeanFunction | None
        Weighted mean ``mean(measurements (k, m), weights (k,)) -> (m,)``, or
        ``None`` for the linear weighted mean. Pass it together with
        ``residual`` to average angular components on the circle in the
        unscented filter.
    is_vectorized : bool
        Whether ``measure`` accepts a batch ``(k, n)``. When False it is
        called once per row.

    Raises
    ------
    ValueError
        If ``dimension`` is not positive or ``noise_covariance`` has the wrong shape.

    Examples
    --------
    >>> model = FunctionMeasurementModel(
    ...     measure=lambda x: np.linalg.norm(x[..., :2], axis=-1, keepdims=True),
    ...     noise_covariance=[0.1],
    ...     dimension=1,
    ... )
    """

    def __init__(
        self,
        measure: MeasurementFunction,
        noise_covariance: ArrayLike | MeasurementNoiseFunction,
        dimension: int,
        jacobian: MeasurementFunction | None = None,
        residual: MeasurementResidualFunction | None = None,
        mean: MeasurementMeanFunction | None = None,
        is_vectorized: bool = True,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension: int = dimension
        self._measure: MeasurementFunction = measure
        if callable(noise_covariance):
            self._noise_covariance: MeasurementNoiseFunction = noise_covariance
        else:
            matrix = read_only(as_covariance(noise_covariance, "noise_covariance", dimension))
            self._noise_covariance = lambda mean: matrix
        self._jacobian: MeasurementFunction | None = jacobian
        self._residual: MeasurementResidualFunction | None = residual
        self._mean: MeasurementMeanFunction | None = mean
        self._is_vectorized: bool = is_vectorized

    @property
    @override
    def dimension(self) -> int:
        """Number of measurement components ``m``."""
        return self._dimension

    @override
    def measure(self, states: FloatArray) -> FloatArray:
        """
        Evaluate the user measurement function.

        Raises
        ------
        ValueError
            If the function does not return ``m`` components per state.
        """
        if self._is_vectorized or states.ndim == 1:
            shape = (*states.shape[:-1], self._dimension)
            return as_returned(self._measure(states), "measure", shape, states.ndim > 1)
        return np.stack(
            [as_returned(self._measure(row), "measure", (self._dimension,)) for row in states]
        )

    @override
    def jacobian(self, mean: FloatArray, add: StateAddition = addition) -> FloatArray:
        """
        Analytic Jacobian if supplied, otherwise finite differences along ``add``.

        Raises
        ------
        ValueError
            If the user Jacobian is not ``(m, n)``.
        """
        if self._jacobian is None:
            return super().jacobian(mean, add)
        return as_returned(self._jacobian(mean), "jacobian", (self._dimension, mean.shape[0]))

    @override
    def noise_covariance(self, mean: FloatArray) -> FloatArray:
        """
        Constant ``R`` or the user noise function evaluated at ``mean``.

        Raises
        ------
        ValueError
            If the user function does not return an ``(m, m)`` matrix.
        """
        shape = (self._dimension, self._dimension)
        return as_returned(self._noise_covariance(mean), "noise_covariance", shape)

    @override
    def residual(self, measurements: FloatArray, reference: FloatArray) -> FloatArray:
        """User residual if supplied, otherwise plain subtraction."""
        if self._residual is None:
            return super().residual(measurements, reference)
        return np.asarray(self._residual(measurements, reference), dtype=np.float64)

    @override
    def mean(self, measurements: FloatArray, weights: FloatArray) -> FloatArray:
        """User mean if supplied, otherwise the linear weighted mean."""
        if self._mean is None:
            return super().mean(measurements, weights)
        return np.asarray(self._mean(measurements, weights), dtype=np.float64)
