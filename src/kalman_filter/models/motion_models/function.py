"""Motion model defined by user-supplied functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ..._arrays import as_covariance, as_returned, read_only
from ...array_types import FloatArray
from ..motion_model import MotionModel
from ..state_layout import StateLayout

type TransitionFunction = Callable[[FloatArray, float, FloatArray | None], FloatArray]
type TransitionJacobianFunction = Callable[[FloatArray, float, FloatArray | None], FloatArray]
type ProcessNoiseFunction = Callable[[FloatArray, float], FloatArray]
type StateResidualFunction = Callable[[FloatArray, FloatArray], FloatArray]
type StateAddFunction = Callable[[FloatArray, FloatArray], FloatArray]
type StateMeanFunction = Callable[[FloatArray, FloatArray], FloatArray]


class FunctionMotionModel(MotionModel):
    """
    Arbitrary (nonlinear) motion model built from callables.

    Parameters
    ----------
    transition : TransitionFunction
        ``f(states, dt, control) -> states``.
    process_noise : ArrayLike | ProcessNoiseFunction
        Constant ``Q`` (matrix or diagonal), or ``Q(mean, dt) -> (n, n)``.
    dimension : int
        Number of state variables ``n``.
    jacobian : TransitionJacobianFunction | None
        Analytic ``df/dx(mean, dt, control) -> (n, n)``, or ``None`` to use
        central finite differences.
    residual : StateResidualFunction | None
        State difference, or ``None`` for plain subtraction.
    add : StateAddFunction | None
        Inverse of ``residual`` (apply an increment to states), or ``None``
        for plain addition. Pass both to wrap angular components.
    mean : StateMeanFunction | None
        Weighted mean ``mean(states (k, n), weights (k,)) -> (n,)``, or
        ``None`` for the linear weighted mean. Pass it to average angular
        components on the circle in the unscented filter.
    names : Sequence[str] | None
        State variable names, or ``None`` to generate ``x0, x1, ...``.
    is_vectorized : bool
        Whether ``transition`` accepts a batch ``(k, n)``. When False it is
        called once per row, which is slower but needs no batch support.

    Raises
    ------
    ValueError
        If ``dimension`` or ``names`` are inconsistent, or only one of
        ``residual`` and ``add`` is given.

    Examples
    --------
    >>> model = FunctionMotionModel(
    ...     transition=lambda x, dt, u: x * np.exp(-dt),
    ...     process_noise=[0.1],
    ...     dimension=1,
    ... )
    """

    def __init__(
        self,
        transition: TransitionFunction,
        process_noise: ArrayLike | ProcessNoiseFunction,
        dimension: int,
        jacobian: TransitionJacobianFunction | None = None,
        residual: StateResidualFunction | None = None,
        add: StateAddFunction | None = None,
        mean: StateMeanFunction | None = None,
        names: Sequence[str] | None = None,
        is_vectorized: bool = True,
    ) -> None:
        if (residual is None) != (add is None):
            raise ValueError("residual and add must be given together")
        self._layout: StateLayout = (
            StateLayout.anonymous(dimension) if names is None else StateLayout(names)
        )
        if self._layout.dimension != dimension:
            raise ValueError(f"names must have length {dimension}, got {self._layout.dimension}")
        self._transition: TransitionFunction = transition
        if callable(process_noise):
            self._process_noise: ProcessNoiseFunction = process_noise
        else:
            matrix = read_only(as_covariance(process_noise, "process_noise", dimension))
            self._process_noise = lambda mean, dt: matrix
        self._jacobian: TransitionJacobianFunction | None = jacobian
        self._residual: StateResidualFunction | None = residual
        self._add: StateAddFunction | None = add
        self._mean: StateMeanFunction | None = mean
        self._is_vectorized: bool = is_vectorized

    @property
    @override
    def layout(self) -> StateLayout:
        """Names of the state variables."""
        return self._layout

    @override
    def transition(
        self, states: FloatArray, dt: float, control: FloatArray | None = None
    ) -> FloatArray:
        """
        Evaluate the user transition function.

        Raises
        ------
        ValueError
            If the function returns another shape than ``states``.
        """
        if self._is_vectorized or states.ndim == 1:
            return as_returned(
                self._transition(states, dt, control), "transition", states.shape, states.ndim > 1
            )
        row_shape = states.shape[1:]
        return np.stack(
            [
                as_returned(self._transition(row, dt, control), "transition", row_shape)
                for row in states
            ]
        )

    @override
    def jacobian(
        self, mean: FloatArray, dt: float, control: FloatArray | None = None
    ) -> FloatArray:
        """
        Analytic Jacobian if supplied, otherwise finite differences.

        Raises
        ------
        ValueError
            If the user Jacobian is not ``(n, n)``.
        """
        if self._jacobian is None:
            return super().jacobian(mean, dt, control)
        shape = (self.dimension, self.dimension)
        return as_returned(self._jacobian(mean, dt, control), "jacobian", shape)

    @override
    def process_noise(self, mean: FloatArray, dt: float) -> FloatArray:
        """
        Constant ``Q`` or the user noise function evaluated at ``(mean, dt)``.

        Raises
        ------
        ValueError
            If the user function does not return an ``(n, n)`` matrix.
        """
        shape = (self.dimension, self.dimension)
        return as_returned(self._process_noise(mean, dt), "process_noise", shape)

    @override
    def residual(self, states: FloatArray, reference: FloatArray) -> FloatArray:
        """User residual if supplied, otherwise plain subtraction."""
        if self._residual is None:
            return super().residual(states, reference)
        return np.asarray(self._residual(states, reference), dtype=np.float64)

    @override
    def add(self, states: FloatArray, increments: FloatArray) -> FloatArray:
        """User addition if supplied, otherwise plain addition."""
        if self._add is None:
            return super().add(states, increments)
        return np.asarray(self._add(states, increments), dtype=np.float64)

    @override
    def mean(self, states: FloatArray, weights: FloatArray) -> FloatArray:
        """User mean if supplied, otherwise the linear weighted mean."""
        if self._mean is None:
            return super().mean(states, weights)
        return np.asarray(self._mean(states, weights), dtype=np.float64)
