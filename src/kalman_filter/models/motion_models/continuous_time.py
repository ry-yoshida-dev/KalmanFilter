"""Motion model defined by a stochastic differential equation."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import override

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm

from ..._arrays import as_covariance, as_matrix, as_returned, read_only
from ..._numerics import addition, difference, numerical_jacobian, symmetrize
from ...array_types import FloatArray
from ..motion_model import MotionModel
from ..state_layout import StateLayout

type DriftFunction = Callable[[FloatArray, FloatArray | None], FloatArray]
type DriftJacobianFunction = Callable[[FloatArray], FloatArray]


class ContinuousTimeMotionModel(MotionModel):
    """
    Continuous-time model ``dx/dt = a(x, u) + L w(t)`` with white noise of density ``Q_c``.

    This is the dynamics of the continuous-discrete filter: physics is written
    as an ordinary differential equation, and any filter evaluates it at
    arbitrary, irregular measurement times. :meth:`transition` integrates the
    drift with the classic fourth-order Runge-Kutta method in sub-steps no
    longer than ``max_step``. :meth:`process_noise` discretizes the noise with
    Van Loan's method around the drift Jacobian at the current mean, which is
    exact for linear drift and a first-order approximation otherwise.

    Parameters
    ----------
    drift : DriftFunction
        ``a(states, control) -> dx/dt`` for states ``(n,)`` or ``(k, n)``.
    noise_spectral_density : ArrayLike
        Power spectral density ``Q_c`` of ``w(t)`` (matrix or diagonal), with
        shape ``(q, q)``; ``(n, n)`` when ``noise_gain`` is omitted.
    dimension : int
        Number of state variables ``n``.
    noise_gain : ArrayLike | None
        Matrix ``L`` with shape ``(n, q)`` that maps the noise into the state,
        or ``None`` for the identity.
    drift_jacobian : DriftJacobianFunction | None
        Analytic ``da/dx(mean) -> (n, n)`` without control, used for the
        noise discretization, or ``None`` for finite differences.
    max_step : float
        Longest Runge-Kutta sub-step; a prediction over ``dt`` uses
        ``ceil(|dt| / max_step)`` sub-steps.
    names : Sequence[str] | None
        State variable names, or ``None`` to generate ``x0, x1, ...``.
    is_vectorized : bool
        Whether ``drift`` accepts a batch ``(k, n)``. When False it is called
        once per row.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or ``max_step`` is not positive.

    Examples
    --------
    >>> # Van der Pol oscillator driven by noise on the velocity.
    >>> model = ContinuousTimeMotionModel(
    ...     drift=lambda x, u: np.stack(
    ...         [x[..., 1], (1.0 - x[..., 0] ** 2) * x[..., 1] - x[..., 0]], axis=-1
    ...     ),
    ...     noise_spectral_density=[[0.1]],
    ...     noise_gain=[[0.0], [1.0]],
    ...     dimension=2,
    ...     names=["position", "velocity"],
    ... )
    """

    def __init__(
        self,
        drift: DriftFunction,
        noise_spectral_density: ArrayLike,
        dimension: int,
        noise_gain: ArrayLike | None = None,
        drift_jacobian: DriftJacobianFunction | None = None,
        max_step: float = 0.01,
        names: Sequence[str] | None = None,
        is_vectorized: bool = True,
    ) -> None:
        if not max_step > 0.0:
            raise ValueError(f"max_step must be positive, got {max_step}")
        self._layout: StateLayout = (
            StateLayout.anonymous(dimension) if names is None else StateLayout(names)
        )
        if self._layout.dimension != dimension:
            raise ValueError(f"names must have length {dimension}, got {self._layout.dimension}")
        gain = np.eye(dimension) if noise_gain is None else as_matrix(noise_gain, "noise_gain")
        if gain.shape[0] != dimension:
            raise ValueError(f"noise_gain must have {dimension} rows, got {gain.shape[0]}")
        density = as_covariance(noise_spectral_density, "noise_spectral_density", gain.shape[1])
        self._diffusion: FloatArray = read_only(gain @ density @ gain.T)
        self._drift: DriftFunction = drift
        self._drift_jacobian: DriftJacobianFunction | None = drift_jacobian
        self._max_step: float = float(max_step)
        self._is_vectorized: bool = is_vectorized

    @property
    @override
    def layout(self) -> StateLayout:
        """Names of the state variables."""
        return self._layout

    @property
    def diffusion(self) -> FloatArray:
        """Noise intensity ``L Q_c L^T`` entering the state, with shape ``(n, n)`` (read-only)."""
        return self._diffusion

    def drift(self, states: FloatArray, control: FloatArray | None = None) -> FloatArray:
        """
        Time derivative ``a(x, u)`` for states ``(n,)`` or ``(k, n)``.

        Raises
        ------
        ValueError
            If the drift function returns another shape than ``states``.
        """
        if self._is_vectorized or states.ndim == 1:
            return as_returned(self._drift(states, control), "drift", states.shape, states.ndim > 1)
        return np.stack(
            [as_returned(self._drift(row, control), "drift", states.shape[1:]) for row in states]
        )

    def drift_jacobian(self, mean: FloatArray) -> FloatArray:
        """
        Jacobian ``da/dx`` at ``mean`` without control, with shape ``(n, n)``.

        Raises
        ------
        ValueError
            If the user Jacobian is not ``(n, n)``.
        """
        if self._drift_jacobian is not None:
            shape = (self.dimension, self.dimension)
            return as_returned(self._drift_jacobian(mean), "drift_jacobian", shape)
        return numerical_jacobian(
            function=self.drift, point=mean, add=addition, difference=difference
        )

    @override
    def transition(
        self, states: FloatArray, dt: float, control: FloatArray | None = None
    ) -> FloatArray:
        """Integrate the drift over ``dt`` with fourth-order Runge-Kutta sub-steps."""
        steps = math.ceil(abs(dt) / self._max_step)
        step = dt / max(steps, 1)
        current = np.array(states, dtype=np.float64)
        for _ in range(steps):
            first = self.drift(current, control)
            second = self.drift(current + 0.5 * step * first, control)
            third = self.drift(current + 0.5 * step * second, control)
            fourth = self.drift(current + step * third, control)
            current = current + step / 6.0 * (first + 2.0 * second + 2.0 * third + fourth)
        return current

    @override
    def process_noise(self, mean: FloatArray, dt: float) -> FloatArray:
        """Van Loan discretization ``Q = int_0^dt e^{A s} L Q_c L^T e^{A^T s} ds``."""
        dimension = self.dimension
        duration = abs(dt)
        drift_jacobian = self.drift_jacobian(mean)
        block = np.zeros((2 * dimension, 2 * dimension))
        block[:dimension, :dimension] = -drift_jacobian
        block[:dimension, dimension:] = self._diffusion
        block[dimension:, dimension:] = drift_jacobian.T
        exponential = expm(block * duration)
        transition = exponential[dimension:, dimension:].T
        covariance: FloatArray = symmetrize(transition @ exponential[:dimension, dimension:])
        return covariance
