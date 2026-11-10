"""Abstract motion (process) model."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .._numerics import numerical_jacobian
from ..array_types import FloatArray
from .state_layout import StateLayout


class MotionModel(ABC):
    """
    Discrete-time motion model ``x_k = f(x_{k-1}, dt, u) + w``, ``w ~ N(0, Q)``.

    Every method that takes ``states`` accepts either a single state with
    shape ``(n,)`` or a batch with shape ``(k, n)`` and returns an array with
    the same leading shape. Filters rely on this to evaluate sigma points and
    finite differences in one vectorized call.
    """

    @property
    @abstractmethod
    def layout(self) -> StateLayout:
        """Names of the state variables."""

    @property
    def dimension(self) -> int:
        """Number of state variables ``n``."""
        return self.layout.dimension

    @abstractmethod
    def transition(
        self, states: FloatArray, dt: float, control: FloatArray | None = None
    ) -> FloatArray:
        """
        Propagate states through ``f`` without noise.

        Parameters
        ----------
        states : FloatArray
            States with shape ``(n,)`` or ``(k, n)``.
        dt : float
            Time step.
        control : FloatArray | None
            Control input ``u``, or ``None`` when there is none.

        Returns
        -------
        FloatArray
            Propagated states with the same shape as ``states``.
        """

    def jacobian(
        self, mean: FloatArray, dt: float, control: FloatArray | None = None
    ) -> FloatArray:
        """
        Jacobian ``df/dx`` at ``mean`` with shape ``(n, n)``.

        The default uses central finite differences; override it when an
        analytic Jacobian is available.
        """
        return numerical_jacobian(
            function=lambda points: self.transition(points, dt, control),
            point=mean,
            add=self.add,
            difference=self.residual,
        )

    @abstractmethod
    def process_noise(self, mean: FloatArray, dt: float) -> FloatArray:
        """
        Process noise covariance ``Q`` for a step of length ``dt``.

        Parameters
        ----------
        mean : FloatArray
            Current state mean with shape ``(n,)``, for state-dependent noise.
        dt : float
            Time step.

        Returns
        -------
        FloatArray
            Covariance with shape ``(n, n)``.
        """

    def residual(self, states: FloatArray, reference: FloatArray) -> FloatArray:
        """
        Difference ``states - reference`` in state space.

        Override it, together with :meth:`add`, to wrap angular components.
        """
        return states - reference

    def add(self, states: FloatArray, increments: FloatArray) -> FloatArray:
        """
        Apply increments to states; the inverse of :meth:`residual`.

        Filters place sigma points and apply corrections with it, so
        ``residual(add(x, d), x) == d`` must hold. Override it to wrap angular components
        back into their range.
        """
        return states + increments

    def mean(self, states: FloatArray, weights: FloatArray) -> FloatArray:
        """
        Weighted mean of states with shape ``(k, n)``.

        Override it to average angular components on the circle.
        """
        result: FloatArray = weights @ states
        return result
