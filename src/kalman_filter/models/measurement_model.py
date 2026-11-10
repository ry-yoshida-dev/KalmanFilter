"""Abstract measurement (observation) model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from .._numerics import addition, numerical_jacobian
from ..array_types import FloatArray

type StateAddition = Callable[[FloatArray, FloatArray], FloatArray]


class MeasurementModel(ABC):
    """
    Measurement model ``z = h(x) + v``, ``v ~ N(0, R)``.

    ``measure`` accepts a single state ``(n,)`` or a batch ``(k, n)`` and
    returns ``(m,)`` or ``(k, m)``.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of measurement components ``m``."""

    @abstractmethod
    def measure(self, states: FloatArray) -> FloatArray:
        """
        Map states to expected measurements without noise.

        Parameters
        ----------
        states : FloatArray
            States with shape ``(n,)`` or ``(k, n)``.

        Returns
        -------
        FloatArray
            Expected measurements with shape ``(m,)`` or ``(k, m)``.
        """

    def jacobian(self, mean: FloatArray, add: StateAddition = addition) -> FloatArray:
        """
        Jacobian ``dh/dx`` at ``mean`` with shape ``(m, n)``.

        The derivative is taken with respect to the increment ``d`` in
        ``add(mean, d)``; filters pass :meth:`MotionModel.add`, so the Jacobian
        matches the error state the covariance describes (e.g. a rotation
        increment rather than a change of rotation-vector coordinates). The
        default uses central finite differences; override it when an analytic
        Jacobian is available.

        Parameters
        ----------
        mean : FloatArray
            Linearization point with shape ``(n,)``.
        add : StateAddition
            State addition ``(states, increments) -> states``; plain addition by default.
        """
        return numerical_jacobian(
            function=self.measure, point=mean, add=add, difference=self.residual
        )

    @abstractmethod
    def noise_covariance(self, mean: FloatArray) -> FloatArray:
        """
        Measurement noise covariance ``R`` with shape ``(m, m)``.

        Parameters
        ----------
        mean : FloatArray
            Current state mean, for state-dependent noise.
        """

    def residual(self, measurements: FloatArray, reference: FloatArray) -> FloatArray:
        """
        Difference ``measurements - reference`` in measurement space.

        Override it to wrap angular components.
        """
        return measurements - reference

    def mean(self, measurements: FloatArray, weights: FloatArray) -> FloatArray:
        """
        Weighted mean of measurements with shape ``(k, m)``.

        Override it to average angular components on the circle.
        """
        result: FloatArray = weights @ measurements
        return result
