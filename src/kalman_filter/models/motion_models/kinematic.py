"""Kinematic motion model with per-variable derivative order."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import factorial
from typing import SupportsFloat, override

import numpy as np

from ...array_types import FloatArray, IntArray
from ..motion_model import MotionModel
from ..state_layout import StateLayout
from ..state_role import StateRole

MAX_DERIVATIVE_ORDER: int = len(StateRole) - 1


class KinematicMotionModel(MotionModel):
    """
    Piecewise-polynomial kinematics driven by continuous white noise.

    Each variable is modelled up to its own derivative order: ``0`` is a random
    walk, ``1`` constant velocity, ``2`` constant acceleration, ``3`` constant
    jerk. The highest derivative is driven by continuous white noise with the
    given spectral density, which yields the exact discretized ``F(dt)`` and
    ``Q(dt)`` for any (also irregular) time step.

    The state is ordered by derivative level: every variable's value first,
    then the velocities, then the accelerations. Derivatives are named
    ``<variable>_velocity``, ``<variable>_acceleration``, and
    ``<variable>_jerk``. For example, a bounding-box state
    ``[u, v, s, r, u_velocity, v_velocity, s_velocity]`` is obtained when ``r``
    has order 0.

    Parameters
    ----------
    derivative_orders : Mapping[str, int]
        Derivative order of each variable, in state order.
    noise_intensities : Mapping[str, SupportsFloat] | SupportsFloat
        Spectral density of the white noise driving each variable's highest
        derivative, or one value for all variables.

    Raises
    ------
    ValueError
        If an order is outside ``0..3`` or an intensity is negative.
    KeyError
        If an intensity is missing for a variable, or given for an unknown one.

    Examples
    --------
    >>> model = KinematicMotionModel.constant_velocity(["x", "y"], noise_intensity=0.5)
    >>> model.layout.names
    ('x', 'y', 'x_velocity', 'y_velocity')
    """

    def __init__(
        self,
        derivative_orders: Mapping[str, int],
        noise_intensities: Mapping[str, SupportsFloat] | SupportsFloat,
    ) -> None:
        if len(derivative_orders) == 0:
            raise ValueError("derivative_orders must not be empty")
        for variable, order in derivative_orders.items():
            if not 0 <= order <= MAX_DERIVATIVE_ORDER:
                raise ValueError(
                    f"derivative order of {variable!r} must be in 0..{MAX_DERIVATIVE_ORDER}, "
                    + f"got {order}"
                )
        if isinstance(noise_intensities, Mapping):
            unknown = set(noise_intensities) - set(derivative_orders)
            if unknown:
                raise KeyError(f"noise_intensities has unknown variables: {sorted(unknown)}")
        intensities = (
            {variable: float(noise_intensities[variable]) for variable in derivative_orders}
            if isinstance(noise_intensities, Mapping)
            else dict.fromkeys(derivative_orders, float(noise_intensities))
        )
        if any(value < 0.0 for value in intensities.values()):
            raise ValueError(f"noise_intensities must be non-negative: {intensities}")

        self._derivative_orders: dict[str, int] = dict(derivative_orders)
        names = [
            variable + StateRole.from_level(level).suffix
            for level in range(max(derivative_orders.values()) + 1)
            for variable, order in derivative_orders.items()
            if order >= level
        ]
        self._layout: StateLayout = StateLayout(names)
        self._transition_terms: _PolynomialTerms = self._transition_pattern()
        self._noise_terms: _PolynomialTerms = self._noise_pattern(intensities)

    @classmethod
    def from_state_roles(
        cls,
        variables: Mapping[StateRole, Sequence[str]],
        noise_intensities: Mapping[str, SupportsFloat] | SupportsFloat,
    ) -> KinematicMotionModel:
        """
        Model whose state has the given variables in each role.

        Each variable's derivative order is the highest role it appears in.

        Raises
        ------
        ValueError
            If a variable appears twice in a role, or in a role without
            appearing in every lower role.

        Examples
        --------
        >>> KinematicMotionModel.from_state_roles(
        ...     {StateRole.STATIC: ["u", "v", "s", "r"], StateRole.DYNAMIC: ["u", "v", "s"]},
        ...     noise_intensities=1.0,
        ... ).layout.names
        ('u', 'v', 's', 'r', 'u_velocity', 'v_velocity', 's_velocity')
        """
        for role, names in variables.items():
            if len(set(names)) != len(names):
                raise ValueError(f"variables of role {role.value} must be unique: {list(names)}")
        orders = dict.fromkeys(variables.get(StateRole.STATIC, ()), 0)
        for role in list(StateRole)[1:]:
            for variable in variables.get(role, ()):
                if orders.get(variable) != role.level - 1:
                    raise ValueError(f"{variable!r} has role {role.value} but not every lower role")
                orders[variable] = role.level
        return cls(orders, noise_intensities)

    @classmethod
    def constant_velocity(
        cls, variables: Sequence[str], noise_intensity: SupportsFloat
    ) -> KinematicMotionModel:
        """Constant-velocity (white-noise acceleration) model for ``variables``."""
        return cls(dict.fromkeys(variables, 1), noise_intensity)

    @classmethod
    def constant_acceleration(
        cls, variables: Sequence[str], noise_intensity: SupportsFloat
    ) -> KinematicMotionModel:
        """Constant-acceleration (white-noise jerk) model for ``variables``."""
        return cls(dict.fromkeys(variables, 2), noise_intensity)

    def _derivative_indices(self, variable: str) -> list[int]:
        """State indices of ``variable`` and its derivatives, by derivative level."""
        order = self._derivative_orders[variable]
        return [
            self._layout.role_index(StateRole.from_level(level), variable)
            for level in range(order + 1)
        ]

    def _transition_pattern(self) -> _PolynomialTerms:
        """Terms of ``F(dt)``: ``dt^(j - i) / (j - i)!`` from level ``j`` into level ``i <= j``."""
        terms: list[tuple[int, int, int, float]] = []
        for variable in self._derivative_orders:
            indices = self._derivative_indices(variable)
            for i, row in enumerate(indices):
                for j, column in enumerate(indices[i:], start=i):
                    terms.append((row, column, j - i, 1.0 / factorial(j - i)))
        return _as_terms(terms)

    def _noise_pattern(self, intensities: Mapping[str, float]) -> _PolynomialTerms:
        """
        Terms of ``Q(dt)`` for white noise of density ``q`` on the highest derivative ``o``.

        ``Q_ij = q dt^p / ((o - i)! (o - j)! p)`` with ``p = 2 o + 1 - i - j``.
        """
        terms: list[tuple[int, int, int, float]] = []
        for variable, order in self._derivative_orders.items():
            indices = self._derivative_indices(variable)
            for i, row in enumerate(indices):
                for j, column in enumerate(indices):
                    power = 2 * order + 1 - i - j
                    coefficient = intensities[variable] / (
                        factorial(order - i) * factorial(order - j) * power
                    )
                    terms.append((row, column, power, coefficient))
        return _as_terms(terms)

    def _evaluate(self, terms: _PolynomialTerms, dt: float) -> FloatArray:
        """Scatter ``coefficient * dt^power`` into an ``(n, n)`` matrix."""
        rows, columns, powers, coefficients = terms
        matrix = np.zeros((self.dimension, self.dimension))
        matrix[rows, columns] = coefficients * dt**powers
        return matrix

    @property
    @override
    def layout(self) -> StateLayout:
        """Names of the state variables."""
        return self._layout

    @property
    def derivative_orders(self) -> dict[str, int]:
        """Derivative order of each base variable."""
        return dict(self._derivative_orders)

    def transition_matrix(self, dt: float) -> FloatArray:
        """Discrete transition matrix ``F(dt)``."""
        return self._evaluate(self._transition_terms, dt)

    @override
    def transition(
        self, states: FloatArray, dt: float, control: FloatArray | None = None
    ) -> FloatArray:
        """
        Apply ``F(dt) x``.

        Raises
        ------
        ValueError
            If ``control`` is given, since the model has no control input.
        """
        if control is not None:
            raise ValueError("KinematicMotionModel does not accept a control input")
        propagated: FloatArray = states @ self.transition_matrix(dt).T
        return propagated

    @override
    def jacobian(
        self, mean: FloatArray, dt: float, control: FloatArray | None = None
    ) -> FloatArray:
        """Return ``F(dt)``."""
        return self.transition_matrix(dt)

    @override
    def process_noise(self, mean: FloatArray, dt: float) -> FloatArray:
        """Discretized white-noise covariance ``Q(dt)``; ``dt`` is taken by magnitude."""
        return self._evaluate(self._noise_terms, abs(dt))


type _PolynomialTerms = tuple[IntArray, IntArray, FloatArray, FloatArray]


def _as_terms(terms: list[tuple[int, int, int, float]]) -> _PolynomialTerms:
    """Split ``(row, column, power, coefficient)`` terms into index and value arrays."""
    rows, columns, powers, coefficients = zip(*terms, strict=True)
    return (
        np.array(rows, dtype=np.intp),
        np.array(columns, dtype=np.intp),
        np.array(powers, dtype=np.float64),
        np.array(coefficients, dtype=np.float64),
    )
