"""State vector ``x``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy.typing import ArrayLike

from ..array_types import FloatArray, IntArray
from ..models import StateLayout, StateRole
from .parameter import KalmanFilterParameter

type StateKey = int | slice | str | StateRole | tuple[StateRole, str]


@dataclass(eq=False, repr=False)
class StateVector(KalmanFilterParameter):
    """
    State estimate ``x`` with shape ``(n,)``, accessible by role, variable name, or index.

    Writing an element changes the estimate of the filter that returned it.

    Attributes
    ----------
    value : FloatArray
        State with shape ``(n,)``.
    layout : StateLayout
        Variable names of the state.

    Raises
    ------
    ValueError
        If ``value`` is not 1-D with one entry per variable of ``layout``.

    Examples
    --------
    >>> kf.x[StateRole.STATIC, "u"]  # one variable
    >>> kf.x[StateRole.DYNAMIC]  # every velocity, in state order
    >>> kf.x["u_velocity"]  # by layout name
    >>> kf.x[0], kf.x[:4]  # by position
    >>> kf.x[StateRole.DYNAMIC, "s"] = 0.0
    >>> kf.x[StateRole.DYNAMIC] = np.zeros(3)
    """

    layout: StateLayout

    def __post_init__(self) -> None:
        if self.value.shape != (self.layout.dimension,):
            raise ValueError(
                f"state vector must have shape ({self.layout.dimension},), got {self.value.shape}"
            )

    @property
    def dimension(self) -> int:
        """Dimension ``n`` of the state."""
        return self.layout.dimension

    @overload
    def __getitem__(self, key: int | str | tuple[StateRole, str]) -> float: ...

    @overload
    def __getitem__(self, key: slice | StateRole) -> FloatArray: ...

    def __getitem__(self, key: StateKey) -> float | FloatArray:
        """
        Value of one variable, or a copy of the values of a role or slice.

        Raises
        ------
        KeyError
            If the variable is not part of the layout.
        IndexError
            If an integer index is out of range.
        """
        if isinstance(key, StateRole | slice):
            # A copy, so that only __setitem__ (which validates) writes to the estimate.
            group: FloatArray = self.value[self._indices(key)].copy()
            return group
        return float(self.value[self._indices(key)])

    def __setitem__(self, key: StateKey, value: float | ArrayLike) -> None:
        """
        Overwrite one variable, or every variable of a role or slice.

        Raises
        ------
        KeyError
            If the variable is not part of the layout.
        ValueError
            If the new values are not finite or do not match the role's size.
        """
        indices = self._indices(key)
        new_values = np.asarray(value, dtype=np.float64)
        expected_shape = np.shape(self.value[indices])
        if new_values.shape not in {expected_shape, ()}:
            raise ValueError(f"{key!r} needs shape {expected_shape}, got {new_values.shape}")
        if not np.all(np.isfinite(new_values)):
            raise ValueError(f"{key!r} must be finite, got {value!r}")
        self.value[indices] = new_values

    def _indices(self, key: StateKey) -> int | slice | IntArray:
        """Position(s) in :attr:`value` that ``key`` refers to."""
        if isinstance(key, StateRole):
            return self.layout.role_indices(key)
        if isinstance(key, tuple):
            role, variable = key
            return self.layout.role_index(role, variable)
        if isinstance(key, str):
            return self.layout.index(key)
        return key
