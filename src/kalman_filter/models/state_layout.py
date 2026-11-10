"""Named layout of a state or measurement vector."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import override

import numpy as np

from ..array_types import IntArray
from .state_role import StateRole


class StateLayout:
    """
    Ordered variable names of a vector, with name-to-index lookup.

    Parameters
    ----------
    names : Sequence[str]
        Unique variable names in vector order.

    Raises
    ------
    ValueError
        If ``names`` is empty or contains duplicates.

    Examples
    --------
    >>> layout = StateLayout(["x", "y", "x_velocity", "y_velocity"])
    >>> layout.indices(["x", "y"])
    array([0, 1])
    """

    __slots__: tuple[str, ...] = ("_index_by_name", "_names")

    def __init__(self, names: Sequence[str]) -> None:
        if len(names) == 0:
            raise ValueError("names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError(f"names must be unique: {list(names)}")
        self._names: tuple[str, ...] = tuple(names)
        self._index_by_name: dict[str, int] = {name: i for i, name in enumerate(self._names)}

    @classmethod
    def anonymous(cls, dimension: int) -> StateLayout:
        """
        Layout with generated names ``x0, x1, ...``.

        Raises
        ------
        ValueError
            If ``dimension`` is not positive.
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        return cls([f"x{index}" for index in range(dimension)])

    @property
    def names(self) -> tuple[str, ...]:
        """Variable names in vector order."""
        return self._names

    @property
    def dimension(self) -> int:
        """Number of variables."""
        return len(self._names)

    def index(self, name: str) -> int:
        """
        Index of a variable.

        Raises
        ------
        KeyError
            If ``name`` is not part of the layout.
        """
        if name not in self._index_by_name:
            raise KeyError(f"unknown variable {name!r}; available: {list(self._names)}")
        return self._index_by_name[name]

    def indices(self, names: Sequence[str]) -> IntArray:
        """
        Indices of several variables, in the given order.

        Raises
        ------
        KeyError
            If any name is not part of the layout.
        """
        return np.array([self.index(name) for name in names], dtype=np.intp)

    def role_index(self, role: StateRole, variable: str) -> int:
        """
        Index of ``variable`` in ``role``, i.e. of the name ``variable + role.suffix``.

        Raises
        ------
        KeyError
            If the variable has no entry in that role.
        """
        return self.index(variable + role.suffix)

    def role_variables(self, role: StateRole) -> tuple[str, ...]:
        """
        Variables that have an entry in ``role``, in state order.

        For ``STATIC`` these are the names without a derivative suffix; for the
        other roles, the names ending with the role's suffix, with it removed.
        """
        if role is StateRole.STATIC:
            suffixes = [other.suffix for other in StateRole if other is not StateRole.STATIC]
            return tuple(
                name for name in self._names if not any(name.endswith(s) for s in suffixes)
            )
        return tuple(
            name.removesuffix(role.suffix) for name in self._names if name.endswith(role.suffix)
        )

    def role_indices(self, role: StateRole) -> IntArray:
        """Indices of all variables in ``role``, in state order."""
        return np.array(
            [self.role_index(role, variable) for variable in self.role_variables(role)],
            dtype=np.intp,
        )

    def __contains__(self, name: object) -> bool:
        return name in self._index_by_name

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, StateLayout) and self._names == other._names

    @override
    def __hash__(self) -> int:
        return hash(self._names)

    @override
    def __repr__(self) -> str:
        return f"StateLayout({list(self._names)})"
