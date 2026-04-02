from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Iterator

from ..utils import StateRole

@dataclass
class KFIndexMap:
    """
    Container class for index map of state vector.
    This class is used to map the state role and variable name to the index of the state vector.

    Attributes:
    ----------
    static: dict[str, int]
        Index map for static variables.
        Key is the variable name, value is the index.
    dynamic: dict[str, int]
        Index map for dynamic variables.
        Key is the variable name, value is the index.
    acceleration: dict[str, int]
        Index map for acceleration variables.
        Key is the variable name, value is the index.
    """
    static: dict[str, int] = field(default_factory=dict)
    dynamic: dict[str, int] = field(default_factory=dict)
    acceleration: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        static_values = self.static.values()
        dynamic_values = self.dynamic.values()
        acceleration_values = self.acceleration.values()
        if len(static_values) < len(dynamic_values):
            raise ValueError(f"Static variables must be greater than or equal to dynamic variables. static: {len(static_values)}, dynamic: {len(dynamic_values)}")
        if len(dynamic_values) < len(acceleration_values):
            raise ValueError(f"Dynamic variables must be greater than or equal to acceleration variables. dynamic: {len(dynamic_values)}, acceleration: {len(acceleration_values)}")

    @property
    def iterate(self) -> Iterator[tuple[StateRole, str, int]]:
        """
        Iterate over the index map.
        * Iterating over this will yield (StateRole, variable name, index) in the following order:
        static -> dynamic -> acceleration

        Returns:
        ----------
        Iterator[tuple[StateRole, str, int]]:
            Iterator over the index map.
            Tuple of (StateRole, variable name, index).
        """
        for state_role, values in self:
            for variable_name, value in values.items():
                yield state_role, variable_name, value

    def __iter__(self) -> Iterator[tuple[StateRole, dict[str, int]]]:
        """
        Iterate over the index map.
        * Iterating over this will yield (StateRole, dict[str, int]) in the following order:
        static -> dynamic -> acceleration

        Returns:
        ----------
        Iterator[tuple[StateRole, dict[str, int]]]:
            Iterator over the index map.
            Tuple of (StateRole, dict[str, int]).
        """
        yield (StateRole.STATIC, self.static)
        yield (StateRole.DYNAMIC, self.dynamic)
        yield (StateRole.ACCELERATION, self.acceleration)

    def __setitem__(
        self, 
        key: StateRole, 
        value: dict[str, int]
        ) -> None:
        """
        Set the index map for the given state role.

        Parameters:
        ----------
        key: StateRole
            State role to set the index map for.
        value: dict[str, int]
            Index map for the given state role.
        """
        match key:
            case StateRole.STATIC:
                self.static = value
            case StateRole.DYNAMIC:
                self.dynamic = value
            case StateRole.ACCELERATION:
                self.acceleration = value

    def __getitem__(
        self, 
        key: StateRole
        ) -> dict[str, int]:
        """
        Get the index map for the given state role.

        Parameters:
        ----------
        key: StateRole
            State role to get the index map for.
        
        Returns:
        ----------
        dict[str, int]
            Index map for the given state role.
            Key is the variable name, value is the index.
        """
        match key:
            case StateRole.STATIC:
                return self.static
            case StateRole.DYNAMIC:
                return self.dynamic
            case StateRole.ACCELERATION:
                return self.acceleration

    def __len__(self) -> int:
        """
        Return the length of the index map.

        Returns:
        ----------
        int
            Number of variables in the index map.
            Sum of the lengths of the static, dynamic, and acceleration index maps.
        """
        return len(self.static) + len(self.dynamic) + len(self.acceleration)
