"""Derivative level of a state variable."""

from __future__ import annotations

from enum import Enum


class StateRole(Enum):
    """
    Role of a state variable: the derivative level of the quantity it describes.

    A variable ``u`` of role ``STATIC`` is named ``u`` in a :class:`StateLayout`;
    its derivatives are named with the role's suffix (``u_velocity``,
    ``u_acceleration``, ``u_jerk``), as :class:`KinematicMotionModel` lays them out.

    Attributes
    ----------
    STATIC
        0th derivative (e.g. position).
    DYNAMIC
        1st derivative (e.g. velocity).
    ACCELERATION
        2nd derivative.
    JERK
        3rd derivative.
    """

    STATIC = "static"
    DYNAMIC = "dynamic"
    ACCELERATION = "acceleration"
    JERK = "jerk"

    @property
    def suffix(self) -> str:
        """Suffix appended to the variable name in a layout (empty for ``STATIC``)."""
        return _SUFFIXES[self]

    @property
    def level(self) -> int:
        """Derivative order of the role."""
        return list(StateRole).index(self)

    @classmethod
    def from_level(cls, level: int) -> StateRole:
        """
        Role of the given derivative order.

        Raises
        ------
        ValueError
            If ``level`` is outside ``0..3``.
        """
        roles = list(cls)
        if not 0 <= level < len(roles):
            raise ValueError(f"derivative level must be in 0..{len(roles) - 1}, got {level}")
        return roles[level]


_SUFFIXES: dict[StateRole, str] = {
    StateRole.STATIC: "",
    StateRole.DYNAMIC: "_velocity",
    StateRole.ACCELERATION: "_acceleration",
    StateRole.JERK: "_jerk",
}
