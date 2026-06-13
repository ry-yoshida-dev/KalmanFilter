from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state_vector import StateVector, KFIndexMap
from ..array_types import FloatArray
from .base import KFParameter
from .utils import StateRole

@dataclass
class StateTransitionMatrix(KFParameter):
    """
    State transition matrix.
    Represented as F in mathematical notation.

    Attributes:
    ----------
    value: FloatArray
        Value of the state transition matrix (inherited from base).
    """

    @classmethod
    def create_matrix(
        cls,
        x: StateVector, 
        dt: float=1.0
        ) -> StateTransitionMatrix:
        """
        Create the state transition matrix based on the state vector format.

        Parameters:
        ----------
        x: StateVector
            The state vector.
        dt: float
            The time step between the current state and the next state.

        Returns:
        ----------
        StateTransitionMatrix
            The state transition matrix.
        """
        dim = x.dim
        F_value = np.zeros((dim, dim))
        index_map = x.index_map

        for state_role, variable, _ in index_map.iterate:
            cls._set_transition_coefficient(
                F_value=F_value, 
                index_map=index_map, 
                state_role=state_role, 
                variable=variable, 
                dt=dt
                )
        return cls(value=F_value)

    @staticmethod
    def _set_transition_coefficient(
        F_value: FloatArray, 
        index_map: KFIndexMap, 
        state_role: StateRole, 
        variable: str, 
        dt: float
        ) -> None:
        """
        Set the transition coefficient for the given state role and variable.

        Parameters:
        ----------
        F_value: FloatArray
            The state transition matrix.
        index_map: KFIndexMap
            The index map.
            example of index_map structure:
            {
                "static": {"u": 0, "v": 1, "s": 2, "r": 3},
                "dynamic": {"u": 4, "v": 5, "s": 6},
                "acceleration": {"u": 7, "v": 8, "s": 9}
            }
        state_role: StateRole
            The state role.
        variable: str
            The variable of the state role.
            example of variable: "u"
        dt: float
            The time step between the current state and the next state.
        """
        current_index = index_map[state_role][variable]
        F_value[current_index, current_index] = 1
        match state_role:
            case StateRole.STATIC:
                pass
            case StateRole.DYNAMIC:
                static_index = index_map[StateRole.STATIC][variable]
                F_value[static_index, current_index] = dt
            case StateRole.ACCELERATION:
                static_index = index_map[StateRole.STATIC][variable]
                dynamic_index = index_map[StateRole.DYNAMIC][variable]
                F_value[static_index, current_index] = 0.5 * dt * dt
                F_value[dynamic_index, current_index] = dt 

