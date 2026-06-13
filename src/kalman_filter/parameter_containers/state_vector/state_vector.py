from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..kalman_gain import KalmanGain
    from ..state_transition_matrix import StateTransitionMatrix
    from ..measurement_vector import MeasurementVector
    from ..control_input_matrix import ControlInputMatrix
    from ..control_input_vector import ControlInputVector
    from ..measurement_matrix import MeasurementMatrix
from ...array_types import FloatArray
from .index_map import KFIndexMap
from ..base import KFParameter
from ..utils import StateRole

@dataclass
class StateVector(KFParameter):
    """
    Container class for state vector.
    * State vector is usually presented as x in mathematical notation.

    Attributes:
    ----------
    value: FloatArray
        Value of the state vector with shape (dim, 1) (inherited from base).
    format: dict[StateRole, list[str]]
        Format of the state vector.
        Key is the state role, value is the list of variable names.
    index_map: KFIndexMap
        Index map of the state vector. 
        Index map converts the state role and variable name to the index of the state vector.
    """
    format: dict[StateRole, list[str]]
    index_map: KFIndexMap

    def __post_init__(self):
        if len(self) != len(self.index_map):
            raise ValueError(f"Length of state vector and index map must be the same. state_vector: {len(self)}, index_map: {len(self.index_map)}")

    def predict(
        self, 
        F: StateTransitionMatrix, 
        B: ControlInputMatrix | None = None, 
        u: ControlInputVector | None = None
        ) -> StateVector:
        """ 
        Predict the next state vector. 

        Parameters
        ----------
        F: StateTransitionMatrix
            State transition matrix.
        B: ControlInputMatrix | None
            Control input matrix.
        u: ControlInputVector | None
            Control input vector.

        Returns
        -------
        StateVector
            Predicted state vector.
        """
        if B is not None and u is not None:
            x = F.value @ self.value + B.value @ u.value
        else:
            x = F.value @ self.value
        return StateVector(
            value=x, 
            format=self.format, 
            index_map=self.index_map
            )

    def update(
        self, 
        K: KalmanGain, 
        innovation: FloatArray
        ) -> None:
        """ 
        Update the state vector from the current observation.

        Parameters
        ----------
        K: KalmanGain
            Kalman gain matrix.
            KalmanGain K and innovation are computed from the current observation.
        innovation: FloatArray
            Innovation vector that is difference of measurement vector and the predicted measurement vector.
            innovation is computed in MeasurementVector class.
        """
        self.value += K.value @ innovation
    
    @classmethod
    def register_zero_vec_from_format(
        cls,
        format: dict[str, list[str]]
        ) -> StateVector:
        """ 
        Register the zero state vector from the state vector format.

        Parameters
        ----------
        format: Dict[str, list[str]]
            State vector format.
            example of format:
            {
                "static": ["u", "v", "s", "r"], 
                "dynamic": ["u", "v", "s"], 
                "acceleration": []
            }

        Returns
        -------
        StateVector
            Zero state vector.
        """
        flat_keys = [key for group in format.values() for key in group]
        vec = np.zeros([len(flat_keys), 1])
        index_map = KFIndexMap()
        index_counter = 0
        for str_state_role, variables in format.items():
            state_role = StateRole(str_state_role)
            if state_role == StateRole.STATIC and len(variables) != len(set(variables)):
                raise ValueError(f"Each variable must be unique. state_role: {state_role}, variables: {variables}")
            index_map[state_role] = {var: idx for idx, var in enumerate(variables, start=index_counter)}
            index_counter += len(variables)
        return cls(
            value=vec, 
            format={StateRole(str_state_role): variables for str_state_role, variables in format.items()}, 
            index_map=index_map
            )

    def replace_by_measurement(
        self, 
        z: MeasurementVector, 
        H: MeasurementMatrix
        ) -> None:
        """ 
        Replace the state vector by the measurement.

        Parameters
        ----------
        z: MeasurementVector
            Measurement vector.
        """
        if H.is_square_matrix:
            self.value = np.linalg.solve(H.value, z.value)
        else:
            self.value = H.pseudo_inverse_matrix @ z.value

    def __getitem__(
        self, 
        key: StateRole | tuple[StateRole, str]
        ) -> float | FloatArray:
        """
        Access value with key of format or index like:
        - state_vector[StateRole.STATIC, "u"]
        - state_vector[StateRole.STATIC]
        - state_vector[0]

        Parameters
        ----------
        key: StateRole | tuple[StateRole, str]
            Key to access.

        Returns
        -------
        float | FloatArray
            Value of the key.
            If key is a StateRole, return a scalar of that group.
            If key is a tuple, return a FloatArray with shape (len(group), 1).
        """
        self._validate_key(key)
        if isinstance(key, tuple) and len(key) == 2:
            state_role, variable = key
            idx = self.index_map[state_role][variable]
            return self.value[idx, 0]
        else:
            state_role = key
            indices = [self.index_map[key][var] for var in self.format[state_role]]
            return self.value[indices, 0]
    
    def __setitem__(
        self, 
        key: StateRole | tuple[StateRole, str], 
        new_value: float | FloatArray
        ) -> None:
        """
        Set value with key of format or index like:
        - state_vector[StateRole.STATIC, "u"] = 10
        - state_vector[StateRole.STATIC] = [10, 20, 30, 40]
        
        Parameters
        ----------
        key: str|tuple[StateRole, str]
            Key to set.
        new_value: float | FloatArray
            Value to set.
        """
        self._validate_key(key)
        if isinstance(key, tuple) and len(key) == 2:
            state_role, variable = key
            idx = self.index_map[state_role][variable]
            self.value[idx, 0] = new_value
        else:
            state_role = key
            indices = [self.index_map[state_role][var] for var in self.format[state_role]]
            if not isinstance(new_value, np.ndarray) and not isinstance(new_value, float):
                raise ValueError(f"Invalid new value: {new_value}")
            self.value[indices, 0] = new_value

    def _validate_key(
        self, 
        key: StateRole | tuple[StateRole, str]
        ) -> None:
        """
        Validate the key.

        Parameters
        ----------
        key: StateRole|tuple[StateRole, str]
            Key to validate.
        """
        if isinstance(key, tuple) and len(key) == 2:
            state_role, variable = key
            if variable not in self.index_map[state_role]:
                raise ValueError(f"Invalid variable: {variable}")

    def __len__(self):
        """
        Return the dimension of the state vector.

        Returns
        -------
        int
            Dimension of the state vector.
        """
        return len(self.value)

    @property
    def dim(self) -> int:
        """
        Return the dimension of the state vector.

        Returns
        -------
        int
            Dimension of the state vector.
        """
        return len(self)



