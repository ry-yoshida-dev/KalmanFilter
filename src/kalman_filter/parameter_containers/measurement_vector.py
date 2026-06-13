from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .state_vector import StateVector
    from .measurement_matrix import MeasurementMatrix
from ..array_types import FloatArray
from .base import KFParameter

@dataclass
class MeasurementVector(KFParameter):
    """
    Container class for measurement vector.
    * Measurement vector is usually presented as z in mathematical notation.

    Attributes:
    ----------
    value: FloatArray
        Value of the measurement vector (inherited from base).
        Shape: (len(format), 1)
    format: list[str]
        Format of the measurement vector.
        example of format: ["u", "v", "s", "r"]
    _index_map: dict[str, int]
        Index map of the measurement vector. 
        Key is the variable name, value is the index.
    dim: int
        Dimension of the measurement vector. 
    """
    format: list[str]
    _index_map: dict[str, int]
    dim: int

    @classmethod
    def register_zero_vec_from_format(
        cls,
        format: list[str]
        ) -> MeasurementVector:
        """ 
        Register the zero measurement vector by the given format.

        Parameters
        ----------
        format: list[str]
            Measurement vector format.

        Returns
        -------
        MeasurementVector
            Zero measurement vector.
        """
        flat_keys = list(format)
        dim = len(flat_keys)
        vec = np.zeros([len(flat_keys), 1])
        index_map = {var: idx for idx, var in enumerate(flat_keys)}
        return cls(
            value=vec, 
            format=format, 
            _index_map=index_map, 
            dim=dim
            )
    
    def compute_innovation(
        self, 
        x: StateVector, 
        H: MeasurementMatrix
        ) -> FloatArray:
        """ Compute the innovation by the current state vector and measurement matrix.

        Parameters
        ----------
        x: StateVector
            State vector.
        H: MeasurementMatrix
            Measurement matrix.

        Returns
        -------
        FloatArray
            Innovation vector.
        """
        return self.value - H.state2measurement(x)
    
    def update(
        self, 
        H: MeasurementMatrix, 
        x: StateVector
        ) -> None:
        """ 
        Update the measurement vector by the current state vector and measurement matrix.

        Parameters
        ----------
        H: MeasurementMatrix
            Measurement matrix.
        x: StateVector
            State vector.
        """
        self.value = H.state2measurement(x)

    def __setitem__(
        self, 
        key: str | int | list[int], 
        value: float | FloatArray
        ) -> None:
        """
        Set value with key of format or index like:
        - z["confidence"] = 0.9
        - z[0] = 0.9

        Parameters
        ----------
        key: str|int
            Key to set.
        value: float
            Value to set.
        """
        if isinstance(key, int | list):
            indices = key
        else:
            index = self._index_map.get(key)
            if index is None:
                raise ValueError(f"Key {key} not found in measurement vector format")
            indices = [index]
        self.value[indices, 0] = value

    def __getitem__(
        self, 
        key: str | int | list[int]
        ) -> float | FloatArray:
        """
        Get value with key of format or index like:
        - z["confidence"]
        - z[0]
        """
        if isinstance(key, int | list):
            indices = key
        else:
            index = self._index_map.get(key)
            if index is None:
                raise ValueError(f"Key {key} not found in measurement vector format")
            indices = [index]
        return self.value[indices, 0]