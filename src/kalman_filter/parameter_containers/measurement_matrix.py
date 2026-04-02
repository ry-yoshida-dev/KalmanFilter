from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .state_vector import StateVector
    from .measurement_vector import MeasurementVector

from .base import KFParameter

@dataclass
class MeasurementMatrix(KFParameter):
    """
    Container class for measurement matrix.
    Measurement matrix is a matrix that projects the state vector into the measurement vector.
    * Measurement matrix is usually presented as H in mathematical notation.

    Attributes:
    ----------
    value: np.ndarray
        Value of the measurement matrix (inherited from base).
    """

    @classmethod
    def create_matrix(
        cls,
        state_vector: StateVector, 
        measurement_vector: MeasurementVector,
        matrix_value: np.ndarray | list[float] | None = None
        ) -> MeasurementMatrix:
        """ 
        Create the measurement matrix.

        Parameters
        ----------
        state_vector: StateVector
            State vector.
        measurement_vector: MeasurementVector
            Measurement vector.

        Returns
        -------
        MeasurementMatrix
            Measurement matrix.
        """
        width = measurement_vector.dim
        height = state_vector.dim
        
        if matrix_value is None:
            # Create a rectangular identity matrix: maps first 'width' components of state to measurement
            # If width == height, this is a standard identity matrix
            # If width < height, this maps the first 'width' state components directly
            # If width > height, this is not a valid Kalman filter configuration
            value = np.eye(width, height)
            return cls(value=value)
        elif isinstance(matrix_value, np.ndarray):
            if matrix_value.shape != (width, height):
                raise ValueError(f"Matrix value must have shape ({width}, {height}). Current shape: {matrix_value.shape}")
            return cls(value=matrix_value)
        else:
            if len(matrix_value) != width * height:
                raise ValueError(f"Matrix value must have {width * height} elements. Current length: {len(matrix_value)}")
            value = np.array(matrix_value).reshape(width, height)
            return cls(value=value)

    def state2measurement(
        self, 
        x: StateVector
        ) -> np.ndarray:
        """ 
        Project the state into measurement space.
        
        NOTE
        This method returns the value of MeasurementVector, not the MeasurementVector object itself.
        -> MeasurementVector object needs not only value, but also format and index_map.
        -> Use MeasurementVector.replace_value() to replace the value of MeasurementVector.

        Parameters
        ----------
        x: StateVector
            State vector.

        Returns
        -------
        np.ndarray
            Value of MeasurementVector.
        """
        return self.value @ x.value

    @property
    def is_square_matrix(self) -> bool:
        """
        Check if the measurement matrix is square.
        
        Returns
        -------
        bool
            True if the measurement matrix is square, False otherwise.
        """
        return self.value.shape[0] == self.value.shape[1]

    @property
    def pseudo_inverse_matrix(self) -> np.ndarray:
        """
        Return the pseudo inverse of the measurement matrix.
        
        Returns
        -------
        np.ndarray
            The pseudo inverse of the measurement matrix.
        """
        return np.linalg.pinv(self.value)
