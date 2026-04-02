from __future__ import annotations
import numpy as np
from abc import ABC
from dataclasses import dataclass

@dataclass
class KFParameter(ABC):
    """
    Base container for Kalman filter parameters.
    All subclasses hold a `value` array (np.ndarray).
    """
    value: np.ndarray
    
    def replace_value(
        self, 
        value: np.ndarray
        ) -> None:
        """ 
        Replace the value of the matrix by the new value.

        Parameters
        ----------
        value: np.ndarray
            Value to register.

        Raises
        ------
        ValueError: If the shape of the new value is not the same as the original value.
        """
        if self.value.shape != value.shape:
            raise ValueError(f"Shape mismatch. Original shape: {self.value.shape}, new shape: {value.shape}")
        self.value = value
    
    def __add__(self, other: "KFParameter") -> np.ndarray:
        """ Matrix addition with + operator.
        """
        return self.value + other.value
    
    def __sub__(self, other: "KFParameter") -> np.ndarray:
        """ Matrix subtraction with - operator.
        """
        return self.value - other.value
    
    def __matmul__(self, other: "KFParameter") -> np.ndarray:
        """ Matrix multiplication with @ operator.
        """
        return self.value @ other.value

    @property
    def shape(self):
        """ Shape of the matrix.
        """
        return self.value.shape

    @property
    def T(self):
        """ Transpose of the matrix.
        """
        return self.value.T
    
    @property
    def inv(self):
        """ Inverse of the matrix.
        """
        return np.linalg.inv(self.value)

