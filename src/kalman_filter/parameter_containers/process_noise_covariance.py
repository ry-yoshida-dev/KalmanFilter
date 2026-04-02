from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .state_vector import StateVector
from .base import KFParameter

@dataclass
class ProcessNoiseCovariance(KFParameter):
    """
    Container class for process noise covariance matrix.
    * Process noise covariance matrix is usually presented as Q in mathematical notation.
    * Shape: (len(state_vector), len(state_vector))
    
    Higher value means more uncertainty in the process.

    Attributes:
    ----------
    value: np.ndarray
        Value of the process noise covariance matrix (inherited from base).
    """

    @classmethod
    def create_matrix(
        cls, 
        weights: list[float] | np.ndarray
        ) -> ProcessNoiseCovariance:
        """ Create the process noise covariance matrix.

        Parameters
        ----------
        weights: list[float] | np.ndarray
            Weights for diagonal elements of the process noise covariance matrix.

        Returns
        -------
        ProcessNoiseCovariance
            Process noise covariance matrix.
        """
        return ProcessNoiseCovariance(value=np.diag(weights))

    @classmethod
    def create_matrix_from_state_vector(
        cls, 
        state_vector: StateVector
        ) -> ProcessNoiseCovariance:
        """ Create the process noise covariance matrix from the state vector.

        Returns
        -------
        ProcessNoiseCovariance
            Process noise covariance matrix that is the identity matrix.
        """
        return cls(value=np.eye(len(state_vector)))
