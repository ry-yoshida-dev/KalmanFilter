from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .error_covariance_matrix import ErrorCovarianceMatrix
    from .measurement_matrix import MeasurementMatrix
    from .measurement_noise_covariance import MeasurementNoiseCovariance

from ..array_types import FloatArray
from .base import KFParameter

@dataclass
class KalmanGain(KFParameter):
    """
    Container class for Kalman gain matrix.
    * Kalman gain matrix is usually presented as K in mathematical notation.

    Attributes:
    ----------
    value: FloatArray
        Value of the Kalman gain matrix (inherited from base).
    """

    def update(
        self, 
        P: ErrorCovarianceMatrix, 
        H: MeasurementMatrix, 
        R: MeasurementNoiseCovariance, 
        PHT: FloatArray | None = None
        ) -> None:
        """ Update the Kalman gain matrix.
        
        Parameters
        ----------
        P: ErrorCovarianceMatrix
            Error covariance matrix.
        H: MeasurementMatrix
            Measurement matrix.
        R: MeasurementNoiseCovariance
            Measurement noise covariance matrix.
        PHT: FloatArray | None
            The matrix computed from P @ H.T.
            InnovationCovarianceMatrix needs P @ H.T in the update process.
        """
        pht_value = PHT if PHT is not None else (P.value @ H.T)
        S_value = R.value + (H.value @ pht_value)
        self.value = pht_value @ np.linalg.solve(S_value, np.eye(S_value.shape[0]))

    @classmethod
    def create_zero_matrix(
        cls, 
        dim_x: int, 
        dim_z: int
        ) -> KalmanGain:
        """
        Create a zero Kalman gain matrix.

        Parameters:
        ----------
        dim_x: int
            Dimension of the state vector.
        dim_z: int
            Dimension of the measurement vector.

        Returns:
        ---------
        KalmanGain
            Zero Kalman gain matrix.
        """
        return cls(value=np.zeros((dim_x, dim_z)))

