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
class InnovationCovarianceMatrix(KFParameter):
    """
    Container class for innovation covariance matrix.
    * Innovation covariance matrix is usually presented as S in mathematical notation.

    Attributes:
    ----------
    value: FloatArray
        Value of the innovation covariance matrix (inherited from base).
    """

    def update(
        self, 
        P: ErrorCovarianceMatrix, 
        H: MeasurementMatrix, 
        R: MeasurementNoiseCovariance, 
        PHT: FloatArray | None = None
        ) -> None:
        """ Update the innovation covariance matrix.

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
            KalmanGain needs P @ H.T in the update process.
        """
        pht_value = PHT if PHT is not None else (P.value @ H.T)
        self.value = R.value + (H.value @ pht_value)

    @classmethod
    def create_zero_matrix(
        cls, 
        dim_z: int
        ) -> InnovationCovarianceMatrix:
        """
        Create a zero innovation covariance matrix.

        Parameters:
        ----------
        dim_z: int
            Dimension of the measurement vector.

        Returns:
        ---------
        InnovationCovarianceMatrix
            Zero innovation covariance matrix.
        """
        return cls(value=np.zeros((dim_z, dim_z)))

    