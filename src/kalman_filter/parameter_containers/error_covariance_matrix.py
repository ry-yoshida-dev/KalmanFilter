from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .state_transition_matrix import StateTransitionMatrix
    from .process_noise_covariance import ProcessNoiseCovariance
    from .kalman_gain import KalmanGain
    from .measurement_matrix import MeasurementMatrix
    from .measurement_noise_covariance import MeasurementNoiseCovariance
    from .process_noise_gain import ProcessNoiseGain

from .base import KFParameter

@dataclass
class ErrorCovarianceMatrix(KFParameter):
    """
    Container class for error covariance matrix.
    * Error covariance matrix is usually presented as P in mathematical notation.

    Attributes:
    ----------
    value: np.ndarray
        Value of the error covariance matrix (inherited from base).
    """

    def predict(
        self, 
        F: StateTransitionMatrix, 
        Q: ProcessNoiseCovariance, 
        G: ProcessNoiseGain | None, 
        alpha: float = 1
        ) -> ErrorCovarianceMatrix:
        """ Predict the next error covariance matrix.

        Parameters
        ----------
        F: StateTransitionMatrix
            State transition matrix.
        Q: ProcessNoiseCovariance
            Process noise covariance matrix.
        G: ProcessNoiseGain | None
            Process noise gain matrix.
        alpha: float, optional
            The tuning coefficient for the error covariance matrix.

        Returns
        -------
        ErrorCovarianceMatrix
            Predicted error covariance matrix.
        """
        if G is None:
            value = F.value @ self.value @ F.T + Q.value
        else:
            value = F.value @ self.value @ F.T + G.value @ Q.value @ G.T
        value *= alpha
        return ErrorCovarianceMatrix(value=value)

    def update(
        self, 
        K: KalmanGain, 
        H: MeasurementMatrix, 
        R: MeasurementNoiseCovariance, 
        is_correlated_noise: bool = False
        ) -> None:
        """ Update the error covariance matrix from the current observation.

        Parameters
        ----------
        K: KalmanGain
            Kalman gain matrix.
        H: MeasurementMatrix
            Measurement matrix.
        R: MeasurementNoiseCovariance
            Measurement noise covariance matrix.
        is_correlated_noise: bool, optional
            If True, use correlated noise update. Default is False.
        """
        if is_correlated_noise:
            value = self.value - K.value @ (H.value @ self.value)
        else:
            I_KH = np.eye(K.shape[0]) - K.value @ H.value
            value = I_KH @ self.value @ I_KH.T + K.value @ R.value @ K.T
        self.value = value

    @classmethod
    def create_matrix(
        cls, 
        weights: list[float] | np.ndarray
        ) -> ErrorCovarianceMatrix:
        """ Create the error covariance matrix.

        Parameters
        ----------
        weights: list[float] | np.ndarray
            Weights for diagonal elements of the error covariance matrix.

        Returns
        -------
        ErrorCovarianceMatrix
            Error covariance matrix.
        """
        return cls(value=np.diag(weights))
