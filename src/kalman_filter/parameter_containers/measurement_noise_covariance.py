from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from ..array_types import FloatWeights
from .base import KFParameter

@dataclass
class MeasurementNoiseCovariance(KFParameter):
    """
    Measurement noise covariance matrix.
    Represented as R in mathematical notation.
    
    Higher value means less confidence in the measurement.

    Attributes:
    ----------
    value: FloatArray
        Value of the measurement noise covariance matrix (inherited from base).
    """

    def validate_shape(
        self, 
        dim_z: int
        ) -> None:
        """
        Validate the measurement noise covariance matrix.
        * Validation process can't be done by this class only because it needs the dimension of the measurement vector.

        Parameters
        ----------
        dim_z: int
            Dimension of the measurement vector.
        """
        if len(self.value) != dim_z:
            raise ValueError(f"Measurement noise covariance matrix must have {dim_z} dimensions. Current dimensions: {len(self.value)}")

    @classmethod
    def create_matrix(
        cls,
        weights: FloatWeights, 
        scaling_factor: float = 1
        ) -> MeasurementNoiseCovariance:
        """ Create the measurement noise covariance matrix.

        Parameters
        ----------
        weights: FloatWeights
            Weights for the measurement noise covariance matrix.
        scaling_factor: float
            Scaling factor for the measurement noise covariance matrix.
            If factor larger than 1, the measurement noise covariance matrix will be scaled up.
            -> It means more uncertainty in the measurement and more confidence in the KF prediction.

        Returns
        -------
        MeasurementNoiseCovariance
            Measurement noise covariance matrix.
        """
        return cls(value=np.diag(weights) * scaling_factor)