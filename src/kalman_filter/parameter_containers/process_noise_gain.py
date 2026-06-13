from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from ..array_types import FloatWeights
from .base import KFParameter

@dataclass
class ProcessNoiseGain(KFParameter):
    """
    Process noise gain matrix.
    Represented as G in mathematical notation.

    Attributes:
    ----------
    value: FloatArray
        Value of the process noise gain matrix (inherited from base).
    """

    @classmethod
    def create(
        cls, 
        diagonal_values: FloatWeights
        ) -> ProcessNoiseGain:
        """
        Create the process noise gain matrix.

        Parameters
        ----------
        diagonal_values: FloatWeights
            Diagonal values for the process noise gain matrix.

        Returns
        -------
        ProcessNoiseGain
            Process noise gain matrix.
        """
        values = np.array(diagonal_values).flatten()
        return cls(value=np.diag(values))

    @classmethod
    def create_identity(
        cls, 
        dim: int
        ) -> ProcessNoiseGain:
        """ 
        Create the identity process noise gain matrix.

        Parameters
        ----------
        dim: int
            Dimension of the process noise gain matrix.

        Returns
        -------
        ProcessNoiseGain
            Identity process noise gain matrix.
        """
        return cls(value=np.eye(dim))