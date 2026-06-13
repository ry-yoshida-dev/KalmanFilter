from dataclasses import dataclass


from .base import KFParameter

@dataclass
class ControlInputMatrix(KFParameter):
    """
    Container class for control input matrix.
    * Control input matrix is usually presented as B in mathematical notation.

    Attributes:
    ----------
    value: FloatArray
        Value of the control input matrix (inherited from base).
    """
