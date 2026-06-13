from dataclasses import dataclass

from .base import KFParameter

@dataclass
class ControlInputVector(KFParameter):
    """
    Container class for control input vector.
    * Control input vector is usually presented as u in mathematical notation.

    Attributes:
    ----------
    value: FloatArray
        Value of the control input vector (inherited from base).
    """
