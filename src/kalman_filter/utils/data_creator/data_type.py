from enum import Enum

class DataType(Enum):
    """
    Data type for Kalman filter test data.

    Attributes:
    ----------
    LINEAR: Linear data.
    SIN_WAVE: Sin wave data.
    RANDOM_WALK: Random walk data.
    """
    LINEAR = "linear"
    SIN_WAVE = "sin_wave"
    RANDOM_WALK = "random_walk"