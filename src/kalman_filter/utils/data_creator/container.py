import numpy as np
from dataclasses import dataclass
from typing import Generator

@dataclass
class KalmanTestData:
    """
    Data container for Kalman filter test data.

    Attributes:
    ----------
    time: np.ndarray
        The time array.
    ground_truth: np.ndarray
        The ground truth data.
    measurements: np.ndarray
        The synthetic measurements data.
        Noise is added to the ground truth data based on gaussian noise.
    """
    time: np.ndarray
    ground_truth: np.ndarray
    measurements: np.ndarray

    def __iter__(self) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        for time, ground_truth, measurements in zip(self.time, self.ground_truth, self.measurements):
            yield time, ground_truth, measurements