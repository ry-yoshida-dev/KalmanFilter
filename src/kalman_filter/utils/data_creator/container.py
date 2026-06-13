from dataclasses import dataclass
from typing import Generator

from ...array_types import FloatArray


@dataclass
class KalmanTestData:
    """
    Data container for Kalman filter test data.

    Attributes:
    ----------
    time: FloatArray
        The time array.
    ground_truth: FloatArray
        The ground truth data.
    measurements: FloatArray
        The synthetic measurements data.
        Noise is added to the ground truth data based on gaussian noise.
    """
    time: FloatArray
    ground_truth: FloatArray
    measurements: FloatArray

    def __iter__(self) -> Generator[tuple[float, FloatArray, FloatArray], None, None]:
        for time, ground_truth, measurements in zip(self.time, self.ground_truth, self.measurements):
            yield time, ground_truth, measurements