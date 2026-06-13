from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from collections.abc import Callable

from ...array_types import FloatArray
from .data_type import DataType
from .container import KalmanTestData

@dataclass
class SampleDataCreator:
    """
    A class to create sample data for Kalman filter.

    Attributes
    ----------
    ndim: int
        The number of dimensions of the data.
    steps: int
        The number of steps of the data.
    sigma: float
        The standard deviation of the measurement noise.
    data_type: DataType
        The type of the data.
    dt: float
        The time step.
    initial_value: float
        The initial value of the data.
    seed: int
        The seed of the random number generator.
    """
    ndim: int
    steps: int
    sigma: float
    dt: float = 0.1
    initial_value: float = 0.0
    seed: int = 114514
 
    def __post_init__(self):
        self.time = np.arange(0, self.steps * self.dt, self.dt)

    def create(
        self, 
        data_type: DataType
        ) -> KalmanTestData:
        """
        Create the test data.

        Parameters
        ----------
        data_type: DataType
            The type of the data.

        Returns
        -------
        KalmanTestData: The test data.
        """
        function = self._define_function(data_type)
        time = np.arange(0, self.steps * self.dt, self.dt)
        
        ground_truth = function(time)

        measurement_noise = np.random.normal(0, self.sigma, size=(self.steps, self.ndim))
        measurements = ground_truth + measurement_noise

        return KalmanTestData(
            time=time,
            ground_truth=ground_truth,
            measurements=measurements
            )

    def _define_function(
        self,
        data_type: DataType
        ) -> Callable[[FloatArray], FloatArray]:
        """
        Define the function to create the ground truth data.

        Parameters
        ----------
        data_type: DataType
            The type of the data.

        Returns
        -------
        Callable[[FloatArray], FloatArray]: 
            A callable function that takes the time array and returns the ground truth data.
        """
        match data_type:
            case DataType.LINEAR:
                return self._create_linear_gt
            case DataType.SIN_WAVE:
                return self._create_sin_wave_gt
            case DataType.RANDOM_WALK:
                return self._create_random_walk_gt

    def _create_linear_gt(
        self,
        time: FloatArray
        ) -> FloatArray:
        """
        Create the linear ground truth data.

        Parameters
        ----------
        time: FloatArray
            The time array.

        Returns
        -------
        FloatArray: The ground truth data in linear format with shape (steps, ndim).
        """
        velocities = np.array([1.0 + i * 0.5 for i in range(self.ndim)])
        return self.initial_value + np.outer(time, velocities)

    def _create_sin_wave_gt(
        self,
        time: FloatArray
        ) -> FloatArray:
        """
        Create the sin wave ground truth data.

        Parameters
        ----------
        time: FloatArray
            The time array.

        Returns
        -------
        FloatArray: The ground truth data in sin wave format with shape (steps, ndim).
        """
        ground_truth = np.zeros((self.steps, self.ndim))
        for i in range(self.ndim):
            omega = 1.0
            phase = i * (np.pi / 2)
            ground_truth[:, i] = np.sin(omega * time + phase) + self.initial_value
        return ground_truth

    def _create_random_walk_gt(
        self,
        time: FloatArray
        ) -> FloatArray:
        """
        Create the random walk ground truth data.

        Parameters
        ----------
        time: FloatArray
            The time array.

        Returns
        -------
        FloatArray: The ground truth data in random walk format with shape (steps, ndim).
        """
        process_noise_std = 0.1
        ground_truth = np.zeros((self.steps, self.ndim))
        ground_truth[0] = self.initial_value
        for t in range(1, self.steps):
            noise = np.random.normal(0, process_noise_std, size=self.ndim)
            ground_truth[t] = ground_truth[t-1] + noise
        return ground_truth


#################
# Example usage #
#################

def main():

    data_creator = SampleDataCreator(
        ndim=2, 
        steps=100, 
        sigma=0.3
        )
    data: KalmanTestData = data_creator.create(DataType.LINEAR)
    print(data)

if __name__ == "__main__":
    main()
    pass