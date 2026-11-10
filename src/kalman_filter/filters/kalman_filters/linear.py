"""Standard (linear) Kalman filter."""

from __future__ import annotations

from typing import override

from numpy.typing import ArrayLike

from ...array_types import FloatArray
from ...models import (
    KinematicMotionModel,
    LinearMeasurementModel,
    LinearMotionModel,
    MeasurementModel,
    MotionModel,
)
from ..covariance_update import CovarianceUpdate
from ..projection import Projection
from .extended import ExtendedKalmanFilter


class LinearKalmanFilter(ExtendedKalmanFilter):
    """
    Standard Kalman filter for linear-Gaussian models.

    It accepts only models given by exact matrices (:class:`LinearMotionModel`,
    :class:`KinematicMotionModel`, and :class:`LinearMeasurementModel`), so
    the result is the exact posterior and nothing is approximated. ``F`` and
    ``H`` are the models' matrices. For nonlinear models use
    :class:`ExtendedKalmanFilter` or :class:`UnscentedKalmanFilter`.

    Parameters
    ----------
    motion_model : LinearMotionModel | KinematicMotionModel
        Process model.
    measurement_model : LinearMeasurementModel
        Default sensor model.
    x : ArrayLike
        Initial state with shape ``(n,)``.
    P : ArrayLike
        Initial covariance with shape ``(n, n)``, or its diagonal.
    covariance_update : CovarianceUpdate
        Keyword-only. Form of the posterior covariance update; Joseph by default.
    fading_memory_factor : float
        Keyword-only. Covariance inflation applied before each prediction (``>= 1``).

    Raises
    ------
    TypeError
        If a model is not linear, here or in a later :meth:`update`.
    ValueError
        If the measurement matrix does not have one column per state variable.

    Examples
    --------
    >>> motion = KinematicMotionModel.constant_velocity(["x"], noise_intensity=0.1)
    >>> sensor = LinearMeasurementModel.from_observed_states(motion.layout, ["x"], [0.5])
    >>> kf = LinearKalmanFilter(motion, sensor, x=[0.0, 0.0], P=[1.0, 1.0])
    >>> kf.predict(dt=0.1)
    >>> kf.update([0.2])
    """

    def __init__(
        self,
        motion_model: LinearMotionModel | KinematicMotionModel,
        measurement_model: LinearMeasurementModel,
        x: ArrayLike,
        P: ArrayLike,
        *,
        covariance_update: CovarianceUpdate = CovarianceUpdate.JOSEPH,
        fading_memory_factor: float = 1.0,
    ) -> None:
        _require_linear_motion(motion_model)
        _require_linear_measurement(measurement_model, motion_model.dimension)
        super().__init__(
            motion_model,
            measurement_model,
            x,
            P,
            covariance_update=covariance_update,
            fading_memory_factor=fading_memory_factor,
        )

    @override
    def _project(self, x: FloatArray, P: FloatArray, model: MeasurementModel) -> Projection:
        _require_linear_measurement(model, self.motion_model.dimension)
        return super()._project(x, P, model)


def _require_linear_motion(model: MotionModel) -> None:
    """Reject a motion model that is not given by exact matrices."""
    if not isinstance(model, LinearMotionModel | KinematicMotionModel):
        raise TypeError(
            "LinearKalmanFilter needs a LinearMotionModel or KinematicMotionModel, "
            + f"got {type(model).__name__}; use ExtendedKalmanFilter or UnscentedKalmanFilter"
        )


def _require_linear_measurement(model: MeasurementModel, state_dimension: int) -> None:
    """Reject a measurement model that is not a matrix with one column per state variable."""
    if not isinstance(model, LinearMeasurementModel):
        raise TypeError(
            f"LinearKalmanFilter needs a LinearMeasurementModel, got {type(model).__name__}; "
            + "use ExtendedKalmanFilter or UnscentedKalmanFilter"
        )
    columns = model.measurement_matrix.shape[1]
    if columns != state_dimension:
        raise ValueError(
            f"measurement_matrix must have {state_dimension} columns (one per state variable), "
            + f"got {columns}"
        )
