"""Extended Kalman filter."""

from __future__ import annotations

from typing import override

from numpy.typing import ArrayLike

from ..._numerics import symmetrize
from ...array_types import FloatArray
from ...models import MeasurementModel, MotionModel
from ..covariance_update import CovarianceUpdate
from ..kalman_filter import KalmanFilter
from ..projection import Projection
from ..propagation import Propagation


class ExtendedKalmanFilter(KalmanFilter):
    """
    Kalman filter that linearizes the models with their Jacobians.

    ``F`` and ``H`` hold the Jacobians evaluated in the latest step. With
    linear models the Jacobians are exact and the result equals
    :class:`LinearKalmanFilter`, which also checks that the models are linear.

    Parameters
    ----------
    motion_model : MotionModel
        Process model.
    measurement_model : MeasurementModel
        Default sensor model.
    x : ArrayLike
        Initial state with shape ``(n,)``.
    P : ArrayLike
        Initial covariance with shape ``(n, n)``, or its diagonal.
    covariance_update : CovarianceUpdate
        Keyword-only. Form of the posterior covariance update; Joseph by default.
    fading_memory_factor : float
        Keyword-only. Covariance inflation applied before each prediction (``>= 1``).

    Examples
    --------
    >>> motion = ConstantTurnRateVelocityMotionModel(1.0, 0.1)
    >>> sensor = RangeBearingMeasurementModel([0, 1], [0.1, 0.01])
    >>> kf = ExtendedKalmanFilter(motion, sensor, x=[10, 0, 1, 0, 0], P=[1, 1, 1, 0.1, 0.01])
    >>> kf.predict(dt=0.1)
    >>> kf.update([10.1, 0.02])
    >>> kf.x, kf.P, kf.K
    """

    def __init__(
        self,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        x: ArrayLike,
        P: ArrayLike,
        *,
        covariance_update: CovarianceUpdate = CovarianceUpdate.JOSEPH,
        fading_memory_factor: float = 1.0,
    ) -> None:
        super().__init__(
            motion_model, measurement_model, x, P, fading_memory_factor=fading_memory_factor
        )
        self.covariance_update: CovarianceUpdate = covariance_update

    @override
    def _uses_joseph_form(self) -> bool:
        return self.covariance_update is CovarianceUpdate.JOSEPH

    @override
    def _propagate(
        self, x: FloatArray, P: FloatArray, dt: float, control: FloatArray | None
    ) -> Propagation:
        transition_matrix = self.motion_model.jacobian(x, dt, control)
        process_noise = self.motion_model.process_noise(x, dt)
        return Propagation(
            x=self.motion_model.transition(x, dt, control),
            P=transition_matrix @ P @ transition_matrix.T + process_noise,
            F=transition_matrix,
            Q=process_noise,
        )

    @override
    def _project(self, x: FloatArray, P: FloatArray, model: MeasurementModel) -> Projection:
        return self._linearize_at(x, P, x, model)

    def _linearize_at(
        self,
        x: FloatArray,
        P: FloatArray,
        linearization_point: FloatArray,
        model: MeasurementModel,
    ) -> Projection:
        """First-order expansion of ``model`` at ``linearization_point``, against the prior."""
        measurement_matrix = model.jacobian(linearization_point, self.motion_model.add)
        noise_covariance = model.noise_covariance(linearization_point)
        offset = self.motion_model.residual(x, linearization_point)
        cross_covariance = P @ measurement_matrix.T
        return Projection(
            z=model.measure(linearization_point) + measurement_matrix @ offset,
            S=symmetrize(measurement_matrix @ cross_covariance + noise_covariance),
            cross_covariance=cross_covariance,
            H=measurement_matrix,
            R=noise_covariance,
        )
