"""Iterated extended Kalman filter."""

from __future__ import annotations

from typing import override

from numpy.typing import ArrayLike

from ...array_types import BoolArray, FloatArray
from ...models import MeasurementModel, MotionModel
from ..covariance_update import CovarianceUpdate
from .extended import ExtendedKalmanFilter


class IteratedExtendedKalmanFilter(ExtendedKalmanFilter):
    """
    Extended Kalman filter that relinearizes the measurement model at the posterior.

    Each update is a Gauss-Newton search for the maximum a posteriori state:
    the measurement Jacobian is re-evaluated at the latest estimate until it
    stops moving. This removes most of the EKF's bias when the measurement is
    strongly nonlinear and precise. ``H``, ``y``, ``S``, and ``K`` refer to the
    final linearization. With a linear measurement model it equals the
    extended filter.

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
    max_iterations : int
        Keyword-only. Upper bound on linearizations per update; ``1`` is the plain EKF.
    tolerance : float
        Keyword-only. Convergence threshold on the state change, in prior standard deviations.
    covariance_update : CovarianceUpdate
        Keyword-only. Form of the posterior covariance update; Joseph by default.
    fading_memory_factor : float
        Keyword-only. Covariance inflation applied before each prediction (``>= 1``).

    Raises
    ------
    ValueError
        If ``max_iterations`` is smaller than 1 or ``tolerance`` is not positive.
    """

    def __init__(
        self,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        x: ArrayLike,
        P: ArrayLike,
        *,
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        covariance_update: CovarianceUpdate = CovarianceUpdate.JOSEPH,
        fading_memory_factor: float = 1.0,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")
        if not tolerance > 0.0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        super().__init__(
            motion_model,
            measurement_model,
            x,
            P,
            covariance_update=covariance_update,
            fading_memory_factor=fading_memory_factor,
        )
        self.max_iterations: int = max_iterations
        self.tolerance: float = tolerance

    @override
    def _correct(
        self, observation: FloatArray, is_observed: BoolArray, model: MeasurementModel
    ) -> None:
        prior_x, prior_P = self._state, self._covariance
        self._iterate_correction(
            observation,
            is_observed,
            model,
            lambda x, _: self._linearize_at(prior_x, prior_P, x, model),
            self.max_iterations,
            self.tolerance,
        )
