"""Iterated unscented Kalman filter (iterated posterior linearization)."""

from __future__ import annotations

from typing import override

from numpy.typing import ArrayLike

from ..._numerics import symmetrize
from ...array_types import BoolArray, FloatArray
from ...models import MeasurementModel, MotionModel
from ..projection import Projection
from ..sigma_point_scheme import SigmaPointScheme
from .unscented import UnscentedKalmanFilter


class IteratedUnscentedKalmanFilter(UnscentedKalmanFilter):
    """
    Unscented filter that refits the measurement model around the posterior.

    This is the iterated posterior linearization filter (IPLF): each iteration
    places sigma points on the current posterior, fits the measurement function
    by statistical linear regression ``h(x) ~ A x + b + e``, ``e ~ N(0, Omega)``,
    and redoes the update of the prior with that affine model. The fit is made
    where the posterior mass is, so precise nonlinear measurements no longer
    produce overconfident or biased estimates. The first iteration is the plain
    unscented update. ``H`` holds the slope ``A`` of the final fit and ``S``
    contains its linearization error ``Omega``.

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
        Keyword-only. Upper bound on linearizations per update; ``1`` is the
        plain unscented filter.
    tolerance : float
        Keyword-only. Convergence threshold on the state change, in prior standard deviations.
    sigma_point_scheme : SigmaPointScheme | None
        Keyword-only. Sigma point placement; ``MerweScaledSigmaPointScheme()`` when ``None``.
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
        sigma_point_scheme: SigmaPointScheme | None = None,
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
            sigma_point_scheme=sigma_point_scheme,
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
            lambda x, P: self._fit_at(prior_x, prior_P, x, P, model),
            self.max_iterations,
            self.tolerance,
        )

    def _fit_at(
        self,
        x: FloatArray,
        P: FloatArray,
        linearization_x: FloatArray,
        linearization_P: FloatArray,
        model: MeasurementModel,
    ) -> Projection:
        """Statistical linear regression of ``model`` over the posterior, against the prior."""
        # The unscented projection of the posterior holds the regression: its H is the
        # slope A, and its S = A P A^T + Omega + R, so Omega + R = S - A P A^T.
        moments = self._project(linearization_x, linearization_P, model)
        slope = moments.H
        error_and_noise = symmetrize(moments.S - slope @ linearization_P @ slope.T)
        offset = self.motion_model.residual(x, linearization_x)
        cross_covariance = P @ slope.T
        return Projection(
            z=moments.z + slope @ offset,
            S=symmetrize(slope @ cross_covariance + error_and_noise),
            cross_covariance=cross_covariance,
            H=slope,
            R=moments.R,
        )
