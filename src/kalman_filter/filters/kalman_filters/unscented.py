"""Unscented (sigma point) Kalman filter."""

from __future__ import annotations

from collections.abc import Callable
from typing import override

import numpy as np
from numpy.typing import ArrayLike

from ..._numerics import symmetrize
from ...array_types import FloatArray
from ...models import MeasurementModel, MotionModel
from ..kalman_filter import KalmanFilter
from ..projection import Projection
from ..propagation import Propagation
from ..sigma_point_scheme import SigmaPointScheme
from ..sigma_point_schemes import MerweScaledSigmaPointScheme


class UnscentedKalmanFilter(KalmanFilter):
    """
    Kalman filter that propagates deterministic sigma points through the models.

    It needs no Jacobians and captures nonlinearities to second order. Pass
    :class:`CubatureSigmaPointScheme` to obtain the cubature Kalman filter.
    ``F`` and ``H`` hold the statistical linearization of the models over
    the sigma points (the regression slope ``Cov(x, f(x)) P^-1``), which
    equals the models' matrices when they are linear. ``R`` is the sensor
    noise; ``S`` also contains the linearization error of the measurement.

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
    sigma_point_scheme : SigmaPointScheme | None
        Keyword-only. Sigma point placement; ``MerweScaledSigmaPointScheme()`` when ``None``.
    fading_memory_factor : float
        Keyword-only. Covariance inflation applied before each prediction (``>= 1``).
    """

    def __init__(
        self,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        x: ArrayLike,
        P: ArrayLike,
        *,
        sigma_point_scheme: SigmaPointScheme | None = None,
        fading_memory_factor: float = 1.0,
    ) -> None:
        super().__init__(
            motion_model, measurement_model, x, P, fading_memory_factor=fading_memory_factor
        )
        self.sigma_point_scheme: SigmaPointScheme = (
            MerweScaledSigmaPointScheme() if sigma_point_scheme is None else sigma_point_scheme
        )

    @override
    def _propagate(
        self, x: FloatArray, P: FloatArray, dt: float, control: FloatArray | None
    ) -> Propagation:
        model = self.motion_model
        mean, covariance, cross_covariance = self._transform(
            x, P, lambda states: model.transition(states, dt, control), model.mean, model.residual
        )
        process_noise = model.process_noise(x, dt)
        return Propagation(
            x=mean,
            P=covariance + process_noise,
            F=_regression_slope(P, cross_covariance),
            Q=process_noise,
        )

    @override
    def _project(self, x: FloatArray, P: FloatArray, model: MeasurementModel) -> Projection:
        mean, covariance, cross_covariance = self._transform(
            x, P, model.measure, model.mean, model.residual
        )
        noise_covariance = model.noise_covariance(x)
        return Projection(
            z=mean,
            S=symmetrize(covariance + noise_covariance),
            cross_covariance=cross_covariance,
            H=_regression_slope(P, cross_covariance),
            R=noise_covariance,
        )

    def _transform(
        self,
        x: FloatArray,
        P: FloatArray,
        function: Callable[[FloatArray], FloatArray],
        mean: Callable[[FloatArray, FloatArray], FloatArray],
        residual: Callable[[FloatArray, FloatArray], FloatArray],
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Unscented transform of ``N(x, P)`` through ``function``.

        The sigma points are placed with the motion model's state arithmetic,
        and the outputs averaged and differenced with ``mean`` and ``residual``.
        Returns the output mean, the output covariance, and the input-output
        cross covariance.
        """
        sigma_points = self.sigma_point_scheme.generate(P)
        count = sigma_points.offsets.shape[0]
        outputs = function(self.motion_model.add(np.tile(x, (count, 1)), sigma_points.offsets))
        output_mean = mean(outputs, sigma_points.mean_weights)
        deviation = residual(outputs, output_mean)
        weighted_deviation = deviation * sigma_points.covariance_weights[:, None]
        return (
            output_mean,
            weighted_deviation.T @ deviation,
            sigma_points.offsets.T @ weighted_deviation,
        )


def _regression_slope(covariance: FloatArray, cross_covariance: FloatArray) -> FloatArray:
    """
    Slope ``A`` of the statistical linear regression ``g(x) ~ A x + b``.

    ``A = C^T P^+`` for the input covariance ``P`` and the cross covariance
    ``C = Cov(x, g(x))``; the pseudo-inverse keeps exactly known components
    (zero variance) well defined.
    """
    precision = np.linalg.pinv(covariance, hermitian=True)
    slope: FloatArray = (precision @ cross_covariance).T
    return slope
