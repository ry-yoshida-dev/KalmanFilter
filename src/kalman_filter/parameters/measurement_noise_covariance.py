"""Measurement noise covariance ``R``."""

from __future__ import annotations

from dataclasses import dataclass

from .parameter import KalmanFilterParameter


@dataclass(eq=False, repr=False)
class MeasurementNoiseCovariance(KalmanFilterParameter):
    """
    Measurement noise covariance ``R`` of the latest update.

    Attributes
    ----------
    value : FloatArray
        Covariance with shape ``(observed, observed)``.
    """
