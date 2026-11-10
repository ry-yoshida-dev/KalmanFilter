"""Process noise covariance ``Q``."""

from __future__ import annotations

from dataclasses import dataclass

from .parameter import KalmanFilterParameter


@dataclass(eq=False, repr=False)
class ProcessNoiseCovariance(KalmanFilterParameter):
    """
    Process noise covariance ``Q`` of the latest prediction.

    Attributes
    ----------
    value : FloatArray
        Covariance with shape ``(n, n)``.
    """
