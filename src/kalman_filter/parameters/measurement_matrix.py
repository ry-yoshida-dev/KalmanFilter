"""Measurement matrix ``H``."""

from __future__ import annotations

from dataclasses import dataclass

from .parameter import KalmanFilterParameter


@dataclass(eq=False, repr=False)
class MeasurementMatrix(KalmanFilterParameter):
    """
    Measurement matrix ``H`` (Jacobian of the measurement model) of the latest update.

    Attributes
    ----------
    value : FloatArray
        Matrix with shape ``(observed, n)``.
    """
