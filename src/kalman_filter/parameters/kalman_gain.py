"""Kalman gain ``K``."""

from __future__ import annotations

from dataclasses import dataclass

from .parameter import KalmanFilterParameter


@dataclass(eq=False, repr=False)
class KalmanGain(KalmanFilterParameter):
    """
    Kalman gain ``K`` of the latest update.

    Attributes
    ----------
    value : FloatArray
        Gain with shape ``(n, observed)``.
    """
