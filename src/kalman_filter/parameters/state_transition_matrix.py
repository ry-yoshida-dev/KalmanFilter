"""State transition matrix ``F``."""

from __future__ import annotations

from dataclasses import dataclass

from .parameter import KalmanFilterParameter


@dataclass(eq=False, repr=False)
class StateTransitionMatrix(KalmanFilterParameter):
    """
    State transition matrix ``F`` (Jacobian of the motion model) of the latest prediction.

    Attributes
    ----------
    value : FloatArray
        Matrix with shape ``(n, n)``.
    """
