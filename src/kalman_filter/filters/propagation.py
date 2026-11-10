"""Prior produced by one prediction."""

from __future__ import annotations

from dataclasses import dataclass

from ..array_types import FloatArray


@dataclass(frozen=True)
class Propagation:
    """
    Prior produced by one prediction.

    Attributes
    ----------
    x : FloatArray
        Predicted state ``x_{k|k-1}``.
    P : FloatArray
        Predicted covariance ``P_{k|k-1}``.
    F : FloatArray
        State transition matrix: the Jacobian of the motion model, or its
        statistical linearization for sigma point filters.
    Q : FloatArray
        Process noise covariance applied in this step.
    """

    x: FloatArray
    P: FloatArray
    F: FloatArray
    Q: FloatArray
