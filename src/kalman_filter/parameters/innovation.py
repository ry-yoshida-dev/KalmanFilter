"""Innovation ``y``."""

from __future__ import annotations

from dataclasses import dataclass

from .parameter import KalmanFilterParameter


@dataclass(eq=False, repr=False)
class Innovation(KalmanFilterParameter):
    """
    Innovation ``y = z - h(x)`` of the latest update.

    Attributes
    ----------
    value : FloatArray
        Vector with shape ``(observed,)``.
    """
