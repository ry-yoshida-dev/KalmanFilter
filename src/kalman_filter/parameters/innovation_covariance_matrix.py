"""Innovation covariance matrix ``S``."""

from __future__ import annotations

from dataclasses import dataclass

from .parameter import KalmanFilterParameter


@dataclass(eq=False, repr=False)
class InnovationCovarianceMatrix(KalmanFilterParameter):
    """
    Innovation covariance ``S`` of the latest update.

    Attributes
    ----------
    value : FloatArray
        Covariance with shape ``(observed, observed)``.
    """
