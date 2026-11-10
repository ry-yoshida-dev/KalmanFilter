"""Form of the posterior covariance update."""

from __future__ import annotations

from enum import Enum


class CovarianceUpdate(Enum):
    """
    How the posterior covariance is computed from the gain.

    Attributes
    ----------
    JOSEPH
        ``(I - K H) P (I - K H)^T + K R K^T``. Stays symmetric positive
        semi-definite under rounding; needs a Jacobian, so only filters that
        linearize use it.
    STANDARD
        ``P - K S K^T``. Cheaper, exact in exact arithmetic.
    """

    JOSEPH = "joseph"
    STANDARD = "standard"
