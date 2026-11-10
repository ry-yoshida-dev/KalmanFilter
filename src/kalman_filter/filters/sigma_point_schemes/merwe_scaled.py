"""Van der Merwe scaled sigma points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np

from ..sigma_point_scheme import SigmaPointScheme
from ..sigma_points import SigmaPoints


@dataclass(frozen=True)
class MerweScaledSigmaPointScheme(SigmaPointScheme):
    """
    Scaled unscented transform with ``2n + 1`` points.

    The defaults (``alpha=1``, ``beta=2``, ``kappa=0``) keep every weight
    non-negative, so propagated covariances stay positive semi-definite.
    Smaller ``alpha`` concentrates the points around the mean.

    Attributes
    ----------
    alpha : float
        Spread of the points around the mean, in ``(0, 1]``.
    beta : float
        Prior knowledge of the distribution; ``2`` is optimal for Gaussians.
    kappa : float
        Secondary scaling parameter.
    """

    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")

    @override
    def standard_points(self, dimension: int) -> SigmaPoints:
        """
        Return ``2n + 1`` scaled sigma points.

        Raises
        ------
        ValueError
            If ``alpha^2 * (n + kappa)`` is not positive.
        """
        spread = self.alpha**2 * (dimension + self.kappa)
        if spread <= 0.0:
            raise ValueError(f"alpha^2 * (n + kappa) must be positive, got {spread}")
        axes = np.sqrt(spread) * np.eye(dimension)
        offsets = np.concatenate([np.zeros((1, dimension)), axes, -axes], axis=0)
        mean_weights = np.full(2 * dimension + 1, 0.5 / spread)
        mean_weights[0] = (spread - dimension) / spread
        covariance_weights = mean_weights.copy()
        covariance_weights[0] += 1.0 - self.alpha**2 + self.beta
        return SigmaPoints(offsets, mean_weights, covariance_weights)
