"""Third-degree spherical-radial cubature points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np

from ..sigma_point_scheme import SigmaPointScheme
from ..sigma_points import SigmaPoints


@dataclass(frozen=True)
class CubatureSigmaPointScheme(SigmaPointScheme):
    """
    ``2n`` equally weighted points at ``mean +/- sqrt(n) * sqrt(P)``.

    Using it in :class:`UnscentedKalmanFilter` gives the cubature Kalman
    filter (CKF), which has no tuning parameters and only positive weights.
    """

    @override
    def standard_points(self, dimension: int) -> SigmaPoints:
        """Return ``2n`` cubature points."""
        axes = np.sqrt(dimension) * np.eye(dimension)
        weights = np.full(2 * dimension, 0.5 / dimension)
        return SigmaPoints(np.concatenate([axes, -axes], axis=0), weights, weights)
