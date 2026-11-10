"""Fifth-degree spherical-radial cubature points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np

from ..sigma_point_scheme import SigmaPointScheme
from ..sigma_points import SigmaPoints


@dataclass(frozen=True)
class FifthDegreeCubatureSigmaPointScheme(SigmaPointScheme):
    """
    ``2n^2 + 1`` points that integrate all polynomials up to degree 5 exactly.

    The points are the mean, ``+/- sqrt(n + 2)`` along every axis, and
    ``sqrt((n + 2) / 2) * (+/- e_i +/- e_j)`` for every axis pair. It is markedly
    more accurate than the third-degree :class:`CubatureSigmaPointScheme` on
    strongly nonlinear models while growing only quadratically with ``n``.
    The axis weights are negative for ``n > 4``, so predicted covariances can
    lose definiteness in high dimensions with very nonlinear models.
    """

    @override
    def standard_points(self, dimension: int) -> SigmaPoints:
        """Return ``2n^2 + 1`` fifth-degree cubature points."""
        spread = dimension + 2.0
        axes = np.sqrt(spread) * np.eye(dimension)
        first, second = np.triu_indices(dimension, k=1)
        pairs = np.zeros((first.shape[0], dimension))
        pair_rows = np.arange(first.shape[0])
        pairs[pair_rows, first] = 1.0
        pairs_mixed = pairs.copy()
        pairs[pair_rows, second] = 1.0
        pairs_mixed[pair_rows, second] = -1.0
        diagonals = np.sqrt(spread / 2.0) * np.concatenate([pairs, pairs_mixed], axis=0)
        offsets = np.concatenate(
            [np.zeros((1, dimension)), axes, -axes, diagonals, -diagonals], axis=0
        )
        weights = np.concatenate(
            [
                [2.0 / spread],
                np.full(2 * dimension, (4.0 - dimension) / (2.0 * spread**2)),
                np.full(4 * first.shape[0], 1.0 / spread**2),
            ]
        )
        return SigmaPoints(offsets, weights, weights)
