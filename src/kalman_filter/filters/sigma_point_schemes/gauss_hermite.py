"""Tensor-product Gauss-Hermite quadrature points."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import override

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from ...array_types import FloatArray
from ..sigma_point_scheme import SigmaPointScheme
from ..sigma_points import SigmaPoints

_MAXIMUM_POINT_COUNT: int = 1_000_000


@dataclass(frozen=True)
class GaussHermiteSigmaPointScheme(SigmaPointScheme):
    """
    ``order^n`` points from the product of one-dimensional Gauss-Hermite rules.

    The rule integrates polynomials up to degree ``2 * order - 1`` exactly, so it
    is the most accurate scheme here for strongly nonlinear, low-dimensional
    models, and its weights are all positive. The point count grows
    exponentially with the state dimension, which limits it to small states
    (e.g. ``order=3`` gives 81 points for ``n = 4``). Used in
    :class:`UnscentedKalmanFilter` it gives the Gauss-Hermite Kalman filter.

    Attributes
    ----------
    order : int
        Points per dimension; ``3`` matches the unscented transform's accuracy
        on cubic terms and adds exactness up to degree 5.

    Raises
    ------
    ValueError
        If ``order`` is smaller than 1.
    """

    order: int = 3

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError(f"order must be at least 1, got {self.order}")

    @override
    def standard_points(self, dimension: int) -> SigmaPoints:
        """
        Return the ``order^n`` product-rule points.

        Raises
        ------
        ValueError
            If ``order^n`` exceeds one million points.
        """
        if self.order**dimension > _MAXIMUM_POINT_COUNT:
            raise ValueError(
                f"order^n = {self.order}^{dimension} points is too many; "
                + "use a cubature or unscented scheme for this dimension"
            )
        offsets, weights = _product_rule(self.order, dimension)
        return SigmaPoints(offsets, weights, weights)


@cache
def _product_rule(order: int, dimension: int) -> tuple[FloatArray, FloatArray]:
    """Cached product grid and weights for the standard normal distribution."""
    nodes, node_weights = hermegauss(order)
    node_weights = node_weights / np.sum(node_weights)
    grid = np.meshgrid(*([nodes] * dimension), indexing="ij")
    weight_grid = np.meshgrid(*([node_weights] * dimension), indexing="ij")
    offsets = np.stack([axis.reshape(-1) for axis in grid], axis=-1)
    weights = np.prod(np.stack([axis.reshape(-1) for axis in weight_grid], axis=-1), axis=-1)
    offsets.flags.writeable = False
    weights.flags.writeable = False
    return offsets, weights
