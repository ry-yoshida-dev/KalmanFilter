"""Abstract sigma point selection scheme."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .._numerics import covariance_square_root
from ..array_types import FloatArray
from .sigma_points import SigmaPoints


class SigmaPointScheme(ABC):
    """
    Rule for placing and weighting sigma points around a Gaussian.

    A scheme defines its points for the standard normal distribution; they are
    mapped to any covariance ``P`` with a square root ``L L^T = P``.
    """

    def generate(self, covariance: FloatArray) -> SigmaPoints:
        """
        Sigma points for a Gaussian with ``covariance``, as offsets from its mean.

        Parameters
        ----------
        covariance : FloatArray
            Covariance with shape ``(n, n)``.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the covariance is not positive semi-definite.
        ValueError
            If the scheme is not defined for the dimension of ``covariance``.
        """
        square_root = covariance_square_root(covariance)
        standard = self.standard_points(covariance.shape[0])
        return SigmaPoints(
            offsets=standard.offsets @ square_root.T,
            mean_weights=standard.mean_weights,
            covariance_weights=standard.covariance_weights,
        )

    @abstractmethod
    def standard_points(self, dimension: int) -> SigmaPoints:
        """
        Sigma points of the ``dimension``-variate standard normal distribution.

        Raises
        ------
        ValueError
            If the scheme is not defined for ``dimension``.
        """
