"""Predicted measurement of a state estimate."""

from __future__ import annotations

from dataclasses import dataclass

from ..array_types import FloatArray


@dataclass(frozen=True)
class Projection:
    """
    Predicted measurement of the prior, expressed as an affine model of the state.

    Attributes
    ----------
    z : FloatArray
        Expected measurement with shape ``(m,)``.
    S : FloatArray
        Innovation covariance with shape ``(m, m)``.
    cross_covariance : FloatArray
        State-measurement cross covariance ``P H^T`` with shape ``(n, m)``.
    H : FloatArray
        Measurement matrix with shape ``(m, n)``: the Jacobian of the
        measurement model, or its statistical linearization for sigma point
        filters.
    R : FloatArray
        Measurement noise covariance of the sensor with shape ``(m, m)``.
    """

    z: FloatArray
    S: FloatArray
    cross_covariance: FloatArray
    H: FloatArray
    R: FloatArray
