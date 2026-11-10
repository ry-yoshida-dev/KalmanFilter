"""Weighted sigma point set."""

from __future__ import annotations

from dataclasses import dataclass

from ..array_types import FloatArray


@dataclass(frozen=True)
class SigmaPoints:
    """
    Deterministic samples that match a Gaussian's mean and covariance.

    The points are stored as offsets from the mean. Filters place them with the
    motion model's :meth:`MotionModel.add`, so states with their own arithmetic
    (wrapped angles, rotations) receive the same perturbations as the covariance
    describes.

    Attributes
    ----------
    offsets : FloatArray
        Deviations from the mean with shape ``(k, n)``.
    mean_weights : FloatArray
        Weights for recovering the mean, shape ``(k,)``.
    covariance_weights : FloatArray
        Weights for recovering the covariance, shape ``(k,)``.
    """

    offsets: FloatArray
    mean_weights: FloatArray
    covariance_weights: FloatArray
