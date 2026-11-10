"""Private numerical routines shared by models and filters."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .array_types import FloatArray

# Balances truncation (O(h^2)) and rounding (O(eps / h)) error of central differences.
_FINITE_DIFFERENCE_STEP: float = float(np.cbrt(np.finfo(np.float64).eps))
_SEMIDEFINITE_TOLERANCE: float = float(np.sqrt(np.finfo(np.float64).eps))


def symmetrize(matrix: FloatArray) -> FloatArray:
    """Return the symmetric part of one ``(n, n)`` or stacked ``(..., n, n)`` square matrix."""
    return 0.5 * (matrix + np.swapaxes(matrix, -1, -2))


def covariance_square_root(covariance: FloatArray) -> FloatArray:
    """
    Factor ``L`` with ``L @ L.T == covariance``.

    Cholesky is used when the matrix is positive definite. Otherwise an
    eigendecomposition gives a square root of the positive semi-definite part,
    so covariances with exactly known components (zero variance) still work.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix has an eigenvalue clearly below zero.
    """
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        pass
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = float(np.max(np.abs(eigenvalues), initial=0.0))
    if float(eigenvalues[0]) < -_SEMIDEFINITE_TOLERANCE * max(scale, 1.0):
        raise np.linalg.LinAlgError("covariance is not positive semi-definite")
    root: FloatArray = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    return root


def difference(values: FloatArray, reference: FloatArray) -> FloatArray:
    """Plain ``values - reference``; the default residual for vector spaces."""
    return values - reference


def addition(values: FloatArray, increments: FloatArray) -> FloatArray:
    """Plain ``values + increments``; the default state addition for vector spaces."""
    return values + increments


def wrap_angle(angle: FloatArray) -> FloatArray:
    """Wrap angles in radians into ``[-pi, pi)``."""
    return np.mod(angle + np.pi, 2.0 * np.pi) - np.pi


def circular_mean(angles: FloatArray, weights: FloatArray) -> FloatArray:
    """Weighted circular mean of angles along the first axis."""
    sine: FloatArray = weights @ np.sin(angles)
    cosine: FloatArray = weights @ np.cos(angles)
    return np.arctan2(sine, cosine)


def numerical_jacobian(
    function: Callable[[FloatArray], FloatArray],
    point: FloatArray,
    add: Callable[[FloatArray, FloatArray], FloatArray],
    difference: Callable[[FloatArray, FloatArray], FloatArray],
) -> FloatArray:
    """
    Central-difference Jacobian of a batch-capable function.

    ``function`` must map ``(k, n)`` inputs to ``(k, m)`` outputs, so all
    ``2n`` perturbed points are evaluated in a single call. Inputs are perturbed
    with ``add`` and outputs compared with ``difference``, so the Jacobian is
    taken with respect to the state arithmetic of the model (wrapped angles,
    rotations) rather than plain vector arithmetic.
    """
    dimension = point.shape[0]
    steps = _FINITE_DIFFERENCE_STEP * np.maximum(1.0, np.abs(point))
    offsets = np.diag(steps)
    increments = np.concatenate([offsets, -offsets], axis=0)
    perturbed = add(np.tile(point, (2 * dimension, 1)), increments)
    values = function(perturbed)
    forward = values[:dimension]
    backward = values[dimension:]
    jacobian: FloatArray = (difference(forward, backward) / (2.0 * steps[:, None])).T
    return jacobian
