"""Private helpers that convert user input into validated float arrays."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .array_types import FloatArray

_SYMMETRY_TOLERANCE: float = 1e-9
_SEMIDEFINITE_TOLERANCE: float = float(np.sqrt(np.finfo(np.float64).eps))


def as_vector(value: ArrayLike, name: str, dimension: int | None = None) -> FloatArray:
    """
    Convert ``value`` into a 1-D float64 copy and check its length.

    Scalars, 1-D arrays, and row or column vectors are accepted.

    Raises
    ------
    ValueError
        If ``value`` is a matrix with more than one row and column, or has the wrong length.
    """
    array = np.array(value, dtype=np.float64)
    if sum(1 for size in array.shape if size != 1) > 1:
        raise ValueError(f"{name} must be a vector, got shape {array.shape}")
    vector = array.reshape(-1)
    if dimension is not None and vector.shape[0] != dimension:
        raise ValueError(f"{name} must have length {dimension}, got {vector.shape[0]}")
    return vector


def as_matrix(value: ArrayLike, name: str, shape: tuple[int, int] | None = None) -> FloatArray:
    """Convert ``value`` into a 2-D float64 copy and check its shape."""
    matrix = np.array(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {matrix.ndim}-D")
    if shape is not None and matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
    return matrix


def as_covariance(value: ArrayLike, name: str, dimension: int | None = None) -> FloatArray:
    """
    Convert ``value`` into a symmetric covariance matrix.

    A 1-D input is interpreted as the diagonal (variances).
    """
    array = np.array(value, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    matrix: FloatArray = np.diag(array) if array.ndim == 1 else array
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix or a 1-D variance vector")
    if dimension is not None and matrix.shape[0] != dimension:
        raise ValueError(f"{name} must have dimension {dimension}, got {matrix.shape[0]}")
    check_covariance_values(matrix, name)
    return matrix


def check_covariance_values(matrix: FloatArray, name: str) -> None:
    """
    Check an ``(n, n)`` covariance for finiteness, symmetry, and positive semi-definiteness.

    Raises
    ------
    ValueError
        If a value is not finite, the matrix is not symmetric, a variance is
        negative, or an eigenvalue is clearly below zero.
    """
    magnitude = float(np.max(np.abs(matrix), initial=0.0))
    if not np.isfinite(magnitude):
        raise ValueError(f"{name} must contain only finite values")
    asymmetry = float(np.max(np.abs(matrix - matrix.T), initial=0.0))
    if asymmetry > _SYMMETRY_TOLERANCE * max(magnitude, 1e-3):
        raise ValueError(f"{name} must be symmetric")
    if np.any(np.diag(matrix) < 0.0):
        raise ValueError(f"{name} must have non-negative variances on its diagonal")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(eigenvalues[0]) < -_SEMIDEFINITE_TOLERANCE * float(np.max(np.abs(eigenvalues))):
        raise ValueError(
            f"{name} must be positive semi-definite, got eigenvalue {float(eigenvalues[0])}"
        )


def as_returned(
    value: ArrayLike, name: str, shape: tuple[int, ...], is_batch: bool = False
) -> FloatArray:
    """
    Convert the return value of a user-supplied function into a float array of ``shape``.

    ``is_batch`` tells that the function was given a batch of rows, so the
    error message suggests ``is_vectorized=False``, the usual cause.

    Raises
    ------
    ValueError
        If the value has another shape.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        hint = (
            "; if it does not accept a batch of rows, pass is_vectorized=False" if is_batch else ""
        )
        raise ValueError(f"{name} returned shape {array.shape}, expected {shape}{hint}")
    return array


def read_only(array: FloatArray) -> FloatArray:
    """Mark ``array`` as read-only and return it."""
    array.flags.writeable = False
    return array
