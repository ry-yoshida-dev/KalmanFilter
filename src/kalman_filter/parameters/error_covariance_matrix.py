"""Error covariance matrix ``P``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..array_types import FloatArray
from ..models import StateLayout
from .parameter import KalmanFilterParameter


@dataclass(eq=False, repr=False)
class ErrorCovarianceMatrix(KalmanFilterParameter):
    """
    Covariance ``P`` of the state estimate with access by variable names or indices.

    Writing an element changes the covariance of the filter that returned it;
    the symmetric element is written too.

    Attributes
    ----------
    value : FloatArray
        Covariance with shape ``(n, n)``.
    layout : StateLayout
        Variable names of the state.

    Raises
    ------
    ValueError
        If ``value`` is not ``(n, n)`` for the ``n`` variables of ``layout``.

    Examples
    --------
    >>> kf.P["x", "x"]
    >>> kf.P[0, 0]
    >>> kf.P["x", "x_velocity"] = 0.0
    >>> kf.P.standard_deviation("x")
    """

    layout: StateLayout

    def __post_init__(self) -> None:
        dimension = self.layout.dimension
        if self.value.shape != (dimension, dimension):
            raise ValueError(
                f"covariance must have shape ({dimension}, {dimension}), got {self.value.shape}"
            )

    def __getitem__(self, key: tuple[str | int, str | int]) -> float:
        """
        Covariance of two variables, given by name or index.

        Raises
        ------
        KeyError
            If ``key`` is not a pair, or a name is not part of the layout.
        IndexError
            If an index is out of range.
        """
        row, column = self._indices(key)
        return float(self.value[row, column])

    def __setitem__(self, key: tuple[str | int, str | int], value: float) -> None:
        """
        Overwrite the covariance of two variables and its symmetric element.

        Raises
        ------
        KeyError
            If a name is not part of the layout.
        IndexError
            If an index is out of range.
        ValueError
            If ``value`` is not finite, or is a negative variance.
        """
        row, column = self._indices(key)
        if not np.isfinite(value):
            raise ValueError(f"covariance must be finite, got {value}")
        if row == column and value < 0.0:
            raise ValueError(f"variance must be non-negative, got {value}")
        self.value[row, column] = value
        self.value[column, row] = value

    def _indices(self, key: tuple[str | int, ...] | str | int) -> tuple[int, int]:
        """Row and column of a pair of names or indices, with negative indices resolved."""
        # The public methods are annotated with pairs only; this guards untyped callers.
        if isinstance(key, str | int) or len(key) != 2:
            raise KeyError(f"P is indexed by a pair of names or indices, got {key!r}")
        positions = range(self.layout.dimension)
        row, column = (
            self.layout.index(item) if isinstance(item, str) else positions[item] for item in key
        )
        return row, column

    def standard_deviation(self, name: str) -> float:
        """
        Standard deviation of one variable.

        Raises
        ------
        KeyError
            If ``name`` is not part of the layout.
        """
        index = self.layout.index(name)
        return float(np.sqrt(self.value[index, index]))

    @property
    def standard_deviations(self) -> FloatArray:
        """Standard deviations of all variables with shape ``(n,)``."""
        result: FloatArray = np.sqrt(np.diag(self.value))
        return result
