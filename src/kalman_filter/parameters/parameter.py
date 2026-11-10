"""Abstract container of one Kalman filter parameter."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import cast, override

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.typing import DTypeLike

from ..array_types import FloatArray


@dataclass(eq=False, repr=False)
class KalmanFilterParameter(NDArrayOperatorsMixin, ABC):
    """
    One named quantity of a Kalman filter (``x``, ``P``, ``F``, ...).

    The raw array is :attr:`value`. The container also behaves like that array
    in NumPy functions (``np.trace(kf.P)``) and in arithmetic, comparison, and
    matrix products (``kf.F @ kf.P``, ``2 * kf.P``, ``kf.x - truth``), which
    return plain arrays. In-place operators on a filter's attribute
    (``kf.x += dx``) therefore replace the attribute through the filter.

    Attributes
    ----------
    value : FloatArray
        The array.
    """

    value: FloatArray

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of :attr:`value`."""
        return self.value.shape

    @property
    def T(self) -> FloatArray:  # noqa: N802
        """Transpose of :attr:`value`."""
        return self.value.T

    def __len__(self) -> int:
        """Length of the first axis of :attr:`value`."""
        return len(self.value)

    def __array__(self, dtype: DTypeLike | None = None, copy: bool | None = None) -> FloatArray:
        array: FloatArray = np.asarray(self.value, dtype=dtype)
        return array.copy() if copy else array

    @override
    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
    ) -> object:
        """Apply NumPy ufuncs, and so the operators, to :attr:`value`."""
        arrays = [_unwrap(item) for item in inputs]
        outputs = kwargs.get("out")
        if isinstance(outputs, tuple) and any(
            isinstance(item, KalmanFilterParameter) for item in cast(tuple[object, ...], outputs)
        ):
            # In-place operators (``kf.x += dx``) return a new array instead, so the
            # filter's setter validates it before it replaces the estimate.
            del kwargs["out"]
        result: object = getattr(ufunc, method)(*arrays, **kwargs)
        return result

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"


def _unwrap(item: object) -> object:
    """Raw array of a parameter; anything else unchanged."""
    return item.value if isinstance(item, KalmanFilterParameter) else item
