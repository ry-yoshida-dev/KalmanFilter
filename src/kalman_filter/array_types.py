"""NumPy array type aliases used across the package."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type IntArray = NDArray[np.intp]
