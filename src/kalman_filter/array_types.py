"""NumPy array type aliases for Kalman filter parameters."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatDType: TypeAlias = np.floating[Any]
FloatArray: TypeAlias = NDArray[FloatDType]
FloatWeights: TypeAlias = list[float] | FloatArray
FloatMatrixInput: TypeAlias = FloatArray | list[float] | None
