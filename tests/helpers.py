"""Test helpers for Kalman filter tests."""

from __future__ import annotations

import numpy as np

from kalman_filter import MeasurementVector


def make_measurement_vector(
    values: list[float],
    format: list[str] | None = None,
) -> MeasurementVector:
    """
    Build a measurement vector from scalar values.

    Parameters
    ----------
    values : list[float]
        Measurement entries in column order.
    format : list[str] | None
        Variable names. Defaults to ``["u", "v", "s", "r"]`` when omitted.

    Returns
    -------
    MeasurementVector
        Measurement vector with shape ``(len(values), 1)``.
    """
    names = format if format is not None else ["u", "v", "s", "r"]
    vector = MeasurementVector.register_zero_vec_from_format(format=names)
    vector.value = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return vector
