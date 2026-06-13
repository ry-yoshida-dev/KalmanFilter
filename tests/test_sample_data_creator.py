"""Tests for synthetic data generation utilities."""

from __future__ import annotations

import numpy as np

from kalman_filter.utils.data_creator.data_creator import SampleDataCreator
from kalman_filter.utils.data_creator.data_type import DataType


def test_linear_data_has_expected_shape() -> None:
    """Linear synthetic data must match (steps, ndim)."""
    creator = SampleDataCreator(ndim=3, steps=50, sigma=0.1, dt=0.2, seed=42)
    data = creator.create(DataType.LINEAR)

    assert data.time.shape == (50,)
    assert data.ground_truth.shape == (50, 3)
    assert data.measurements.shape == (50, 3)
    assert np.issubdtype(data.ground_truth.dtype, np.floating)
    assert np.issubdtype(data.measurements.dtype, np.floating)


def test_linear_ground_truth_increases_with_time() -> None:
    """Linear ground truth should grow monotonically along each dimension."""
    creator = SampleDataCreator(
        ndim=2,
        steps=20,
        sigma=0.0,
        dt=1.0,
        initial_value=0.0,
        seed=0,
    )
    data = creator.create(DataType.LINEAR)

    assert np.all(np.diff(data.ground_truth[:, 0]) > 0.0)
    assert np.all(np.diff(data.ground_truth[:, 1]) > 0.0)


def test_iterator_yields_step_rows() -> None:
    """KalmanTestData iteration yields scalar time and 1D float rows."""
    creator = SampleDataCreator(ndim=2, steps=5, sigma=0.1, dt=1.0, seed=1)
    data = creator.create(DataType.SIN_WAVE)

    rows = list(data)
    assert len(rows) == 5
    time_value, ground_truth_row, measurement_row = rows[0]
    assert isinstance(time_value, float)
    assert ground_truth_row.shape == (2,)
    assert measurement_row.shape == (2,)
