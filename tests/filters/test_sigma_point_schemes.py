"""Tests for the sigma point schemes."""

from __future__ import annotations

import numpy as np
import pytest

from kalman_filter import (
    CubatureSigmaPointScheme,
    FifthDegreeCubatureSigmaPointScheme,
    GaussHermiteSigmaPointScheme,
    MerweScaledSigmaPointScheme,
    SigmaPointScheme,
)

ALL_SCHEMES: list[SigmaPointScheme] = [
    MerweScaledSigmaPointScheme(),
    MerweScaledSigmaPointScheme(0.3, 2.0, 1.0),
    CubatureSigmaPointScheme(),
    FifthDegreeCubatureSigmaPointScheme(),
    GaussHermiteSigmaPointScheme(),
    GaussHermiteSigmaPointScheme(order=5),
]
FIFTH_DEGREE_SCHEMES: list[SigmaPointScheme] = [
    FifthDegreeCubatureSigmaPointScheme(),
    GaussHermiteSigmaPointScheme(),
]


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
@pytest.mark.parametrize("dimension", [1, 3, 6])
def test_schemes_match_mean_and_covariance(scheme: SigmaPointScheme, dimension: int) -> None:
    rng = np.random.default_rng(dimension)
    factor = rng.normal(size=(dimension, dimension))
    covariance = factor @ factor.T + np.eye(dimension)
    points = scheme.generate(covariance)
    assert np.sum(points.mean_weights) == pytest.approx(1.0)
    assert np.allclose(points.mean_weights @ points.offsets, 0.0, atol=1e-12)
    weighted = points.offsets * points.covariance_weights[:, None]
    assert np.allclose(weighted.T @ points.offsets, covariance)


@pytest.mark.parametrize("scheme", FIFTH_DEGREE_SCHEMES)
@pytest.mark.parametrize("dimension", [2, 3, 5])
def test_fifth_degree_schemes_match_fourth_moments(
    scheme: SigmaPointScheme, dimension: int
) -> None:
    points = scheme.standard_points(dimension)
    offsets, weights = points.offsets, points.mean_weights
    assert weights @ offsets[:, 0] ** 4 == pytest.approx(3.0)
    assert weights @ (offsets[:, 0] ** 2 * offsets[:, 1] ** 2) == pytest.approx(1.0)
    assert weights @ (offsets[:, 0] ** 3 * offsets[:, 1]) == pytest.approx(0.0, abs=1e-12)
    assert weights @ offsets[:, 0] ** 5 == pytest.approx(0.0, abs=1e-12)


def test_scheme_point_counts() -> None:
    assert FifthDegreeCubatureSigmaPointScheme().standard_points(4).offsets.shape == (33, 4)
    assert GaussHermiteSigmaPointScheme(order=3).standard_points(4).offsets.shape == (81, 4)


def test_gauss_hermite_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError, match="order"):
        GaussHermiteSigmaPointScheme(order=0)
    with pytest.raises(ValueError, match="too many"):
        GaussHermiteSigmaPointScheme(order=3).standard_points(20)


def test_semidefinite_covariance_is_supported() -> None:
    covariance = np.diag([1.0, 0.0, 2.0])
    points = FifthDegreeCubatureSigmaPointScheme().generate(covariance)
    assert np.allclose(points.offsets[:, 1], 0.0)
