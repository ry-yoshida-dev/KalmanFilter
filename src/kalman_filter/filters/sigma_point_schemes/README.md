# sigma_point_schemes

## Overview

Implementations of `SigmaPointScheme` for the unscented filter. Each scheme defines points for the standard normal distribution; `SigmaPointScheme.generate` maps them to any covariance.

## Components

| Component | Points | Exact up to degree | Description |
| --------- | ------ | ------------------ | ----------- |
| [`merwe_scaled.py`](./merwe_scaled.py) | `2n + 1` | 3 | Scaled unscented transform (`MerweScaledSigmaPointScheme`) |
| [`cubature.py`](./cubature.py) | `2n` | 3 | Equally weighted spherical-radial cubature; the CKF (`CubatureSigmaPointScheme`) |
| [`fifth_degree_cubature.py`](./fifth_degree_cubature.py) | `2n^2 + 1` | 5 | Fifth-degree spherical-radial cubature (`FifthDegreeCubatureSigmaPointScheme`) |
| [`gauss_hermite.py`](./gauss_hermite.py) | `order^n` | `2 order - 1` | Tensor-product Gauss-Hermite quadrature; the GHKF, for small states (`GaussHermiteSigmaPointScheme`) |
