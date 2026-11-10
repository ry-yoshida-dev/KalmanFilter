# kalman_filter

## Overview

Package root. Models describe the system, and filters hold the estimate and every matrix of their latest step as attributes. Everything public is re-exported from [`__init__.py`](./__init__.py).

## Components

| Component | Description |
| --------- | ----------- |
| [`array_types.py`](./array_types.py) | NumPy array type aliases (`FloatArray`, `BoolArray`, `IntArray`) |
| [`_arrays.py`](./_arrays.py) | Private input conversion and validation helpers |
| [`_numerics.py`](./_numerics.py) | Private numerical routines (covariance square roots, finite-difference Jacobians, angle wrapping) |
| [`models/`](./models/README.md) | Motion and measurement models |
| [`parameters/`](./parameters/README.md) | One container per filter parameter (`StateVector`, `ErrorCovarianceMatrix`, `KalmanGain`, ...) |
| [`filters/`](./filters/README.md) | Kalman, extended, unscented, and iterated filters |
