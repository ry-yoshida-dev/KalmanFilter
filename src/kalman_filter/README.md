# kalman_filter

## Overview

`kalman_filter` is the core package for building and running a linear Kalman filter.
It separates responsibilities into:

- orchestration (`KalmanFilter`)
- initialization and validation (`KFParameters`)
- matrix/vector containers (`parameter_containers`)
- helper utilities (`utils`)

## Components

| Component | Description |
| --------- | ----------- |
| [`array_types.py`](./array_types.py) | NumPy dtype aliases (`FloatArray`, `FloatWeights`, …) |
| [`kalman_filter.py`](./kalman_filter.py) | Main predict/update workflow class (`KalmanFilter`) |
| [`parameters.py`](./parameters.py) | Parameter dataclass (`KFParameters`) that builds internal matrices and vectors |
| [`parameter_containers/`](./parameter_containers/README.md) | Typed container classes for state, covariance, gains, and matrix operations |
| [`utils/`](./utils/README.md) | Utility modules (including data creation helpers) |

## Public API

Main imports:

- `KalmanFilter`
- `KFParameters`
- matrix/vector container types from `parameter_containers`

See [`__init__.py`](./__init__.py) for the full export list.

## Typical flow

1. Build `KFParameters` with your `state_vector_format`, covariance diagonals, and timing settings.
2. Create `KalmanFilter(params=...)`.
3. Call `set_initial_measurement()` once with an initial `MeasurementVector`.
4. Loop:
   - `predict(...)`
   - `update(z=...)`

## Validation behavior

`KFParameters` validates:

- positivity of `R_values`, `P_values`, `Q_values`, `dt`, `alpha`
- consistency of `state_vector_format`
- shape compatibility across `R`, `P`, `Q`, `H`, `F`, `K`, and `S`
