# KalmanFilter

## Overview

KalmanFilter (`kalman_filter`) is a Python package for generic linear Kalman filter prediction/update workflows.
It provides:

- a high-level `KalmanFilter` class for iterative filtering
- a configurable `KFParameters` dataclass to build all required matrices/vectors
- strongly typed parameter containers for state, covariance, gain, and measurement handling

For package-level details, see [src/kalman_filter/README.md](src/kalman_filter/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development:

```bash
pip install -e .
```

If you only need dependencies:

```bash
pip install -r requirements.txt
```

## Quick example

```python
from kalman_filter import KalmanFilter, KFParameters, MeasurementVector

params = KFParameters(
    state_vector_format={
        "static": ["u", "v", "s", "r"],
        "dynamic": ["u", "v", "s"],
        "acceleration": [],
    }
)

kf = KalmanFilter(params=params)

z0 = MeasurementVector.register_vec_from_list(
    values=[100, 80, 400, 1.2],
    format=["u", "v", "s", "r"],
)
kf.set_initial_measurement(z0)

pred_x, pred_P = kf.predict()
new_z = MeasurementVector.register_vec_from_list(
    values=[102, 81, 398, 1.2],
    format=["u", "v", "s", "r"],
)
upd_x, upd_P = kf.update(z=new_z)
```

## Notes

- `KalmanFilter` initializes with a zero measurement vector, so call `set_initial_measurement()` before your first update loop.
- `state_vector_format` controls variable ordering and dimensions of generated matrices.
