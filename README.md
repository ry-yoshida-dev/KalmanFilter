# KalmanFilter

## Overview

KalmanFilter (`kalman_filter`) is a typed Python package for running Kalman filters.
Models describe how the state moves and how sensors see it. Filters hold the estimate (`x`, `P`), which you can read, edit, or replace at any time, and expose the matrices of their latest step (`F`, `Q`, `H`, `R`, `y`, `S`, `K`) as attributes.

| Category | Contents |
| -------- | -------- |
| [Filters](src/kalman_filter/filters/README.md) | Kalman filter (`LinearKalmanFilter`), extended KF, iterated EKF, unscented KF and iterated UKF with Merwe-scaled, cubature (third or fifth degree), or Gauss-Hermite sigma points; Joseph or standard covariance update, fading memory, NaN-masked partial measurements |
| [Models](src/kalman_filter/models/README.md) | Linear, kinematic (constant velocity / acceleration / jerk with exact `F(dt)`, `Q(dt)`), CTRV, continuous-time SDE, and function-defined motion; linear, range-bearing, and function-defined measurements |

## Installation

```bash
uv add git+https://github.com/ry-yoshida-dev/KalmanFilter
```

or, from a clone, `uv sync`.

## Usage

Track a point that moves at roughly constant velocity and is observed in position:

```python
from kalman_filter import (
    KinematicMotionModel,
    LinearKalmanFilter,
    LinearMeasurementModel,
    StateRole,
)

motion = KinematicMotionModel.from_state_roles(
    {StateRole.STATIC: ["u", "v"], StateRole.DYNAMIC: ["u", "v"]}, noise_intensities=0.5
)  # state [u, v, u_velocity, v_velocity]
sensor = LinearMeasurementModel.from_observed_states(motion.layout, ["u", "v"], [0.25, 0.25])
kf = LinearKalmanFilter(motion, sensor, x=[0.0, 0.0, 0.0, 0.0], P=[1.0, 1.0, 10.0, 10.0])

for z in ([0.1, 0.0], [0.2, 0.1], [float("nan"), 0.2]):  # NaN = not observed
    kf.predict(dt=0.1)
    kf.update(z)

kf.x[StateRole.DYNAMIC, "u"], kf.x["u_velocity"], kf.x[2]  # the same element
kf.x[StateRole.STATIC] = [0.0, 0.0]  # writes change the estimate
kf.P["u", "u"]  # covariance by variable names
kf.F, kf.Q  # transition matrix and process noise of the latest prediction
kf.H, kf.R, kf.y, kf.S, kf.K  # matrices of the latest update
kf.K.value  # raw NumPy array of any of them
```

For nonlinear models, use `ExtendedKalmanFilter` or `UnscentedKalmanFilter` (or their iterated variants) in the same way.
