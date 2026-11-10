# filters

## Overview

Kalman-type filters. Each filter holds its estimate `x`, `P` and the matrices of its latest step (`F`, `Q`, `H`, `R`, `y`, `S`, `K`) as attributes. `KalmanFilter` is abstract and implements the shared update and missing-data handling; its implementations define how the estimate is propagated through the models.

## Components

| Component | Description |
| --------- | ----------- |
| [`kalman_filter.py`](./kalman_filter.py) | Abstract filter (`KalmanFilter`) |
| [`kalman_filters/`](./kalman_filters/README.md) | Linear, extended, unscented, and the iterated variants |
| [`propagation.py`](./propagation.py) | Prior produced by one prediction (`Propagation`), passed from subclasses to `KalmanFilter` |
| [`projection.py`](./projection.py) | Predicted measurement as an affine model of the state (`Projection`), passed from subclasses to `KalmanFilter` |
| [`covariance_update.py`](./covariance_update.py) | Joseph or standard covariance update (`CovarianceUpdate`) |
| [`sigma_point_scheme.py`](./sigma_point_scheme.py) | Abstract sigma point placement (`SigmaPointScheme`) |
| [`sigma_point_schemes/`](./sigma_point_schemes/README.md) | Merwe-scaled, cubature (third and fifth degree), and Gauss-Hermite sigma points |
| [`sigma_points.py`](./sigma_points.py) | Weighted sigma point set (`SigmaPoints`) |

## Examples

```python
kf = UnscentedKalmanFilter(motion, sensor, x=[0.0, 0.0], P=[1.0, 1.0])
kf.predict(dt=0.1)
kf.update([0.2])
kf.x, kf.P, kf.F, kf.Q, kf.H, kf.R, kf.y, kf.S, kf.K
kf.x["x"] = 0.5  # element writes change the estimate
kf.P *= 2.0  # x and P can be replaced at any time
```
