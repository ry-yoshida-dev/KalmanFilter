# kalman_filters

## Overview

Implementations of `KalmanFilter`. The iterated variants keep the prediction of their base filter and relinearize the measurement update around the posterior, which pays off for precise, strongly nonlinear sensors.

## Components

| Component | Description |
| --------- | ----------- |
| [`linear.py`](./linear.py) | Standard Kalman filter; accepts only linear models (`LinearKalmanFilter`) |
| [`extended.py`](./extended.py) | Jacobian linearization (`ExtendedKalmanFilter`) |
| [`unscented.py`](./unscented.py) | Sigma point propagation; the cubature or Gauss-Hermite Kalman filter with the matching `SigmaPointScheme` (`UnscentedKalmanFilter`) |
| [`iterated_extended.py`](./iterated_extended.py) | Gauss-Newton update that relinearizes at the posterior (IEKF) (`IteratedExtendedKalmanFilter`) |
| [`iterated_unscented.py`](./iterated_unscented.py) | Iterated posterior linearization by statistical linear regression (IPLF) (`IteratedUnscentedKalmanFilter`) |
