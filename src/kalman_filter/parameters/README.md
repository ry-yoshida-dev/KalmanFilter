# parameters

## Overview

One container per Kalman filter parameter. Each holds the raw array in `value` and behaves like it in NumPy functions and operators, which return plain arrays. The state and its covariance also give access by variable name, and writing to them changes the filter.

## Components

| Component | Description |
| --------- | ----------- |
| [`parameter.py`](./parameter.py) | Abstract container (`KalmanFilterParameter`) |
| [`state_vector.py`](./state_vector.py) | State `x`, by role (`x[StateRole.STATIC, "u"]`, `x[StateRole.DYNAMIC]`) or name (`StateVector`) |
| [`error_covariance_matrix.py`](./error_covariance_matrix.py) | Covariance `P`, by pairs of variable names (`ErrorCovarianceMatrix`) |
| [`state_transition_matrix.py`](./state_transition_matrix.py) | Transition matrix `F` (`StateTransitionMatrix`) |
| [`process_noise_covariance.py`](./process_noise_covariance.py) | Process noise `Q` (`ProcessNoiseCovariance`) |
| [`measurement_matrix.py`](./measurement_matrix.py) | Measurement matrix `H` (`MeasurementMatrix`) |
| [`measurement_noise_covariance.py`](./measurement_noise_covariance.py) | Measurement noise `R` (`MeasurementNoiseCovariance`) |
| [`innovation.py`](./innovation.py) | Innovation `y` (`Innovation`) |
| [`innovation_covariance_matrix.py`](./innovation_covariance_matrix.py) | Innovation covariance `S` (`InnovationCovarianceMatrix`) |
| [`kalman_gain.py`](./kalman_gain.py) | Kalman gain `K` (`KalmanGain`) |

## Examples

```python
kf.x[StateRole.STATIC, "u"]
kf.x[StateRole.DYNAMIC] = 0.0
kf.P["x", "x"] = 4.0
kf.K.value, kf.F @ kf.P, kf.x - truth
```
