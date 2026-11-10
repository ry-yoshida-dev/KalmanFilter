# models

## Overview

Tests for the motion and measurement models in `kalman_filter.models`.

## Components

| Component | Description |
| --------- | ----------- |
| [`test_motion_models.py`](./test_motion_models.py) | Kinematic, linear, CTRV, and function-defined motion models, including Jacobians |
| [`test_measurement_models.py`](./test_measurement_models.py) | Linear, range-bearing, and function-defined measurement models, including Jacobians and angle wrapping |
| [`test_continuous_time.py`](./test_continuous_time.py) | Continuous-time (SDE) motion model |
