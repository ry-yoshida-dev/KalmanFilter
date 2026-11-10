# models

## Overview

Descriptions of the system: how the state evolves (`MotionModel`) and how sensors observe it (`MeasurementModel`). Models accept a single state `(n,)` or a batch `(k, n)` (e.g. sigma points); `residual` and `add` define state arithmetic so angular components stay wrapped.

## Components

| Component | Description |
| --------- | ----------- |
| [`motion_model.py`](./motion_model.py) | Abstract motion model `x_k = f(x_{k-1}, dt, u) + w` |
| [`motion_models/`](./motion_models/README.md) | Concrete motion models |
| [`measurement_model.py`](./measurement_model.py) | Abstract measurement model `z = h(x) + v` |
| [`measurement_models/`](./measurement_models/README.md) | Concrete measurement models |
| [`state_layout.py`](./state_layout.py) | Named variable layout of a state vector (`StateLayout`) |
| [`state_role.py`](./state_role.py) | Derivative level of a state variable: static, dynamic, acceleration, jerk (`StateRole`) |
