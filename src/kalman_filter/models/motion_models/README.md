# motion_models

## Overview

Implementations of `MotionModel`.

## Components

| Component | Description |
| --------- | ----------- |
| [`linear.py`](./linear.py) | Constant matrices `F`, `Q`, and optional `B` (`LinearMotionModel`) |
| [`kinematic.py`](./kinematic.py) | Random walk / constant velocity / acceleration / jerk per variable with exact `F(dt)` and `Q(dt)` (`KinematicMotionModel`) |
| [`constant_turn_rate_velocity.py`](./constant_turn_rate_velocity.py) | Nonlinear planar CTRV model with analytic Jacobian (`ConstantTurnRateVelocityMotionModel`) |
| [`function.py`](./function.py) | Model built from user callables (`FunctionMotionModel`) |
| [`continuous_time.py`](./continuous_time.py) | Stochastic differential equation `dx/dt = a(x, u) + L w` integrated with Runge-Kutta, noise discretized with Van Loan (`ContinuousTimeMotionModel`) |
