# measurement_models

## Overview

Implementations of `MeasurementModel`.

## Components

| Component | Description |
| --------- | ----------- |
| [`linear.py`](./linear.py) | `z = H x + v`, including direct observation of named states (`LinearMeasurementModel`) |
| [`range_bearing.py`](./range_bearing.py) | Polar range and bearing of a planar position (`RangeBearingMeasurementModel`) |
| [`function.py`](./function.py) | Model built from user callables (`FunctionMeasurementModel`) |
