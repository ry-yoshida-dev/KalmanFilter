# filters

## Overview

Tests for the filters in `kalman_filter.filters`.

## Components

| Component | Description |
| --------- | ----------- |
| [`test_kalman_filters.py`](./test_kalman_filters.py) | Linear, extended, and unscented filters: accessible parameters, NaN masking, input validation, angle wrapping, and consistency |
| [`test_iterated_filters.py`](./test_iterated_filters.py) | Iterated extended and unscented filters |
| [`test_sigma_point_schemes.py`](./test_sigma_point_schemes.py) | Moments matched by each sigma point scheme |
