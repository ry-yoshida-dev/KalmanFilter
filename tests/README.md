# tests

## Overview

Pytest suite for the `kalman_filter` package. Covers float array typing, parameter validation, container behavior, the predict/update workflow, and synthetic data generation.

## Components

| Component | Description |
| --------- | ----------- |
| [`conftest.py`](./conftest.py) | Shared pytest fixtures |
| [`helpers.py`](./helpers.py) | Test helpers (`make_measurement_vector`) |
| [`test_array_types.py`](./test_array_types.py) | `FloatArray` alias and float covariance builders |
| [`test_kf_parameters.py`](./test_kf_parameters.py) | `KFParameters` validation and initialization |
| [`test_kalman_filter_workflow.py`](./test_kalman_filter_workflow.py) | `KalmanFilter` predict/update and clone behavior |
| [`test_measurement_containers.py`](./test_measurement_containers.py) | `StateVector`, `MeasurementVector`, and `MeasurementMatrix` |
| [`test_sample_data_creator.py`](./test_sample_data_creator.py) | `SampleDataCreator` output shapes and iteration |

## Examples

Run all tests from the package root:

```bash
uv run pytest
```

Run a single module:

```bash
uv run pytest tests/test_kalman_filter_workflow.py
```
