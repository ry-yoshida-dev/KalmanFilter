# tests

## Overview

Pytest suite, laid out like `src/kalman_filter`: one directory per package. Besides unit behaviour it checks the mathematics: closed-form results, agreement between filters that must coincide, NEES/NIS consistency on simulated data, and, in [`validation/`](./validation/README.md), agreement with references derived without the library.

## Components

| Component | Description |
| --------- | ----------- |
| [`conftest.py`](./conftest.py) | Shared constant-velocity models, filter, and trajectory sampler |
| [`filters/`](./filters/README.md) | Kalman filters, their covariance forms, and sigma points |
| [`models/`](./models/README.md) | Motion and measurement models |
| [`parameters/`](./parameters/README.md) | Parameter containers and access to the state |
| [`validation/`](./validation/README.md) | Independent reference solutions and ill-conditioned problems |

## Examples

```bash
uv run pytest
uv run pytest tests/validation
uv run mypy && uv run basedpyright
```
