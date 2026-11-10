# validation

## Overview

Cross-cutting checks of the filters. The references are written from the textbook equations with plain NumPy and SciPy, so a mistake shared by several parts of the library cannot cancel out.

## Components

| Component | Description |
| --------- | ----------- |
| [`test_reference_solutions.py`](./test_reference_solutions.py) | Exact conditioning of the joint Gaussian of a whole run, the Riccati steady state, and textbook EKF/UKF equations |
| [`test_numerical_conditioning.py`](./test_numerical_conditioning.py) | Precise measurements on vague priors, long runs with tiny noise, and large coordinates |
