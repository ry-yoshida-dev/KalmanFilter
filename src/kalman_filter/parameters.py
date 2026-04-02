from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, replace

from .parameter_containers import (
    StateVector,
    MeasurementVector,
    MeasurementNoiseCovariance,
    ErrorCovarianceMatrix,
    ProcessNoiseCovariance,
    MeasurementMatrix,
    StateTransitionMatrix,
    KalmanGain,
    InnovationCovarianceMatrix,
    )

@dataclass
class KFParameters:
    """
    Parameters for Kalman filter.

    Attributes:
    ----------
    state_vector_format: dict[str, list[str]]
        Format of the state vector.
        Key is the state role(str), value is the list of variable names.
        NOTE: Instead of enum class, str is used in State role because using DataclassInitializer to build the parameters.
    R_values: list[int|float]
        Diagonal values of the measurement noise covariance matrix R.
    P_values: list[int|float]
        Diagonal values of the error covariance matrix P.
    Q_values: list[int|float]
        Diagonal values of the process noise covariance matrix Q.
    measurement_vector_format: list[str] | None
        Format of the measurement vector.
        If None, state_vector_format["static"] will be used as default.
    measurement_matrix_value: np.ndarray | list[float] | None
        Value of the measurement matrix.
        If None, a rectangular identity matrix (np.eye(dim_z, dim_x)) will be used.
        When measurement_vector_format is None, this creates a square identity matrix
        that directly maps static components to measurement vector.
    dt: float
        Time step.
    alpha: float
        Alpha value used for tuning the error covariance matrix.
    """
    state_vector_format: dict[str, list[str]] = field(default_factory=lambda: {
        "static": ["u", "v", "s", "r"], 
        "dynamic": ["u", "v", "s"], 
        "acceleration": []
        })
    R_values: list[int|float] = field(default_factory=lambda: [1, 1, 10, 10])
    P_values: list[int|float] = field(default_factory=lambda: [10, 10, 10, 10, 10000, 10000, 10000])
    Q_values: list[int|float] = field(default_factory=lambda: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01*0.01])
    measurement_vector_format: list[str] | None = field(default_factory=lambda: None)
    measurement_matrix_value: np.ndarray | list[float] | None = field(default_factory=lambda: None)
    dt: float = 1.0
    alpha: float = 1.0

    def __post_init__(self):
        self._validate_parameters()
        self._validate_state_vector_format()
        self._initialize_variables()

    def _validate_parameters(self) -> None:
        if np.min(self.R_values) <= 0:
            raise ValueError(f"R_values must be greater than 0: {self.R_values}")
        if np.min(self.P_values) <= 0:
            raise ValueError(f"P_values must be greater than 0: {self.P_values}")
        if np.min(self.Q_values) <= 0:
            raise ValueError(f"Q_values must be greater than 0: {self.Q_values}")
        if self.dt <= 0:
            raise ValueError(f"dt must be greater than 0: {self.dt}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be greater than 0: {self.alpha}")
        # Early validation: R_values length should match measurement vector dimension
        z_format = self.measurement_vector_format if self.measurement_vector_format is not None else self.state_vector_format["static"]
        expected_dim_z = len(z_format)
        if len(self.R_values) != expected_dim_z:
            raise ValueError(
                f"R_values length ({len(self.R_values)}) must match measurement vector dimension ({expected_dim_z}). "
                f"measurement_vector_format: {self.measurement_vector_format}, "
                f"state_vector_format['static']: {self.state_vector_format['static']}"
            )

    def _validate_state_vector_format(self) -> None:
        """
        Validate the state vector format.
        
        Raises:
        ----------
        ValueError: If the state vector format is not valid.
            - If the static variables are not unique.
            - If the dynamic variables are not in the static variables.
            - If the acceleration variables are not in the dynamic variables.
        """
        static_variables = self.state_vector_format["static"]
        if len(set(static_variables)) != len(static_variables):
            raise ValueError(f"Static variables must be unique: {static_variables}")
        dynamic_variables = self.state_vector_format["dynamic"]
        acceleration_variables = self.state_vector_format["acceleration"]
        if any(variable not in static_variables for variable in dynamic_variables):
            raise ValueError(f"Dynamic variables must be in static variables: {dynamic_variables}")
        if any(variable not in dynamic_variables for variable in acceleration_variables):
            raise ValueError(f"Acceleration variables must be in dynamic variables: {acceleration_variables}")

    def _initialize_variables(self) -> None:
        """
        Initialize the variables and validates.
        """
        self.x = StateVector.register_zero_vec_from_format(format=self.state_vector_format)
        z_format = self.measurement_vector_format if self.measurement_vector_format is not None else self.state_vector_format["static"]
        self.z = MeasurementVector.register_zero_vec_from_format(format=z_format)
        self.R = MeasurementNoiseCovariance.create_matrix(
            weights=self.R_values, 
            scaling_factor=1
            )
        self.P = ErrorCovarianceMatrix.create_matrix(weights=self.P_values)
        self.Q = ProcessNoiseCovariance.create_matrix(weights=self.Q_values)
        self.H = MeasurementMatrix.create_matrix(
            state_vector=self.x, 
            measurement_vector=self.z,
            matrix_value=self.measurement_matrix_value
            )
        self.F = StateTransitionMatrix.create_matrix(
            x=self.x, 
            dt=self.dt
            )
        self.K = KalmanGain.create_zero_matrix(
            dim_x=self.x.dim, 
            dim_z=self.z.dim
            )
        self.S = InnovationCovarianceMatrix.create_zero_matrix(
            dim_z=self.z.dim
            )
        self._validate_variables()

    def _validate_variables(self) -> None:
        """
        Validate the variables shapes.

        Raises:
        ----------
        ValueError: If the variables are not valid.
            - If the shape of the variables are not valid.
        """
        dim_x = self.x.dim
        dim_z = self.z.dim
        if self.R.shape != (dim_z, dim_z):
            raise ValueError(f"R must have shape ({dim_z}, {dim_z}). Current shape: {self.R.shape}")
        if self.P.shape != (dim_x, dim_x):
            raise ValueError(f"P must have shape ({dim_x}, {dim_x}). Current shape: {self.P.shape}")
        if self.Q.shape != (dim_x, dim_x):
            raise ValueError(f"Q must have shape ({dim_x}, {dim_x}). Current shape: {self.Q.shape}")
        if self.H.shape != (dim_z, dim_x):
            raise ValueError(f"H must have shape ({dim_z}, {dim_x}). Current shape: {self.H.shape}")
        if self.F.shape != (dim_x, dim_x):
            raise ValueError(f"F must have shape ({dim_x}, {dim_x}). Current shape: {self.F.shape}")
        if self.K.shape != (dim_x, dim_z):
            raise ValueError(f"K must have shape ({dim_x}, {dim_z}). Current shape: {self.K.shape}")
        if self.S.shape != (dim_z, dim_z):
            raise ValueError(f"S must have shape ({dim_z}, {dim_z}). Current shape: {self.S.shape}")

    def clone(self) -> KFParameters:
        """Return a new instance (faster than deepcopy). __post_init__ runs and creates fresh state."""
        return replace(self)

