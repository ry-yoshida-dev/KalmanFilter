from __future__ import annotations

from dataclasses import dataclass, replace

from .parameter_containers import (
    ControlInputMatrix,
    ControlInputVector,
    ErrorCovarianceMatrix,
    InnovationCovarianceMatrix,
    KalmanGain,
    MeasurementMatrix,
    MeasurementNoiseCovariance,
    MeasurementVector,
    ProcessNoiseCovariance,
    ProcessNoiseGain,
    StateTransitionMatrix,
    StateVector,
    )
from .parameters import KFParameters

@dataclass
class KalmanFilter:
    """
    Kalman filter class.

    Attributes:
    ----------
    params: KFParameters
        The parameters for the Kalman filter.
    """
    params: KFParameters

    def set_initial_measurement(
        self, 
        z: MeasurementVector
        ) -> None:
        """
        Set the initial measurement.

        Parameters
        ----------
        z: MeasurementVector
            Initial measurement vector.
        """
        self.z = z
        self.x.replace_by_measurement(
            z=z, 
            H=self.H
            )

    def predict(
        self, 
        B: ControlInputMatrix | None = None, 
        u: ControlInputVector | None = None, 
        Q: ProcessNoiseCovariance | None = None, 
        G: ProcessNoiseGain | None = None,
        is_update_x: bool = True, 
        is_update_P: bool = True
        ) -> tuple[StateVector, ErrorCovarianceMatrix]:
        """
        Predict the next state and error covariance matrix.

        Parameters
        ----------
        B: ControlInputMatrix | None
            Control input matrix.
        u: ControlInputVector | None
            Control input vector.
        Q: ProcessNoiseCovariance | None
            Process noise covariance matrix.
        G: ProcessNoiseGain | None
            Process noise gain matrix.
        is_update_x: bool
            If True, update the state.
        is_update_P: bool
            If True, update the error covariance matrix.

        Returns
        -------
        tuple[StateVector, ErrorCovarianceMatrix]
            Predicted state and error covariance matrix.
        """
        pred_x = self.x.predict(
            F=self.F, 
            B=B, 
            u=u
            )
        if isinstance(Q, ProcessNoiseCovariance):
            self.Q = Q
        pred_P = self.P.predict(
            F=self.F, 
            Q=self.Q, 
            G=G,
            alpha = self.params.alpha
            )
        if is_update_x:
            self.x = pred_x
            self.z.update(
                H=self.H, 
                x=pred_x
                )
        if is_update_P:
            self.P = pred_P
        return pred_x, pred_P

    def update(
        self, 
        z: MeasurementVector, 
        is_correlated_noise: bool = False
        ) -> tuple[StateVector, ErrorCovarianceMatrix]:
        """
        Update the state.

        Parameters
        ----------
        z: MeasurementVector
            Measurement vector.
        is_correlated_noise: bool
            If True, use correlated noise update.
        
        Returns
        -------
        tuple[StateVector, ErrorCovarianceMatrix]
            Updated state and error covariance matrix.
        """
        innovation = z.compute_innovation(
            x=self.x, 
            H=self.H
            )
        PHT = self.P @ self.H.T
        self.S.update(
            P=self.P, 
            H=self.H, 
            R=self.R, 
            PHT=PHT
            )
        self.K.update(
            P=self.P, 
            H=self.H, 
            R=self.R, 
            PHT=PHT
            )        
        self.x.update(
            K=self.K, 
            innovation=innovation
            )
        self.P.update(
            K=self.K, 
            H=self.H, 
            R=self.R, 
            is_correlated_noise=is_correlated_noise)
        return self.x, self.P

    @property
    def x(self) -> StateVector:
        return self.params.x
    
    @x.setter
    def x(
        self, 
        value: StateVector
        ) -> None:
        """
        Set the state vector.

        Parameters:
        ----------
        value: StateVector
            New state vector.
        """
        self.params.x = value

    @property
    def z(self) -> MeasurementVector:
        return self.params.z
    
    @z.setter
    def z(
        self, 
        value: MeasurementVector
        ) -> None:
        """
        Set the measurement vector.

        Parameters:
        ----------
        value: MeasurementVector
            New measurement vector.
        """
        self.params.z = value

    @property
    def R(self) -> MeasurementNoiseCovariance:
        return self.params.R

    @property
    def P(self) -> ErrorCovarianceMatrix:
        return self.params.P
    
    @P.setter
    def P(self, value: ErrorCovarianceMatrix) -> None:
        self.params.P = value

    @property
    def Q(self) -> ProcessNoiseCovariance:
        return self.params.Q
    
    @Q.setter
    def Q(self, value: ProcessNoiseCovariance) -> None:
        self.params.Q = value

    @property
    def H(self) -> MeasurementMatrix:
        return self.params.H

    @property
    def F(self) -> StateTransitionMatrix:
        return self.params.F

    @property
    def K(self) -> KalmanGain:
        return self.params.K

    @property
    def S(self) -> InnovationCovarianceMatrix:
        return self.params.S

    def clone(self) -> KalmanFilter:
        """Return a new instance with cloned params (faster than deepcopy)."""
        return replace(self, params=self.params.clone())

    def __repr__(self) -> str:
        """
        Returns a human-readable summary of the Kalman filter's current state.
        """
        return '\n'.join([
            'KalmanFilter object',
            f'P: {self.P}',
            f'S: {self.S}',
            f'K: {self.K}',
            f'H: {self.H}',
            f'R: {self.R}',
            f'x: {self.x}',
            f'F: {self.F}',
            f'Q: {self.Q}',
            f'z: {self.z}'
        ])

