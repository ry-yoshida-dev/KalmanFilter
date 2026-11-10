"""Abstract Kalman-type filter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import override

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve

from .._arrays import as_covariance, as_vector
from .._numerics import symmetrize
from ..array_types import BoolArray, FloatArray
from ..models import MeasurementModel, MotionModel
from ..parameters import (
    ErrorCovarianceMatrix,
    Innovation,
    InnovationCovarianceMatrix,
    KalmanGain,
    MeasurementMatrix,
    MeasurementNoiseCovariance,
    ProcessNoiseCovariance,
    StateTransitionMatrix,
    StateVector,
)
from .projection import Projection
from .propagation import Propagation

_LOG_TWO_PI: float = float(np.log(2.0 * np.pi))
_MINIMUM_CONVERGENCE_SCALE: float = 1e-12


class KalmanFilter(ABC):
    """
    Kalman-type filter that holds its estimate and the matrices of its latest step.

    Every quantity of the filter is an attribute holding its own container
    (:class:`StateVector`, :class:`ErrorCovarianceMatrix`, ...), whose raw
    array is ``value``. The estimate ``x`` and ``P`` can be read, replaced,
    or edited element by element at any time; ``F`` and ``Q`` hold the
    values used by the latest :meth:`predict`, and ``H``, ``R``, ``y``,
    ``S``, and ``K`` those of the latest :meth:`update`. Subclasses only
    define how the estimate is propagated through the (possibly nonlinear)
    models.

    ``x`` and ``P`` return views of the current estimate: element writes
    (``kf.x["u"] = 1.0``, ``kf.P["u", "u"] = 4.0``) change the filter, while
    :meth:`predict` and :meth:`update` replace the estimate, so read the
    attribute again after each step. ``F``, ``Q``, ``H``, and ``R`` can be
    read-only views of the model's own matrices.

    Measurement components that are NaN are treated as unobserved. The
    update quantities ``H``, ``R``, ``y``, ``S``, and ``K`` then cover only
    the observed components, given by ``is_observed``.

    Parameters
    ----------
    motion_model : MotionModel
        Process model used by :meth:`predict`.
    measurement_model : MeasurementModel
        Default sensor model used by :meth:`update`.
    x : ArrayLike
        Initial state with shape ``(n,)``.
    P : ArrayLike
        Initial covariance with shape ``(n, n)``, or its diagonal.
    fading_memory_factor : float
        Keyword-only. ``alpha >= 1``; the covariance is multiplied by
        ``alpha^2`` before each prediction so old measurements are forgotten
        faster. ``1`` gives the standard filter.

    Raises
    ------
    ValueError
        If ``x`` or ``P`` does not match the motion model, ``x`` is not
        finite, ``P`` is not a valid covariance, or ``fading_memory_factor``
        is smaller than 1.
    """

    def __init__(
        self,
        motion_model: MotionModel,
        measurement_model: MeasurementModel,
        x: ArrayLike,
        P: ArrayLike,
        *,
        fading_memory_factor: float = 1.0,
    ) -> None:
        if not fading_memory_factor >= 1.0:
            raise ValueError(f"fading_memory_factor must be >= 1, got {fading_memory_factor}")
        self._motion_model: MotionModel = motion_model
        self._measurement_model: MeasurementModel = measurement_model
        self.fading_memory_factor: float = fading_memory_factor
        self._state: FloatArray = _as_state(x, motion_model.dimension)
        self._covariance: FloatArray = as_covariance(P, "P", motion_model.dimension)
        self._transition_matrix: StateTransitionMatrix | None = None
        self._process_noise: ProcessNoiseCovariance | None = None
        self._measurement_matrix: MeasurementMatrix | None = None
        self._measurement_noise: MeasurementNoiseCovariance | None = None
        self._innovation: Innovation | None = None
        self._innovation_covariance: InnovationCovarianceMatrix | None = None
        self._gain: KalmanGain | None = None
        self._is_observed: BoolArray | None = None
        self._log_likelihood: float | None = None

    @property
    def motion_model(self) -> MotionModel:
        """Process model used by :meth:`predict`; fixed, since it defines the state layout."""
        return self._motion_model

    @property
    def measurement_model(self) -> MeasurementModel:
        """Default sensor model of :meth:`update`; pass another one to :meth:`update` instead."""
        return self._measurement_model

    @property
    def x(self) -> StateVector:
        """State estimate ``x``; writing an element changes the filter."""
        return StateVector(self._state, self.motion_model.layout)

    @x.setter
    def x(self, value: ArrayLike) -> None:  # pyright: ignore[reportPropertyTypeMismatch]
        """
        Replace the state estimate.

        Raises
        ------
        ValueError
            If ``value`` does not have length ``n`` or is not finite.
        """
        self._state = _as_state(value, self.motion_model.dimension)

    @property
    def P(self) -> ErrorCovarianceMatrix:
        """Estimate covariance ``P``; writing an element changes the filter."""
        return ErrorCovarianceMatrix(self._covariance, self.motion_model.layout)

    @P.setter
    def P(self, value: ArrayLike) -> None:  # pyright: ignore[reportPropertyTypeMismatch]
        """
        Replace the estimate covariance; a 1-D input is taken as the diagonal.

        Raises
        ------
        ValueError
            If ``value`` is not a valid ``(n, n)`` covariance.
        """
        self._covariance = as_covariance(value, "P", self.motion_model.dimension)

    @property
    def F(self) -> StateTransitionMatrix | None:
        """Transition matrix of the latest prediction; ``None`` before it."""
        return self._transition_matrix

    @property
    def Q(self) -> ProcessNoiseCovariance | None:
        """Process noise covariance of the latest prediction; ``None`` before it."""
        return self._process_noise

    @property
    def H(self) -> MeasurementMatrix | None:
        """Measurement matrix of the latest update; ``None`` before it."""
        return self._measurement_matrix

    @property
    def R(self) -> MeasurementNoiseCovariance | None:
        """Measurement noise covariance of the latest update; ``None`` before it."""
        return self._measurement_noise

    @property
    def y(self) -> Innovation | None:
        """Innovation ``z - h(x)`` of the latest update; ``None`` before it."""
        return self._innovation

    @property
    def S(self) -> InnovationCovarianceMatrix | None:
        """Innovation covariance of the latest update; ``None`` before it."""
        return self._innovation_covariance

    @property
    def K(self) -> KalmanGain | None:
        """Kalman gain of the latest update with shape ``(n, observed)``; ``None`` before it."""
        return self._gain

    @property
    def is_observed(self) -> BoolArray | None:
        """Mask of the measurement components used by the latest update; ``None`` before it."""
        return self._is_observed

    @property
    def log_likelihood(self) -> float | None:
        """``log N(z; h(x), S)`` of the latest update; ``None`` before it."""
        return self._log_likelihood

    def predict(self, dt: float = 1.0, u: ArrayLike | None = None) -> None:
        """
        Predict the estimate ``dt`` ahead and store ``F`` and ``Q``.

        Parameters
        ----------
        dt : float
            Non-negative time step; may differ between calls.
        u : ArrayLike | None
            Control input, or ``None`` when there is none.

        Raises
        ------
        ValueError
            If ``dt`` is negative or not finite, ``u`` is not finite, or the
            motion model returns a state of the wrong shape or a non-finite prior.
        """
        if not (np.isfinite(dt) and dt >= 0.0):
            raise ValueError(f"dt must be finite and non-negative, got {dt}")
        control = None if u is None else as_vector(u, "u")
        if control is not None and not np.all(np.isfinite(control)):
            raise ValueError(f"u must be finite, got {control}")
        dimension = self.motion_model.dimension
        covariance = self._covariance * self.fading_memory_factor**2
        propagation = self._propagate(self._state, covariance, float(dt), control)
        # A copy, so element writes through ``x`` never reach an array the model returned.
        state = np.array(propagation.x, dtype=np.float64)
        if state.shape != (dimension,):
            raise ValueError(
                f"motion model returned a state of shape {state.shape}, expected ({dimension},)"
            )
        if not (np.all(np.isfinite(state)) and np.all(np.isfinite(propagation.P))):
            raise ValueError("predicted estimate is not finite; check the motion model and dt")
        self._state = state
        self._covariance = symmetrize(propagation.P)
        self._transition_matrix = StateTransitionMatrix(propagation.F)
        self._process_noise = ProcessNoiseCovariance(propagation.Q)

    def update(self, z: ArrayLike, measurement_model: MeasurementModel | None = None) -> None:
        """
        Correct the estimate with a measurement and store ``H``, ``R``, ``y``, ``S``, and ``K``.

        Parameters
        ----------
        z : ArrayLike
            Measurement with shape ``(m,)``; NaN marks missing components.
        measurement_model : MeasurementModel | None
            Sensor model, or ``None`` for the filter's default.

        Raises
        ------
        ValueError
            If ``z`` does not match the measurement model or is infinite, the
            model returns a measurement of another size, or the posterior is
            not finite.
        numpy.linalg.LinAlgError
            If the innovation covariance is not positive definite.
        """
        model = self.measurement_model if measurement_model is None else measurement_model
        observation = as_vector(z, "z", model.dimension)
        if np.any(np.isinf(observation)):
            raise ValueError(
                f"z must not be infinite (use NaN for missing components), got {observation}"
            )
        is_observed = ~np.isnan(observation)
        if not np.any(is_observed):
            dimension = self.motion_model.dimension
            self._measurement_matrix = MeasurementMatrix(np.empty((0, dimension)))
            self._measurement_noise = MeasurementNoiseCovariance(np.empty((0, 0)))
            self._innovation = Innovation(np.empty(0))
            self._innovation_covariance = InnovationCovarianceMatrix(np.empty((0, 0)))
            self._gain = KalmanGain(np.empty((dimension, 0)))
            self._is_observed = is_observed
            self._log_likelihood = 0.0
            return
        self._correct(observation, is_observed, model)

    def _correct(
        self, observation: FloatArray, is_observed: BoolArray, model: MeasurementModel
    ) -> None:
        """
        Apply the measurement correction to the stored prior.

        The default applies one :meth:`_apply_projection`; iterated filters
        override it to relinearize around the posterior.
        """
        projection = self._project(self._state, self._covariance, model)
        self._apply_projection(
            self._state, self._covariance, projection, observation, is_observed, model
        )

    def _iterate_correction(
        self,
        observation: FloatArray,
        is_observed: BoolArray,
        model: MeasurementModel,
        relinearize: Callable[[FloatArray, FloatArray], Projection],
        max_iterations: int,
        tolerance: float,
    ) -> None:
        """
        Repeat the correction with the measurement model relinearized around the posterior.

        ``relinearize(x, P)`` must return the affine measurement model fitted
        around the current posterior and expressed against the prior. Iteration
        stops when no state component moves by more than ``tolerance`` prior
        standard deviations, or after ``max_iterations`` linearizations.
        """
        prior_x, prior_P = self._state, self._covariance
        scale = np.maximum(np.sqrt(np.diag(prior_P)), _MINIMUM_CONVERGENCE_SCALE)
        projection = self._project(prior_x, prior_P, model)
        self._apply_projection(prior_x, prior_P, projection, observation, is_observed, model)
        for _ in range(max_iterations - 1):
            previous_x = self._state
            projection = relinearize(self._state, self._covariance)
            self._apply_projection(prior_x, prior_P, projection, observation, is_observed, model)
            step = self.motion_model.residual(self._state, previous_x)
            if np.all(np.abs(step) <= tolerance * scale):
                break

    def _apply_projection(
        self,
        prior_x: FloatArray,
        prior_P: FloatArray,
        projection: Projection,
        observation: FloatArray,
        is_observed: BoolArray,
        model: MeasurementModel,
    ) -> None:
        """Kalman correction of the prior with the observed components of ``projection``."""
        if projection.z.shape != observation.shape:
            raise ValueError(
                f"measurement model returned shape {projection.z.shape}, "
                + f"expected {observation.shape}"
            )
        observed = np.flatnonzero(is_observed)
        filled = np.where(is_observed, observation, projection.z)
        innovation = model.residual(filled, projection.z)[observed]
        innovation_covariance = projection.S[np.ix_(observed, observed)]
        cross_covariance = projection.cross_covariance[:, observed]
        noise_covariance = projection.R[np.ix_(observed, observed)]

        factor = cho_factor(innovation_covariance, lower=True, check_finite=False)
        gain: FloatArray = cho_solve(factor, cross_covariance.T, check_finite=False).T
        posterior_x = self.motion_model.add(prior_x, gain @ innovation)
        if not np.all(np.isfinite(posterior_x)):
            raise ValueError("posterior state is not finite; check the models and measurements")

        jacobian = projection.H[observed]
        if self._uses_joseph_form():
            residual_operator = np.eye(prior_x.shape[0]) - gain @ jacobian
            posterior_P = (
                residual_operator @ prior_P @ residual_operator.T + gain @ noise_covariance @ gain.T
            )
        else:
            posterior_P = prior_P - gain @ innovation_covariance @ gain.T

        whitened = cho_solve(factor, innovation, check_finite=False)
        log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
        log_likelihood = -0.5 * float(
            innovation @ whitened + observed.shape[0] * _LOG_TWO_PI + log_determinant
        )
        self._state = posterior_x
        self._covariance = symmetrize(posterior_P)
        self._measurement_matrix = MeasurementMatrix(jacobian)
        self._measurement_noise = MeasurementNoiseCovariance(noise_covariance)
        self._innovation = Innovation(innovation)
        self._innovation_covariance = InnovationCovarianceMatrix(innovation_covariance)
        self._gain = KalmanGain(gain)
        self._is_observed = is_observed
        self._log_likelihood = log_likelihood

    def _uses_joseph_form(self) -> bool:
        """Whether the posterior covariance uses the Joseph form instead of ``P - K S K^T``."""
        return False

    @abstractmethod
    def _propagate(
        self, x: FloatArray, P: FloatArray, dt: float, control: FloatArray | None
    ) -> Propagation:
        """Propagate the estimate through the motion model and add process noise."""

    @abstractmethod
    def _project(self, x: FloatArray, P: FloatArray, model: MeasurementModel) -> Projection:
        """Project the estimate into the measurement space of ``model``."""

    @override
    def __repr__(self) -> str:
        """Human-readable summary of every quantity of the filter."""
        quantities = {
            "x": self.x,
            "P": self.P,
            "F": self._transition_matrix,
            "Q": self._process_noise,
            "H": self._measurement_matrix,
            "R": self._measurement_noise,
            "y": self._innovation,
            "S": self._innovation_covariance,
            "K": self._gain,
        }
        rows = [type(self).__name__] + [
            f"{name}: {None if value is None else value.value}"
            for name, value in quantities.items()
        ]
        return "\n".join(rows)


def _as_state(value: ArrayLike, dimension: int) -> FloatArray:
    """Validated copy of a state estimate."""
    state = as_vector(value, "x", dimension)
    if not np.all(np.isfinite(state)):
        raise ValueError(f"x must be finite, got {state}")
    return state
