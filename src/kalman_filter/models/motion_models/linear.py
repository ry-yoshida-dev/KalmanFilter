"""Time-invariant linear motion model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

from numpy.typing import ArrayLike

from ..._arrays import as_covariance, as_matrix, as_vector, read_only
from ...array_types import FloatArray
from ..motion_model import MotionModel
from ..state_layout import StateLayout


class LinearMotionModel(MotionModel):
    """
    Linear motion model ``x_k = F x_{k-1} + B u + w`` with constant matrices.

    The matrices do not depend on ``dt``; use
    :class:`KinematicMotionModel` for models whose matrices scale with time.
    The stored matrices are read-only.

    Parameters
    ----------
    transition_matrix : ArrayLike
        State transition matrix ``F`` with shape ``(n, n)``.
    process_noise_covariance : ArrayLike
        Process noise ``Q`` with shape ``(n, n)``, or its diagonal.
    control_matrix : ArrayLike | None
        Control input matrix ``B`` with shape ``(n, p)``, or ``None`` when the
        model has no control input.
    names : Sequence[str] | None
        State variable names, or ``None`` to generate ``x0, x1, ...``.

    Raises
    ------
    ValueError
        If the matrix shapes are inconsistent.

    Examples
    --------
    >>> model = LinearMotionModel(
    ...     transition_matrix=[[1.0, 1.0], [0.0, 1.0]],
    ...     process_noise_covariance=[0.01, 0.01],
    ... )
    """

    def __init__(
        self,
        transition_matrix: ArrayLike,
        process_noise_covariance: ArrayLike,
        control_matrix: ArrayLike | None = None,
        names: Sequence[str] | None = None,
    ) -> None:
        transition = as_matrix(transition_matrix, "transition_matrix")
        dimension = transition.shape[0]
        if transition.shape != (dimension, dimension):
            raise ValueError(f"transition_matrix must be square, got {transition.shape}")
        self._transition_matrix: FloatArray = read_only(transition)
        self._process_noise_covariance: FloatArray = read_only(
            as_covariance(process_noise_covariance, "process_noise_covariance", dimension)
        )
        self._control_matrix: FloatArray | None = None
        if control_matrix is not None:
            control = as_matrix(control_matrix, "control_matrix")
            if control.shape[0] != dimension:
                raise ValueError(f"control_matrix must have {dimension} rows, got {control.shape}")
            self._control_matrix = read_only(control)
        self._layout: StateLayout = (
            StateLayout.anonymous(dimension) if names is None else StateLayout(names)
        )
        if self._layout.dimension != dimension:
            raise ValueError(f"names must have length {dimension}, got {self._layout.dimension}")

    @property
    @override
    def layout(self) -> StateLayout:
        """Names of the state variables."""
        return self._layout

    @property
    def transition_matrix(self) -> FloatArray:
        """State transition matrix ``F``."""
        return self._transition_matrix

    @property
    def control_matrix(self) -> FloatArray | None:
        """Control input matrix ``B``, or ``None`` when the model has no control input."""
        return self._control_matrix

    @override
    def transition(
        self, states: FloatArray, dt: float, control: FloatArray | None = None
    ) -> FloatArray:
        """
        Apply ``F x + B u``.

        Raises
        ------
        ValueError
            If ``control`` is given but the model has no control matrix, or
            ``control`` does not have one entry per column of ``B``.
        """
        propagated: FloatArray = states @ self._transition_matrix.T
        if control is None:
            return propagated
        if self._control_matrix is None:
            raise ValueError("control was given but the model has no control_matrix")
        control_vector = as_vector(control, "control", self._control_matrix.shape[1])
        return propagated + self._control_matrix @ control_vector

    @override
    def jacobian(
        self, mean: FloatArray, dt: float, control: FloatArray | None = None
    ) -> FloatArray:
        """Return ``F``."""
        return self._transition_matrix

    @override
    def process_noise(self, mean: FloatArray, dt: float) -> FloatArray:
        """Return the constant ``Q``."""
        return self._process_noise_covariance
