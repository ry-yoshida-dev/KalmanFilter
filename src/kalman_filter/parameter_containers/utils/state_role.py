from enum import Enum

class StateRole(Enum):
    """
    State role in Kalman filter.
    
    Attributes:
    ----------
    STATIC : State role for static variables (0th derivative, e.g., position).
    DYNAMIC : State role for dynamic variables (1st derivative, e.g., velocity).
    ACCELERATION : State role for acceleration variables (2nd derivative, e.g., acceleration).
    """
    STATIC = "static"
    DYNAMIC = "dynamic"
    ACCELERATION = "acceleration"
