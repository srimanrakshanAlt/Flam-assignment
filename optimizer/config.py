"""Configuration and boundary settings for the parametric curve optimizer."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ParameterBounds:
    """Range boundaries for the unknown parameters as specified in the assignment."""
    theta_min_deg: float = 0.0
    theta_max_deg: float = 50.0
    
    M_min: float = -0.05
    M_max: float = 0.05
    
    X_min: float = 0.0
    X_max: float = 100.0
    
    t_min: float = 6.0
    t_max: float = 60.0

# Instantiated bounds for global use
BOUNDS = ParameterBounds()
