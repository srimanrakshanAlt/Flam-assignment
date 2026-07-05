"""Mathematical model functions and coordinate transformations for parametric curve fitting."""
import numpy as np
from typing import Tuple, Union

# Vector or scalar inputs are supported
NumericArray = Union[float, np.ndarray]

def parametric_x(t: NumericArray, theta_rad: float, M: float, X: float) -> NumericArray:
    """Compute the X coordinate of the parametric curve for given t.
    
    Formula: x = t * cos(theta) - e^(M*|t|) * sin(0.3t) * sin(theta) + X
    """
    return t * np.cos(theta_rad) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta_rad) + X

def parametric_y(t: NumericArray, theta_rad: float, M: float) -> NumericArray:
    """Compute the Y coordinate of the parametric curve for given t.
    
    Formula: y = 42 + t * sin(theta) + e^(M*|t|) * sin(0.3t) * cos(theta)
    """
    return 42.0 + t * np.sin(theta_rad) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta_rad)

def project_to_t(x: NumericArray, y: NumericArray, theta_rad: float, X: float) -> NumericArray:
    """Project coordinate points (x, y) onto the longitudinal axis to extract the parameter t.
    
    Derived via: t = (x - X) * cos(theta) + (y - 42) * sin(theta)
    """
    return (x - X) * np.cos(theta_rad) + (y - 42.0) * np.sin(theta_rad)

def project_to_z_actual(x: NumericArray, y: NumericArray, theta_rad: float, X: float) -> NumericArray:
    """Project coordinate points (x, y) onto the transverse axis to compute the actual oscillatory term.
    
    Derived via: z_actual = -(x - X) * sin(theta) + (y - 42) * cos(theta)
    """
    return -(x - X) * np.sin(theta_rad) + (y - 42.0) * np.cos(theta_rad)

def project_to_z_expected(t: NumericArray, M: float) -> NumericArray:
    """Compute the expected transverse coordinate component.
    
    Formula: z_expected = e^(M*|t|) * sin(0.3t)
    """
    return np.exp(M * np.abs(t)) * np.sin(0.3 * t)

def reconstruct_curve_points(t: NumericArray, theta_rad: float, M: float, X: float) -> Tuple[NumericArray, NumericArray]:
    """Reconstruct coordinates (x, y) from the parameter t."""
    x = parametric_x(t, theta_rad, M, X)
    y = parametric_y(t, theta_rad, M)
    return x, y
