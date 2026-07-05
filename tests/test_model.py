"""Unit tests for the mathematical model and projection formulas of the curve fitting solver."""
import pytest
import numpy as np
from optimizer.model import (
    parametric_x,
    parametric_y,
    project_to_t,
    project_to_z_actual,
    project_to_z_expected,
    reconstruct_curve_points
)

def test_coordinate_projections():
    """Verify that coordinate rotations project (x, y) back to t and expected z correctly.
    
    Proof:
    t = (x - X) * cos(theta) + (y - 42) * sin(theta)
    """
    theta_deg = 30.0
    theta_rad = np.radians(theta_deg)
    M = 0.03
    X = 55.0
    
    # Test values for t
    t_test = np.linspace(6.0, 60.0, 100)
    
    # Generate exact (x, y) coordinates from known t
    x_exact = parametric_x(t_test, theta_rad, M, X)
    y_exact = parametric_y(t_test, theta_rad, M)
    
    # Project back using coordinate transforms
    t_projected = project_to_t(x_exact, y_exact, theta_rad, X)
    
    # Check that t is reconstructed perfectly
    np.testing.assert_allclose(t_projected, t_test, rtol=1e-12, atol=1e-12)
    
    # Verify that the transverse component projection matches the mathematical definition
    z_actual = project_to_z_actual(x_exact, y_exact, theta_rad, X)
    z_expected = project_to_z_expected(t_test, M)
    
    np.testing.assert_allclose(z_actual, z_expected, rtol=1e-12, atol=1e-12)

def test_scalar_vs_array():
    """Ensure math functions handle both scalar inputs and numpy arrays consistently."""
    theta_rad = 0.5
    M = 0.02
    X = 50.0
    t_val = 10.0
    
    # Scalar execution
    x_scalar = parametric_x(t_val, theta_rad, M, X)
    y_scalar = parametric_y(t_val, theta_rad, M)
    
    # Array execution
    t_array = np.array([t_val])
    x_array = parametric_x(t_array, theta_rad, M, X)
    y_array = parametric_y(t_array, theta_rad, M)
    
    assert isinstance(x_scalar, float)
    assert isinstance(y_scalar, float)
    np.testing.assert_allclose(x_scalar, x_array[0])
    np.testing.assert_allclose(y_scalar, y_array[0])

def test_reconstruction_interface():
    """Verify that reconstruct_curve_points matches individual coordinate functions."""
    t = np.array([12.0, 24.0, 36.0])
    theta_rad = 0.3
    M = -0.01
    X = 45.0
    
    x_ind = parametric_x(t, theta_rad, M, X)
    y_ind = parametric_y(t, theta_rad, M)
    
    x_rec, y_rec = reconstruct_curve_points(t, theta_rad, M, X)
    
    np.testing.assert_allclose(x_rec, x_ind)
    np.testing.assert_allclose(y_rec, y_ind)
