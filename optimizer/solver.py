"""Solver class to perform robust parameter estimation using grid search and L-BFGS-B optimization."""
import logging
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, Tuple
from optimizer.config import BOUNDS
from optimizer.model import project_to_t, project_to_z_actual, project_to_z_expected

logger = logging.getLogger(__name__)

class CurveFitter:
    """Performs parameter extraction (theta, M, X) from a set of (x, y) coordinates."""
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        """Initialize the fitter with dataset coordinates.
        
        Args:
            x_data: Numpy array of X coordinates.
            y_data: Numpy array of Y coordinates.
        """
        self.x_data = np.asarray(x_data, dtype=float)
        self.y_data = np.asarray(y_data, dtype=float)
        
        if len(self.x_data) != len(self.y_data):
            raise ValueError("X and Y coordinate arrays must have matching dimensions.")
            
    def compute_loss(self, params: Tuple[float, float, float]) -> float:
        """Compute the L1 regression loss of the transverse projection for candidate parameters.
        
        Args:
            params: Tuple of (theta_degrees, M, X).
            
        Returns:
            L1 loss value with boundary penalty if t goes outside specified range [6, 60].
        """
        theta_deg, M, X = params
        theta_rad = np.radians(theta_deg)
        
        # Calculate parameter t for each point
        t = project_to_t(self.x_data, self.y_data, theta_rad, X)
        
        # Range validation penalty to ensure t stays in [6, 60] range
        penalty = 0.0
        t_min_viol = BOUNDS.t_min - t
        t_max_viol = t - BOUNDS.t_max
        
        if np.any(t_min_viol > 0.0) or np.any(t_max_viol > 0.0):
            # Quadratic penalty for violations
            penalty += np.sum(np.maximum(0.0, t_min_viol)**2) * 100.0
            penalty += np.sum(np.maximum(0.0, t_max_viol)**2) * 100.0
            
        actual_z = project_to_z_actual(self.x_data, self.y_data, theta_rad, X)
        expected_z = project_to_z_expected(t, M)
        
        # Minimize the mean absolute error (L1 distance)
        l1_loss = np.mean(np.abs(actual_z - expected_z))
        return l1_loss + penalty
        
    def run_grid_search(self, grid_steps: int = 12) -> Tuple[float, float, float]:
        """Perform a coarse grid search to find global starting parameters.
        
        Args:
            grid_steps: Number of divisions per parameter axis.
            
        Returns:
            Tuple of (theta_deg, M, X) with the lowest initial loss.
        """
        logger.info("Executing coarse grid search to avoid local minima...")
        theta_grid = np.linspace(BOUNDS.theta_min_deg + 1.0, BOUNDS.theta_max_deg - 1.0, grid_steps)
        M_grid = np.linspace(BOUNDS.M_min + 0.01, BOUNDS.M_max - 0.01, grid_steps)
        X_grid = np.linspace(BOUNDS.X_min + 5.0, BOUNDS.X_max - 5.0, grid_steps)
        
        best_loss = float('inf')
        best_params = (25.0, 0.0, 50.0)
        
        for theta_deg in theta_grid:
            for M in M_grid:
                for X in X_grid:
                    loss = self.compute_loss((theta_deg, M, X))
                    if loss < best_loss:
                        best_loss = loss
                        best_params = (theta_deg, M, X)
                        
        logger.info("Grid search found optimal starting point: theta=%.2f°, M=%.4f, X=%.2f (Loss: %.6f)",
                    best_params[0], best_params[1], best_params[2], best_loss)
        return best_params
        
    def fit(self, initial_guess: Tuple[float, float, float] = None) -> Dict[str, Any]:
        """Solve for the unknown parameters using grid search + L-BFGS-B gradient minimization.
        
        Args:
            initial_guess: Optional starting guess. If None, runs grid search.
            
        Returns:
            Dictionary containing optimized parameters and loss evaluation.
        """
        if initial_guess is None:
            initial_guess = self.run_grid_search()
            
        bounds = [
            (BOUNDS.theta_min_deg, BOUNDS.theta_max_deg),
            (BOUNDS.M_min, BOUNDS.M_max),
            (BOUNDS.X_min, BOUNDS.X_max)
        ]
        
        logger.info("Refining parameters with L-BFGS-B minimization...")
        res = minimize(
            self.compute_loss,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B',
            options={'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if not res.success:
            logger.warning("Optimization did not converge cleanly: %s", res.message)
            
        theta_deg_opt, M_opt, X_opt = res.x
        theta_rad_opt = np.radians(theta_deg_opt)
        
        # Calculate final metrics
        t_opt = project_to_t(self.x_data, self.y_data, theta_rad_opt, X_opt)
        
        x_pred = t_opt * np.cos(theta_rad_opt) - np.exp(M_opt * t_opt) * np.sin(0.3 * t_opt) * np.sin(theta_rad_opt) + X_opt
        y_pred = 42.0 + t_opt * np.sin(theta_rad_opt) + np.exp(M_opt * t_opt) * np.sin(0.3 * t_opt) * np.cos(theta_rad_opt)
        
        l1_error_x = np.mean(np.abs(self.x_data - x_pred))
        l1_error_y = np.mean(np.abs(self.y_data - y_pred))
        euclidean_errors = np.sqrt((self.x_data - x_pred)**2 + (self.y_data - y_pred)**2)
        mean_euclidean = np.mean(euclidean_errors)
        max_euclidean = np.max(euclidean_errors)
        
        return {
            "theta_deg": theta_deg_opt,
            "theta_rad": theta_rad_opt,
            "M": M_opt,
            "X": X_opt,
            "t_values": t_opt,
            "l1_error_x": l1_error_x,
            "l1_error_y": l1_error_y,
            "mean_euclidean_error": mean_euclidean,
            "max_euclidean_error": max_euclidean,
            "success": res.success,
            "message": res.message
        }
