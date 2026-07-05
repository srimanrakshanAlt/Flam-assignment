"""AI/R&D Assignment - Parametric Curve Fitting Solution Script.

This script loads xy_data.csv, estimates the parameters (theta, M, X) using
analytical coordinate projection and numerical optimization, and generates
the required validation outputs and plots.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def parametric_curve(t, theta_rad, M, X):
    """Generate X and Y coordinates for the parametric curve given parameter t."""
    wave = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(theta_rad) - wave * np.sin(theta_rad) + X
    y = 42.0 + t * np.sin(theta_rad) + wave * np.cos(theta_rad)
    return x, y

def main():
    print("AI/R&D Curve Fitting Solver")
    print("===========================")
    
    # 1. Load data
    csv_path = "xy_data.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    x_actual = df['x'].to_numpy(dtype=float)
    y_actual = df['y'].to_numpy(dtype=float)
    
    # 2. Optimization using coordinate projection (analytical decoupling)
    # Objective: Minimize L1 error of transverse component
    def objective(params):
        theta_deg, M_val, X_val = params
        theta_r = np.radians(theta_deg)
        
        # Recover hidden parameter t for each point
        t_recovered = (x_actual - X_val) * np.cos(theta_r) + (y_actual - 42.0) * np.sin(theta_r)
        
        # Penalize if t values fall outside [6, 60] range
        penalty = 0.0
        if np.any(t_recovered < 5.0) or np.any(t_recovered > 61.0):
            penalty += np.sum(np.maximum(0, 5.0 - t_recovered)**2) * 100.0
            penalty += np.sum(np.maximum(0, t_recovered - 61.0)**2) * 100.0
            
        actual_z = -(x_actual - X_val) * np.sin(theta_r) + (y_actual - 42.0) * np.cos(theta_r)
        expected_z = np.exp(M_val * np.abs(t_recovered)) * np.sin(0.3 * t_recovered)
        
        return np.mean(np.abs(actual_z - expected_z)) + penalty

    # Grid search for robust initial guess
    best_loss = float('inf')
    best_guess = (25.0, 0.0, 50.0)
    for t_deg in np.linspace(1, 49, 10):
        for m in np.linspace(-0.04, 0.04, 9):
            for x_s in np.linspace(5, 95, 10):
                loss = objective((t_deg, m, x_s))
                if loss < best_loss:
                    best_loss = loss
                    best_guess = (t_deg, m, x_s)
                    
    # Local minimization refinement
    res = minimize(
        objective, 
        best_guess, 
        bounds=[(0.0, 50.0), (-0.05, 0.05), (0.0, 100.0)], 
        method='L-BFGS-B',
        options={'ftol': 1e-12, 'gtol': 1e-12}
    )
    
    theta_deg_opt, M_opt, X_opt = res.x
    theta_rad_opt = np.radians(theta_deg_opt)
    
    # 3. Recover t and generate predictions
    t_recovered = (x_actual - X_opt) * np.cos(theta_rad_opt) + (y_actual - 42.0) * np.sin(theta_rad_opt)
    x_pred, y_pred = parametric_curve(t_recovered, theta_rad_opt, M_opt, X_opt)
    
    # Calculate point-wise validation error
    l1_x = np.abs(x_actual - x_pred)
    l1_y = np.abs(y_actual - y_pred)
    mean_l1_error = np.mean(l1_x + l1_y)
    
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    
    # 4. Generate Output Plots
    # Style plots cleanly
    plt.rcParams.update({'font.size': 10, 'figure.facecolor': 'white'})
    
    # Plot 1: Overlay Plot (only plot every 20th point for visibility)
    plt.figure(figsize=(10, 7))
    plt.plot(x_pred[np.argsort(t_recovered)], y_pred[np.argsort(t_recovered)], color='red', linewidth=2.5, label='Predicted Fitted Curve')
    plt.scatter(x_actual[::20], y_actual[::20], color='blue', s=15, alpha=0.6, label='Actual CSV Points (Every 20th)')
    plt.title('Predicted Curve vs Sampled Actual CSV Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/curve_comparison_overlay.png', dpi=150)
    plt.close()
    
    # Plot 2: Side-by-Side Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.scatter(x_actual, y_actual, color='blue', s=5, alpha=0.5)
    ax1.set_title('Actual CSV Points (All)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Sort model line by t so it draws continuously
    sort_idx = np.argsort(t_recovered)
    ax2.plot(x_pred[sort_idx], y_pred[sort_idx], color='red', linewidth=2.5)
    ax2.set_title('Predicted Parametric Curve')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle('Actual vs Predicted Curve Comparison')
    plt.tight_layout()
    plt.savefig('outputs/curve_comparison_side_by_side.png', dpi=150)
    plt.close()
    
    # Plot 3: Residual Analysis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax1.plot(l1_x, color='purple', alpha=0.7)
    ax1.set_ylabel('X Absolute Error')
    ax1.set_title('Coordinate Residual Analysis')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.plot(l1_y, color='orange', alpha=0.7)
    ax2.set_ylabel('Y Absolute Error')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    ax3.plot(l1_x + l1_y, color='green', alpha=0.7)
    ax3.set_ylabel('L1 Error per Point')
    ax3.set_xlabel('Point Index')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('outputs/residual_analysis.png', dpi=150)
    plt.close()
    
    # Plot 4: Residual Scatter Plot
    plt.figure(figsize=(8, 7))
    plt.scatter(x_actual - x_pred, y_actual - y_pred, color='brown', s=8, alpha=0.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Residual Scatter (X Error vs Y Error)')
    plt.xlabel('X Error (Actual - Predicted)')
    plt.ylabel('Y Error (Actual - Predicted)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('outputs/residual_scatter.png', dpi=150)
    plt.close()
    
    # 5. Save results to text file
    latex_eq = (
        f"\\left(t\\cos({theta_rad_opt:.6f})-e^{{{M_opt:.6f}\\left|t\\right|}}\\sin(0.3t)\\sin({theta_rad_opt:.6f})+{X_opt:.6f},"
        f"\\ 42+t\\sin({theta_rad_opt:.6f})+e^{{{M_opt:.6f}\\left|t\\right|}}\\sin(0.3t)\\cos({theta_rad_opt:.6f})\\right)"
    )
    
    with open('outputs/final_result.txt', 'w', encoding='utf-8') as f:
        f.write("Final Parameters\n")
        f.write("----------------\n")
        f.write(f"theta = {theta_deg_opt:.10f} degrees\n")
        f.write(f"theta = {theta_rad_opt:.10f} radians\n")
        f.write(f"M     = {M_opt:.10f}\n")
        f.write(f"X     = {X_opt:.10f}\n")
        f.write(f"L1 error = {mean_l1_error:.12f}\n\n")
        f.write("Final Equation\n")
        f.write("--------------\n")
        f.write(f"{latex_eq}\n")
        
    # 6. Terminal Outputs
    print("\nFinal Parameters")
    print("----------------")
    print(f"theta = {theta_deg_opt:.10f} degrees")
    print(f"theta = {theta_rad_opt:.10f} radians")
    print(f"M     = {M_opt:.10f}")
    print(f"X     = {X_opt:.10f}")
    print(f"L1 error = {mean_l1_error:.12f}")
    print("\nFinal Equation")
    print("--------------")
    print(latex_eq)
    print("\nFiles generated:")
    print("outputs/curve_comparison_overlay.png")
    print("outputs/curve_comparison_side_by_side.png")
    print("outputs/residual_analysis.png")
    print("outputs/residual_scatter.png")
    print("outputs/final_result.txt")

if __name__ == '__main__':
    main()
