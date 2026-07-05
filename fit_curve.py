import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def main():
    # Load the data
    data_path = "xy_data.csv"
    df = pd.read_csv(data_path)
    x = df['x'].values
    y = df['y'].values
    
    # We want to minimize the L1 or L2 distance
    # The equations are:
    # t = (x - X) * cos(theta) + (y - 42) * sin(theta)
    # expected_z = exp(M * t) * sin(0.3 * t)
    # actual_z = -(x - X) * sin(theta) + (y - 42) * cos(theta)
    
    def loss_func(params):
        theta_deg, M, X = params
        theta = np.radians(theta_deg)
        
        # Compute t for each point
        t = (x - X) * np.cos(theta) + (y - 42) * np.sin(theta)
        
        # Check if t is within range (6, 60). Add a penalty if not.
        penalty = 0.0
        if np.any(t < 5.0) or np.any(t > 61.0):
            penalty += np.sum(np.maximum(0, 5.0 - t)**2) + np.sum(np.maximum(0, t - 61.0)**2) * 1000
            
        actual_z = -(x - X) * np.sin(theta) + (y - 42) * np.cos(theta)
        expected_z = np.exp(M * t) * np.sin(0.3 * t)
        
        # L1 loss
        loss = np.mean(np.abs(actual_z - expected_z)) + penalty
        return loss

    # Grid search for robust initial guess
    best_loss = float('inf')
    best_params = None
    
    print("Starting grid search...")
    for theta_deg in np.linspace(1, 49, 10):
        for M in np.linspace(-0.04, 0.04, 9):
            for X in np.linspace(5, 95, 10):
                loss = loss_func([theta_deg, M, X])
                if loss < best_loss:
                    best_loss = loss
                    best_params = (theta_deg, M, X)
                    
    print(f"Grid search best loss: {best_loss:.4f} at theta={best_params[0]:.2f}, M={best_params[1]:.4f}, X={best_params[2]:.2f}")
    
    # Refine using minimize
    res = minimize(loss_func, best_params, bounds=[(0.0, 50.0), (-0.05, 0.05), (0.0, 100.0)], method='L-BFGS-B')
    print("\nOptimization results:")
    print(res)
    
    theta_deg_opt, M_opt, X_opt = res.x
    theta_opt = np.radians(theta_deg_opt)
    
    # Verify errors
    t_opt = (x - X_opt) * np.cos(theta_opt) + (y - 42) * np.sin(theta_opt)
    x_pred = t_opt * np.cos(theta_opt) - np.exp(M_opt * t_opt) * np.sin(0.3 * t_opt) * np.sin(theta_opt) + X_opt
    y_pred = 42 + t_opt * np.sin(theta_opt) + np.exp(M_opt * t_opt) * np.sin(0.3 * t_opt) * np.cos(theta_opt)
    
    l1_x = np.mean(np.abs(x - x_pred))
    l1_y = np.mean(np.abs(y - y_pred))
    mean_dist = np.mean(np.sqrt((x - x_pred)**2 + (y - y_pred)**2))
    
    print(f"\nFinal Extracted Parameters:")
    print(f"  Theta (degrees)  : {theta_deg_opt:.6f}° (Exact: {round(theta_deg_opt):.1f}°)")
    print(f"  Theta (radians)  : {theta_opt:.6f} rad")
    print(f"  M                : {M_opt:.6f} (Exact: {M_opt:.2f})")
    print(f"  X                : {X_opt:.6f} (Exact: {X_opt:.1f})")
    print(f"\nFitting Performance:")
    print(f"  L1 error in X    : {l1_x:.8f}")
    print(f"  L1 error in Y    : {l1_y:.8f}")
    print(f"  Mean distance    : {mean_dist:.8f}")
    
    # Generate verification plot
    print("\nGenerating fitted_curve.png plot...")
    t_model = np.linspace(6, 60, 1000)
    x_model = t_model * np.cos(theta_opt) - np.exp(M_opt * t_model) * np.sin(0.3 * t_model) * np.sin(theta_opt) + X_opt
    y_model = 42 + t_model * np.sin(theta_opt) + np.exp(M_opt * t_model) * np.sin(0.3 * t_model) * np.cos(theta_opt)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    fig.patch.set_facecolor('#1E1E24')
    ax.set_facecolor('#1E1E24')
    
    ax.scatter(x, y, color='#3A86C8', alpha=0.5, s=15, label='Data Points (xy_data.csv)', zorder=2)
    ax.plot(x_model, y_model, color='#FF007F', linewidth=2.5, label=f'Fitted Curve (theta={round(theta_deg_opt)}°, M={M_opt:.2f}, X={X_opt:.1f})', zorder=3)
    
    ax.set_title("Parametric Curve Fitting Verification", color='white', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("X coordinate", color='#A0A0A0', fontsize=12, labelpad=10)
    ax.set_ylabel("Y coordinate", color='#A0A0A0', fontsize=12, labelpad=10)
    ax.grid(True, color='#2D2D35', linestyle='--', linewidth=0.5)
    
    legend = ax.legend(facecolor='#1E1E24', edgecolor='#2D2D35', fontsize=11, loc='upper left')
    for text in legend.get_texts():
        text.set_color('white')
        
    ax.tick_params(colors='#A0A0A0', labelsize=10)
    
    stats_text = (
        f"Optimal Parameters:\n"
        f"  $\\theta$ = {theta_deg_opt:.4f}°\n"
        f"  $M$ = {M_opt:.4f}\n"
        f"  $X$ = {X_opt:.4f}\n\n"
        f"Fitting Loss:\n"
        f"  Mean $L_1$ Error $\\approx$ {mean_dist:.6f}"
    )
    props = dict(boxstyle='round', facecolor='#2D2D35', alpha=0.8, edgecolor='#FF007F')
    ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, color='white', fontsize=11,
            verticalalignment='bottom', bbox=props)
            
    plt.tight_layout()
    plt.savefig("fitted_curve.png", facecolor=fig.get_facecolor(), edgecolor='none')
    print("Plot saved to fitted_curve.png successfully!")

if __name__ == '__main__':
    main()
