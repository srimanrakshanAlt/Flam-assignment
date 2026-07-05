"""Visualization engine to generate beautiful, publication-quality plots of fitted curves."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from optimizer.model import parametric_x, parametric_y

def generate_curve_plot(
    x_data: np.ndarray,
    y_data: np.ndarray,
    fit_results: Dict[str, Any],
    output_path: str = "fitted_curve.png",
    dark_theme: bool = True
) -> None:
    """Generate and save a beautiful dark or light themed verification plot.
    
    Args:
        x_data: Original X coordinates.
        y_data: Original Y coordinates.
        fit_results: Dictionary containing optimized parameter results from CurveFitter.
        output_path: Filepath where the plot will be saved.
        dark_theme: Boolean indicating if a premium dark theme should be applied.
    """
    theta_deg = fit_results["theta_deg"]
    theta_rad = fit_results["theta_rad"]
    M = fit_results["M"]
    X = fit_results["X"]
    mean_err = fit_results["mean_euclidean_error"]
    
    # Generate model prediction curve with high resolution
    t_model = np.linspace(6.0, 60.0, 1000)
    x_model = parametric_x(t_model, theta_rad, M, X)
    y_model = parametric_y(t_model, theta_rad, M)
    
    # Configure style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    
    if dark_theme:
        bg_color = '#0F172A'       # Tailwind Slate 900
        card_color = '#1E293B'     # Tailwind Slate 800
        text_color = '#F8FAFC'     # Tailwind Slate 50
        grid_color = '#334155'     # Tailwind Slate 700
        accent_color = '#38BDF8'   # Tailwind Sky 400 (Data)
        line_color = '#F43F5E'     # Tailwind Rose 500 (Fitted)
        shadow_color = '#E2E8F0'
    else:
        bg_color = '#FFFFFF'
        card_color = '#F8FAFC'
        text_color = '#0F172A'
        grid_color = '#E2E8F0'
        accent_color = '#0284C7'
        line_color = '#E11D48'
        shadow_color = '#334155'
        
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Plot raw data points
    ax.scatter(
        x_data,
        y_data,
        color=accent_color,
        alpha=0.55,
        s=12,
        label='Observed Coordinates (xy_data.csv)',
        zorder=2
    )
    
    # Plot predicted parametric curve
    ax.plot(
        x_model,
        y_model,
        color=line_color,
        linewidth=2.5,
        label=f'Fitted Model ($\\theta$={theta_deg:.2f}°, M={M:.3f}, X={X:.1f})',
        zorder=3
    )
    
    # Titles & Labels
    ax.set_title("Parametric Curve Fitting & Variable Extraction", color=text_color, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("X Coordinate", color='#94A3B8' if dark_theme else '#475569', fontsize=12, labelpad=10)
    ax.set_ylabel("Y Coordinate", color='#94A3B8' if dark_theme else '#475569', fontsize=12, labelpad=10)
    
    # Grid configuration
    ax.grid(True, color=grid_color, linestyle='--', linewidth=0.5)
    
    # Legend styling
    legend = ax.legend(
        facecolor=card_color,
        edgecolor=grid_color,
        fontsize=11,
        loc='upper left',
        framealpha=0.9
    )
    for text in legend.get_texts():
        text.set_color(text_color)
        
    # Tick colors
    ax.tick_params(colors='#94A3B8' if dark_theme else '#475569', labelsize=10)
    
    # Infobox displaying details
    info_text = (
        f"Optimal Parameters:\n"
        f"  $\\theta$ = {theta_deg:.4f}°\n"
        f"  $M$ = {M:.6f}\n"
        f"  $X$ = {X:.4f}\n\n"
        f"Model Metrics:\n"
        f"  Mean $L_1$ Euclidean Err: {mean_err:.6e}"
    )
    props = dict(boxstyle='round,pad=0.8', facecolor=card_color, alpha=0.9, edgecolor=grid_color)
    ax.text(
        0.05, 0.05,
        info_text,
        transform=ax.transAxes,
        color=text_color,
        fontsize=11,
        fontfamily='monospace',
        verticalalignment='bottom',
        bbox=props
    )
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=bg_color, edgecolor='none')
    plt.close()
