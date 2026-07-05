"""Command Line Interface (CLI) implementation for running parameter extraction and compiling output."""
import argparse
import sys
import os
import json
import logging
import pandas as pd
import numpy as np
from optimizer.solver import CurveFitter
from optimizer.visualization import generate_curve_plot

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("optimizer.cli")

def compile_dashboard_data(csv_path: str, output_js_path: str) -> None:
    """Read coordinate CSV file and compile it to a JSON/JavaScript variable for serverless browser execution.
    
    Args:
        csv_path: Path to the input xy_data.csv.
        output_js_path: Path to write the dashboard data.js.
    """
    logger.info("Compiling CSV data for browser dashboard integration...")
    if not os.path.exists(csv_path):
        logger.error("Input CSV file '%s' not found.", csv_path)
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    if 'x' not in df.columns or 'y' not in df.columns:
        logger.error("CSV file must contain 'x' and 'y' columns.")
        sys.exit(1)
        
    # Standardize data to basic list of dicts
    points = [{"x": float(row.x), "y": float(row.y)} for row in df.itertuples()]
    
    # Format as a JavaScript constant declaration
    js_content = f"// Automatically generated coordinate dataset\nconst COORDINATES_DATA = {json.dumps(points, indent=2)};\n"
    
    os.makedirs(os.path.dirname(output_js_path), exist_ok=True)
    with open(output_js_path, 'w', encoding='utf-8') as f:
        f.write(js_content)
        
    logger.info("Successfully compiled %d data points to '%s'.", len(points), output_js_path)

def main() -> None:
    """Main CLI entrypoint execution logic."""
    parser = argparse.ArgumentParser(
        description="R&D / AI Parametric Curve Fitting Optimization & Dashboard Compiler."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="xy_data.csv",
        help="Path to the input xy coordinates CSV file (default: xy_data.csv)"
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Run the parameter optimization solver."
    )
    parser.add_argument(
        "--plot",
        type=str,
        nargs="?",
        const="fitted_curve.png",
        help="Generate and save verification curve plot to the specified path (default: fitted_curve.png)"
    )
    parser.add_argument(
        "--light-theme",
        action="store_true",
        help="Use light theme for generated plot (default: dark theme)"
    )
    parser.add_argument(
        "--compile-dashboard",
        type=str,
        nargs="?",
        const=os.path.join("dashboard", "data.js"),
        help="Compile CSV dataset to a JavaScript data file for local dashboard rendering (default: dashboard/data.js)"
    )
    
    args = parser.parse_args()
    
    # If no action arguments are provided, print help
    if not (args.fit or args.plot or args.compile_dashboard):
        parser.print_help()
        sys.exit(0)
        
    # Read the data if needed
    x_data, y_data = None, None
    if args.fit or args.plot:
        if not os.path.exists(args.csv):
            logger.error("Dataset CSV file '%s' not found. Please provide path via --csv.", args.csv)
            sys.exit(1)
        try:
            df = pd.read_csv(args.csv)
            x_data = df['x'].values
            y_data = df['y'].values
        except Exception as e:
            logger.error("Failed to read CSV dataset: %s", str(e))
            sys.exit(1)
            
    fit_results = None
    if args.fit or args.plot:
        fitter = CurveFitter(x_data, y_data)
        fit_results = fitter.fit()
        
        # Display optimization parameters
        print("\n" + "="*50)
        print("           OPTIMIZATION RESULTS SUMMARY")
        print("="*50)
        print(f"Status           : {'CONVERGED' if fit_results['success'] else 'FAILED'}")
        print(f"Message          : {fit_results['message']}")
        print(f"Theta (Degrees)  : {fit_results['theta_deg']:.6f}° (Exact: ~30.0°)")
        print(f"Theta (Radians)  : {fit_results['theta_rad']:.6f} rad")
        print(f"M (decay rate)   : {fit_results['M']:.6f} (Exact: ~0.03)")
        print(f"X (horizontal sh): {fit_results['X']:.6f} (Exact: ~55.0)")
        print("-"*50)
        print(f"Mean L1 X error  : {fit_results['l1_error_x']:.8f}")
        print(f"Mean L1 Y error  : {fit_results['l1_error_y']:.8f}")
        print(f"Mean Euclidean   : {fit_results['mean_euclidean_error']:.8f}")
        print(f"Max Euclidean    : {fit_results['max_euclidean_error']:.8f}")
        print("="*50 + "\n")
        
    if args.plot and fit_results:
        plot_path = args.plot
        dark_mode = not args.light_theme
        logger.info("Generating verification plot...")
        try:
            generate_curve_plot(x_data, y_data, fit_results, plot_path, dark_theme=dark_mode)
            logger.info("Verification plot saved successfully to '%s'.", plot_path)
        except Exception as e:
            logger.error("Failed to generate plot: %s", str(e))
            
    if args.compile_dashboard:
        compile_dashboard_data(args.csv, args.compile_dashboard)

if __name__ == '__main__':
    main()
