"""Optimizer package exports for parametric curve fitting."""

from optimizer.config import BOUNDS, ParameterBounds
from optimizer.model import (
    parametric_x,
    parametric_y,
    project_to_t,
    project_to_z_actual,
    project_to_z_expected,
    reconstruct_curve_points,
    NumericArray
)
from optimizer.solver import CurveFitter
from optimizer.visualization import generate_curve_plot

__all__ = [
    "BOUNDS",
    "ParameterBounds",
    "parametric_x",
    "parametric_y",
    "project_to_t",
    "project_to_z_actual",
    "project_to_z_expected",
    "reconstruct_curve_points",
    "NumericArray",
    "CurveFitter",
    "generate_curve_plot"
]
__version__ = "1.0.0"
