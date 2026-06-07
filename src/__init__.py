from src._types import ICFunc
from src.analytics import analytical_solution_sin, l2_error, max_error, relative_l2_error
from src.boundary import BCConfig, BCType, apply_boundary
from src.reaction import ReactionDiffusionSolver, ReactionResult, ReactionType
from src.solvers import (
    BaseSolver2D,
    CrankNicolson2D,
    ExplicitEuler2D,
    SolverResult,
    explicit_diffusion_step,
)
from src.visualization import animate_diffusion, plot_convergence, plot_heatmap

__all__ = [
    "ICFunc",
    "analytical_solution_sin",
    "l2_error",
    "max_error",
    "relative_l2_error",
    "BCConfig",
    "BCType",
    "apply_boundary",
    "ReactionDiffusionSolver",
    "ReactionResult",
    "ReactionType",
    "BaseSolver2D",
    "CrankNicolson2D",
    "ExplicitEuler2D",
    "SolverResult",
    "explicit_diffusion_step",
    "animate_diffusion",
    "plot_convergence",
    "plot_heatmap",
]
