import numpy as np


def analytical_solution_sin(
    X: np.ndarray,
    Y: np.ndarray,
    t: float,
    alpha: float = 0.01,
    Lx: float = 1.0,
    Ly: float = 1.0,
) -> np.ndarray:
    """Closed-form solution for sin(πx/Lx) sin(πy/Ly) IC on Dirichlet 0 BC."""
    kx = np.pi / Lx
    ky = np.pi / Ly
    lam = alpha * (kx**2 + ky**2)
    return np.sin(kx * X) * np.sin(ky * Y) * np.exp(-lam * t)


def l2_error(u_numerical: np.ndarray, u_analytical: np.ndarray) -> float:
    """Root-mean-square error (RMSE) between two fields."""
    diff = u_numerical - u_analytical
    return float(np.sqrt(np.mean(diff**2)))


def max_error(u_numerical: np.ndarray, u_analytical: np.ndarray) -> float:
    """Maximum absolute pointwise error."""
    return float(np.max(np.abs(u_numerical - u_analytical)))


def relative_l2_error(u_numerical: np.ndarray, u_analytical: np.ndarray) -> float:
    """Relative ℓ² error; returns 0 if both are zero, inf if only reference is zero."""
    diff = u_numerical - u_analytical
    denom = np.sqrt(np.sum(u_analytical**2))
    if denom == 0.0:
        return 0.0 if np.all(diff == 0.0) else float("inf")
    return float(np.sqrt(np.sum(diff**2)) / denom)
