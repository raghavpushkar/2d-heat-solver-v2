from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np


class BCType(Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"


def _bc_val(func: Callable[[float], float] | None, const_val: float, t: float) -> float:
    return func(t) if func else const_val


@dataclass
class BCConfig:
    """Boundary condition configuration for a 2-D rectangular domain.

    Each side (left/right/bottom/top) has a type and optional value or
    time-dependent function.  Factory methods provide common presets.
    """

    left: BCType = BCType.DIRICHLET
    right: BCType = BCType.DIRICHLET
    bottom: BCType = BCType.DIRICHLET
    top: BCType = BCType.DIRICHLET

    left_value: float = 0.0
    right_value: float = 0.0
    bottom_value: float = 0.0
    top_value: float = 0.0

    left_func: Callable[[float], float] | None = None
    right_func: Callable[[float], float] | None = None
    bottom_func: Callable[[float], float] | None = None
    top_func: Callable[[float], float] | None = None

    @staticmethod
    def dirichlet_all(value: float = 0.0) -> BCConfig:
        return BCConfig(
            left=BCType.DIRICHLET, right=BCType.DIRICHLET,
            bottom=BCType.DIRICHLET, top=BCType.DIRICHLET,
            left_value=value, right_value=value,
            bottom_value=value, top_value=value,
        )

    @staticmethod
    def neumann_all() -> BCConfig:
        return BCConfig(
            left=BCType.NEUMANN, right=BCType.NEUMANN,
            bottom=BCType.NEUMANN, top=BCType.NEUMANN,
        )

    @staticmethod
    def periodic_all() -> BCConfig:
        return BCConfig(
            left=BCType.PERIODIC, right=BCType.PERIODIC,
            bottom=BCType.PERIODIC, top=BCType.PERIODIC,
        )


def apply_boundary(u: np.ndarray, bc: BCConfig, t: float = 0.0) -> None:
    """Apply boundary conditions to *u* in-place."""
    # Left
    if bc.left == BCType.DIRICHLET:
        u[0, :] = _bc_val(bc.left_func, bc.left_value, t)
    elif bc.left == BCType.NEUMANN:
        u[0, :] = u[1, :]
    elif bc.left == BCType.PERIODIC:
        u[0, :] = u[-2, :]

    # Right
    if bc.right == BCType.DIRICHLET:
        u[-1, :] = _bc_val(bc.right_func, bc.right_value, t)
    elif bc.right == BCType.NEUMANN:
        u[-1, :] = u[-2, :]
    elif bc.right == BCType.PERIODIC:
        u[-1, :] = u[1, :]

    # Bottom
    if bc.bottom == BCType.DIRICHLET:
        u[:, 0] = _bc_val(bc.bottom_func, bc.bottom_value, t)
    elif bc.bottom == BCType.NEUMANN:
        u[:, 0] = u[:, 1]
    elif bc.bottom == BCType.PERIODIC:
        u[:, 0] = u[:, -2]

    # Top
    if bc.top == BCType.DIRICHLET:
        u[:, -1] = _bc_val(bc.top_func, bc.top_value, t)
    elif bc.top == BCType.NEUMANN:
        u[:, -1] = u[:, -2]
    elif bc.top == BCType.PERIODIC:
        u[:, -1] = u[:, 1]
