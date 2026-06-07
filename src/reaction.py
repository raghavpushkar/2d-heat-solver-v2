from dataclasses import dataclass
from enum import Enum

import numpy as np

from src._types import ICFunc
from src.boundary import BCConfig, apply_boundary
from src.solvers import explicit_diffusion_step

DEFAULT_V_INIT = 0.5


class ReactionType(Enum):
    FISHER_KPP = "fisher_kpp"
    GRAY_SCOTT = "gray_scott"


@dataclass
class ReactionResult:
    """Result container for a reaction-diffusion solve.

    Attributes:
        u: Final U-concentration array (nx, ny).
        u_history: U-concentration snapshots over time.
        times: Recorded time values.
        grid_x: 1-D x-coordinate array.
        grid_y: 1-D y-coordinate array.
        v: Final V-concentration array (Gray-Scott only).
        v_history: V-concentration snapshots (Gray-Scott only).
    """

    u: np.ndarray
    u_history: list[np.ndarray]
    times: list[float]
    grid_x: np.ndarray
    grid_y: np.ndarray
    v: np.ndarray | None = None
    v_history: list[np.ndarray] | None = None


def reaction_fisher_kpp(u: np.ndarray, r: float, dt: float) -> np.ndarray:
    """Single Fisher-KPP reaction step, returning new array."""
    return u + dt * r * u * (1.0 - u)


def reaction_gray_scott(
    u: np.ndarray, v: np.ndarray, F: float, k: float, dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Single Gray-Scott reaction step, returning new arrays."""
    uvv = u * v * v
    u_new = u + dt * (-uvv + F * (1.0 - u))
    v_new = v + dt * (uvv - (F + k) * v)
    u_new = np.clip(u_new, 0.0, 1.0)
    v_new = np.clip(v_new, 0.0, 1.0)
    return u_new, v_new


class ReactionDiffusionSolver:
    """Strang-split reaction-diffusion solver.

    Supports Fisher-KPP (scalar) and Gray-Scott (two-species) models
    with explicit diffusion and explicit reaction sub-steps.

    Parameters
    ----------
    nx, ny:
        Grid points in x and y.
    Lx, Ly:
        Domain lengths.
    D:
        Default diffusion coefficient (Fisher-KPP single species).
    reaction_type:
        ``FISHER_KPP`` or ``GRAY_SCOTT``.
    r:
        Fisher-KPP growth rate.
    F:
        Gray-Scott feed rate.
    k:
        Gray-Scott kill rate.
    Du:
        Gray-Scott U diffusivity.
    Dv:
        Gray-Scott V diffusivity.
    v_init:
        Default V initial concentration for Gray-Scott.
    bc:
        Boundary condition configuration.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        Lx: float = 1.0,
        Ly: float = 1.0,
        D: float = 0.001,
        reaction_type: ReactionType = ReactionType.FISHER_KPP,
        r: float = 1.0,
        F: float = 0.04,
        k: float = 0.06,
        Du: float = 0.001,
        Dv: float = 0.0005,
        v_init: float = DEFAULT_V_INIT,
        bc: BCConfig | None = None,
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.D = D
        self.reaction_type = reaction_type
        self.r = r
        self.F = F
        self.k = k
        self.Du = Du
        self.Dv = Dv
        self.v_init = v_init
        self.bc = bc or BCConfig.neumann_all()

        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)

    def solve(
        self,
        ic_func_u: ICFunc,
        t_total: float,
        dt: float = 0.01,
        ic_func_v: ICFunc | None = None,
        record_every: int | None = None,
    ) -> ReactionResult:
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        u = ic_func_u(X, Y).astype(float)

        n_steps = int(np.ceil(t_total / dt))
        rec = record_every or max(1, n_steps // 100)

        v: np.ndarray | None = None
        if self.reaction_type == ReactionType.GRAY_SCOTT:
            v = (
                np.full_like(u, self.v_init)
                if ic_func_v is None
                else ic_func_v(X, Y).astype(float)
            )

        results_u: list[np.ndarray] = []
        results_v: list[np.ndarray] = []
        times: list[float] = []

        half = dt / 2.0
        rx = ry = rx_u = ry_u = rx_v = ry_v = 0.0

        if self.reaction_type == ReactionType.FISHER_KPP:
            rx = self.D * half / self.dx**2
            ry = self.D * half / self.dy**2
        else:
            rx_u = self.Du * half / self.dx**2
            ry_u = self.Du * half / self.dy**2
            rx_v = self.Dv * half / self.dx**2
            ry_v = self.Dv * half / self.dy**2

        for step in range(n_steps):
            t = (step + 1) * dt

            if self.reaction_type == ReactionType.FISHER_KPP:
                explicit_diffusion_step(u, rx, ry)
                apply_boundary(u, self.bc, t=t)
                u = reaction_fisher_kpp(u, self.r, dt)
                explicit_diffusion_step(u, rx, ry)
                apply_boundary(u, self.bc, t=t)

            elif self.reaction_type == ReactionType.GRAY_SCOTT:
                if v is None:
                    raise ValueError("v must be initialised for Gray-Scott")
                explicit_diffusion_step(u, rx_u, ry_u)
                apply_boundary(u, self.bc, t=t)
                explicit_diffusion_step(v, rx_v, ry_v)
                apply_boundary(v, self.bc, t=t)

                u, v = reaction_gray_scott(u, v, self.F, self.k, dt)

                explicit_diffusion_step(u, rx_u, ry_u)
                apply_boundary(u, self.bc, t=t)
                explicit_diffusion_step(v, rx_v, ry_v)
                apply_boundary(v, self.bc, t=t)

            if step % rec == 0 or step == n_steps - 1:
                results_u.append(u.copy())
                if v is not None:
                    results_v.append(v.copy())
                times.append(t)

        return ReactionResult(
            u=results_u[-1] if results_u else u,
            u_history=results_u,
            v=results_v[-1] if results_v else None,
            v_history=results_v if results_v else None,
            times=times,
            grid_x=self.x,
            grid_y=self.y,
        )
