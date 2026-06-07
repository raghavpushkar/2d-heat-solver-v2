import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src._types import ICFunc
from src.boundary import BCConfig, BCType, apply_boundary


def explicit_diffusion_step(u: np.ndarray, rx: float, ry: float) -> None:
    """Apply one FTCS diffusion step, mutating *u* in-place."""
    un = u.copy()
    u[1:-1, 1:-1] = (
        un[1:-1, 1:-1]
        + rx * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])
        + ry * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
    )


@dataclass
class SolverResult:
    """Result container for a 2-D diffusion solve.

    Attributes:
        u: Final solution array (nx, ny).
        times: Recorded time values.
        u_history: Solution snapshots over time.
        grid_x: 1-D x-coordinate array.
        grid_y: 1-D y-coordinate array.
        dt: Time step size used.
        n_steps: Total number of time steps taken.
    """

    u: np.ndarray
    times: list[float]
    u_history: list[np.ndarray]
    grid_x: np.ndarray
    grid_y: np.ndarray
    dt: float
    n_steps: int


class BaseSolver2D(ABC):
    """Abstract base for 2-D diffusion solvers.

    Provides grid setup, initial-condition evaluation, boundary application,
    and a shared record-every helper.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        Lx: float = 1.0,
        Ly: float = 1.0,
        alpha: float = 0.01,
        bc: BCConfig | None = None,
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.alpha = alpha
        self.bc = bc or BCConfig.dirichlet_all(0.0)

        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)

    def _init_grid(self, ic_func: ICFunc) -> np.ndarray:
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        u = ic_func(X, Y)
        apply_boundary(u, self.bc, t=0.0)
        return u

    def _compute_rec(self, n_steps: int, record_every: int | None) -> int:
        return record_every or max(1, n_steps // 100)

    @abstractmethod
    def solve(
        self,
        ic_func: ICFunc,
        dt: float,
        t_total: float,
        record_every: int | None = None,
    ) -> SolverResult: ...


class ExplicitEuler2D(BaseSolver2D):
    """Explicit Euler (FTCS) solver — conditionally stable (CFL ≤ 0.5)."""

    def solve(
        self,
        ic_func: ICFunc,
        dt: float,
        t_total: float,
        record_every: int | None = None,
    ) -> SolverResult:
        rx = self.alpha * dt / self.dx**2
        ry = self.alpha * dt / self.dy**2

        if rx + ry > 0.5:
            warnings.warn(
                f"CFL condition violated: rx+ry={rx+ry:.4f} > 0.5. "
                "Euler FTCS may become unstable.",
                stacklevel=2,
            )

        n_steps = int(np.ceil(t_total / dt))
        rec = self._compute_rec(n_steps, record_every)

        u = self._init_grid(ic_func)
        results: list[np.ndarray] = []
        times: list[float] = []

        for step in range(n_steps):
            t = (step + 1) * dt
            explicit_diffusion_step(u, rx, ry)
            apply_boundary(u, self.bc, t=t)

            if step % rec == 0 or step == n_steps - 1:
                results.append(u.copy())
                times.append(t)

        return SolverResult(
            u=u,
            times=times,
            u_history=results,
            grid_x=self.x,
            grid_y=self.y,
            dt=dt,
            n_steps=n_steps,
        )


class CrankNicolson2D(BaseSolver2D):
    """Crank-Nicolson (implicit) solver — unconditionally stable, O(Δt²+Δx²)."""

    def __init__(
        self,
        nx: int,
        ny: int,
        Lx: float = 1.0,
        Ly: float = 1.0,
        alpha: float = 0.01,
        bc: BCConfig | None = None,
    ):
        super().__init__(nx, ny, Lx, Ly, alpha, bc)
        self._built = False
        self._dt_built: float | None = None
        self._A: sp.csr_matrix | None = None
        self._B: sp.csr_matrix | None = None
        self._b_rows: np.ndarray | None = None

    def _build_system(self, dt: float) -> None:
        """Assemble the full-grid Crank-Nicolson system A u^{n+1} = B u^n + b.

        Every node (interior *and* boundary) gets an explicit equation. Interior
        nodes use the standard CN stencil. Boundary nodes encode their boundary
        condition directly in the matrix, so the BC is enforced *inside* the
        implicit solve rather than patched on afterwards:

        - Dirichlet: A row is identity; value supplied each step via the RHS vector.
        - Neumann (zero-flux): ghost-node method — the ghost point outside the
          wall mirrors the interior neighbour, preserving the second-order CN
          stencil and conserving total heat to round-off.
        - Periodic: opposite edges are tied together.
        """
        nx, ny = self.nx, self.ny
        rx = self.alpha * dt / self.dx**2
        ry = self.alpha * dt / self.dy**2
        N = nx * ny

        a_rows: list[int] = []
        a_cols: list[int] = []
        a_data: list[float] = []
        b_rows: list[int] = []
        b_cols: list[int] = []
        b_data: list[float] = []

        a_diag = 1.0 + rx + ry
        a_off_x = -rx / 2.0
        a_off_y = -ry / 2.0
        b_diag = 1.0 - rx - ry
        b_off_x = rx / 2.0
        b_off_y = ry / 2.0

        bc = self.bc

        def idx(i: int, j: int) -> int:
            return i * ny + j

        for i in range(nx):
            for j in range(ny):
                k = idx(i, j)
                on_left = i == 0
                on_right = i == nx - 1
                on_bottom = j == 0
                on_top = j == ny - 1

                if not (on_left or on_right or on_bottom or on_top):
                    # Interior node: standard Crank-Nicolson stencil.
                    a_rows += [k, k, k, k, k]
                    a_cols += [k, k - ny, k + ny, k - 1, k + 1]
                    a_data += [a_diag, a_off_x, a_off_x, a_off_y, a_off_y]
                    b_rows += [k, k, k, k, k]
                    b_cols += [k, k - ny, k + ny, k - 1, k + 1]
                    b_data += [b_diag, b_off_x, b_off_x, b_off_y, b_off_y]
                    continue

                # Pure-Dirichlet boundary nodes are pinned to their value.
                # (A Dirichlet side overrides any other classification.)
                dirichlet_here = (
                    (on_left and bc.left == BCType.DIRICHLET)
                    or (on_right and bc.right == BCType.DIRICHLET)
                    or (on_bottom and bc.bottom == BCType.DIRICHLET)
                    or (on_top and bc.top == BCType.DIRICHLET)
                )
                if dirichlet_here:
                    a_rows.append(k)
                    a_cols.append(k)
                    a_data.append(1.0)
                    continue

                periodic_here = (
                    (on_left and bc.left == BCType.PERIODIC)
                    or (on_right and bc.right == BCType.PERIODIC)
                    or (on_bottom and bc.bottom == BCType.PERIODIC)
                    or (on_top and bc.top == BCType.PERIODIC)
                )
                if periodic_here:
                    # Tie this edge to the opposite interior row.
                    if on_left:
                        opp = idx(nx - 2, j)
                    elif on_right:
                        opp = idx(1, j)
                    elif on_bottom:
                        opp = idx(i, ny - 2)
                    else:  # on_top
                        opp = idx(i, 1)
                    a_rows += [k, k]
                    a_cols += [k, opp]
                    a_data += [1.0, -1.0]
                    continue

                # Otherwise: zero-flux Neumann on one or more sides.
                # Use the ghost-node method: a ghost point outside each Neumann
                # edge equals its mirror interior point, so the second-order CN
                # stencil is preserved and the scheme stays mass-conservative.
                # Build the stencil with mirrored neighbours where a wall sits.
                xm = idx(i - 1, j) if i > 0 else idx(i + 1, j)
                xp = idx(i + 1, j) if i < nx - 1 else idx(i - 1, j)
                ym = idx(i, j - 1) if j > 0 else idx(i, j + 1)
                yp = idx(i, j + 1) if j < ny - 1 else idx(i, j - 1)

                a_rows += [k, k, k, k, k]
                a_cols += [k, xm, xp, ym, yp]
                a_data += [a_diag, a_off_x, a_off_x, a_off_y, a_off_y]
                b_rows += [k, k, k, k, k]
                b_cols += [k, xm, xp, ym, yp]
                b_data += [b_diag, b_off_x, b_off_x, b_off_y, b_off_y]

        self._A = sp.csr_matrix((a_data, (a_rows, a_cols)), shape=(N, N))
        self._B = sp.csr_matrix((b_data, (b_rows, b_cols)), shape=(N, N))
        # Precompute which flat indices are Dirichlet boundary rows so we can
        # write their prescribed values into the RHS vector each step.
        self._b_rows = self._dirichlet_rows()
        self._built = True

    def _dirichlet_rows(self) -> np.ndarray:
        nx, ny = self.nx, self.ny
        bc = self.bc
        rows: list[int] = []
        for i in range(nx):
            for j in range(ny):
                on_left, on_right = i == 0, i == nx - 1
                on_bottom, on_top = j == 0, j == ny - 1
                if not (on_left or on_right or on_bottom or on_top):
                    continue
                if on_left:
                    side = bc.left
                elif on_right:
                    side = bc.right
                elif on_bottom:
                    side = bc.bottom
                else:
                    side = bc.top
                if side == BCType.DIRICHLET:
                    rows.append(i * ny + j)
        return np.array(rows, dtype=int)

    def _dirichlet_rhs(self, t: float) -> np.ndarray:
        """Prescribed Dirichlet values laid out on the full flat grid."""
        nx, ny = self.nx, self.ny
        rhs = np.zeros(nx * ny)
        scratch = np.zeros((nx, ny))
        apply_boundary(scratch, self.bc, t=t)
        flat = scratch.ravel()
        if self._b_rows is not None and self._b_rows.size:
            rhs[self._b_rows] = flat[self._b_rows]
        return rhs

    def solve(
        self,
        ic_func: ICFunc,
        dt: float,
        t_total: float,
        record_every: int | None = None,
    ) -> SolverResult:
        if not self._built or dt != self._dt_built:
            self._build_system(dt)
            self._dt_built = dt

        assert self._A is not None and self._B is not None

        n_steps = int(np.ceil(t_total / dt))
        rec = self._compute_rec(n_steps, record_every)

        u = self._init_grid(ic_func)
        results: list[np.ndarray] = []
        times: list[float] = []

        for step in range(n_steps):
            t = (step + 1) * dt

            rhs = self._B @ u.ravel()
            # Overwrite boundary rows: Dirichlet rows carry their prescribed
            # value; Neumann/periodic rows are homogeneous (zero), which the
            # B matrix already produces because those rows are empty.
            bvec = self._dirichlet_rhs(t)
            if self._b_rows is not None and self._b_rows.size:
                rhs[self._b_rows] = bvec[self._b_rows]
            u_new = spla.spsolve(self._A, rhs)
            u = u_new.reshape(self.nx, self.ny)

            if step % rec == 0 or step == n_steps - 1:
                results.append(u.copy())
                times.append(t)

        return SolverResult(
            u=u,
            times=times,
            u_history=results,
            grid_x=self.x,
            grid_y=self.y,
            dt=dt,
            n_steps=n_steps,
        )
