import numpy as np

from src.analytics import analytical_solution_sin, l2_error, relative_l2_error
from src.boundary import BCConfig, BCType, apply_boundary
from src.reaction import ReactionDiffusionSolver, ReactionType
from src.solvers import CrankNicolson2D, ExplicitEuler2D


def _circle_ic(cx: float, cy: float, radius: float):
    return lambda X, Y: (np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) < radius).astype(float)


# ============================================================
# Boundary Condition Tests
# ============================================================

class TestBoundary:
    def test_dirichlet_all_zero(self):
        u = np.ones((5, 5))
        bc = BCConfig.dirichlet_all(0.0)
        apply_boundary(u, bc)
        assert np.allclose(u[0, :], 0)
        assert np.allclose(u[-1, :], 0)
        assert np.allclose(u[:, 0], 0)
        assert np.allclose(u[:, -1], 0)
        assert np.allclose(u[1:-1, 1:-1], 1.0)

    def test_neumann_all(self):
        u = np.arange(25, dtype=float).reshape(5, 5)
        r = u.copy()
        bc = BCConfig.neumann_all()
        apply_boundary(u, bc)
        assert np.allclose(u[0, 1:-1], r[1, 1:-1])
        assert np.allclose(u[-1, 1:-1], r[-2, 1:-1])
        assert np.allclose(u[1:-1, 0], r[1:-1, 1])
        assert np.allclose(u[1:-1, -1], r[1:-1, -2])

    def test_mixed_bc(self):
        u = np.arange(25, dtype=float).reshape(5, 5)
        r = u.copy()
        bc = BCConfig(
            left=BCType.NEUMANN, right=BCType.DIRICHLET, right_value=42.0,
            bottom=BCType.NEUMANN, top=BCType.DIRICHLET, top_value=99.0,
        )
        apply_boundary(u, bc)
        assert np.allclose(u[0, 1:-1], r[1, 1:-1])
        assert np.allclose(u[-1, :-1], 42.0)
        assert np.allclose(u[1:-1, 0], r[1:-1, 1])
        assert np.allclose(u[:, -1], 99.0)

    def test_time_dependent_dirichlet(self):
        u = np.ones((5, 5))
        bc = BCConfig.dirichlet_all(0.0)
        bc.left_func = lambda t: t
        bc.right_func = lambda t: 2 * t
        bc.bottom_func = lambda t: 3 * t
        bc.top_func = lambda t: 4 * t
        apply_boundary(u, bc, t=2.0)
        assert np.allclose(u[0, 1:-1], 2.0)
        assert np.allclose(u[1:-1, -1], 8.0)

    def test_periodic_all(self):
        u = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ], dtype=float)
        bc = BCConfig.periodic_all()
        apply_boundary(u, bc)
        assert np.allclose(u[0, :], u[-2, :])
        assert np.allclose(u[-1, :], u[1, :])
        assert np.allclose(u[:, 0], u[:, -2])
        assert np.allclose(u[:, -1], u[:, 1])


# ============================================================
# Sinusoidal IC helper
# ============================================================

def sin_ic(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


# ============================================================
# Explicit Euler Solver Tests
# ============================================================

class TestExplicitEuler:
    def test_basic_solve(self):
        solver = ExplicitEuler2D(nx=30, ny=30, alpha=0.01)
        result = solver.solve(sin_ic, dt=0.001, t_total=0.05)
        assert result.u.shape == (30, 30)
        assert len(result.times) > 0
        assert result.u.max() <= 1.0

    def test_stable_within_cfl(self):
        solver = ExplicitEuler2D(nx=30, ny=30, alpha=0.01, bc=BCConfig.dirichlet_all(0.0))
        dt = 0.001  # well within CFL
        result = solver.solve(sin_ic, dt=dt, t_total=0.05)
        assert result.u.max() < 10, "Euler should be stable within CFL"

    def test_unstable_beyond_cfl(self):
        solver = ExplicitEuler2D(nx=30, ny=30, alpha=0.01, bc=BCConfig.dirichlet_all(0.0))

        def random_ic(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            np.random.seed(42)
            return np.random.random(X.shape) * 0.1

        dt = 0.045  # > CFL for 30x30 grid (CFL dt ~ 0.03)
        result = solver.solve(random_ic, dt=dt, t_total=0.5)
        assert result.u.max() > 10, "Euler should blow up beyond CFL with noisy IC"

    def test_gaussian_bc_zero(self):
        def gaussian_ic(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            return np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)

        solver = ExplicitEuler2D(nx=50, ny=50, alpha=0.01)
        result = solver.solve(gaussian_ic, dt=0.001, t_total=0.05)
        assert result.u[0, 0] == 0.0  # Dirichlet zero enforced
        assert result.u[-1, -1] == 0.0


# ============================================================
# Crank-Nicolson Solver Tests
# ============================================================

class TestCrankNicolson:
    def test_basic_solve(self):
        solver = CrankNicolson2D(nx=30, ny=30, alpha=0.01)
        result = solver.solve(sin_ic, dt=0.001, t_total=0.05)
        assert result.u.shape == (30, 30)
        assert len(result.times) > 0

    def test_stable_large_dt(self):
        solver = CrankNicolson2D(nx=30, ny=30, alpha=0.01)

        def random_ic(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            np.random.seed(42)
            return np.random.random(X.shape) * 0.1

        result = solver.solve(random_ic, dt=0.1, t_total=0.5)
        assert result.u.max() < 1.0, "CN should stay stable with large dt"

    def test_rebuild_on_dt_change(self):
        solver = CrankNicolson2D(nx=30, ny=30, alpha=0.01)
        r1 = solver.solve(sin_ic, dt=0.001, t_total=0.01)
        r2 = solver.solve(sin_ic, dt=0.002, t_total=0.01)
        assert r1.u.shape == r2.u.shape
        assert solver._dt_built == 0.002

    def test_accuracy_vs_analytical(self):
        bc = BCConfig.dirichlet_all(0.0)
        solver = CrankNicolson2D(nx=60, ny=60, Lx=1.0, Ly=1.0, alpha=0.01, bc=bc)
        result = solver.solve(sin_ic, dt=0.002, t_total=0.1)
        X, Y = np.meshgrid(result.grid_x, result.grid_y, indexing="ij")
        u_exact = analytical_solution_sin(X, Y, t=0.1, alpha=0.01)
        rel_err = relative_l2_error(result.u, u_exact)
        assert rel_err < 0.003, f"CN relative error {rel_err*100:.4f}% > 0.3%"

    def test_convergence_rate(self):
        bc = BCConfig.dirichlet_all(0.0)
        errors = []
        grids = [20, 40, 60]
        for nx in grids:
            solver = CrankNicolson2D(nx=nx, ny=nx, alpha=0.01, bc=bc)
            r = solver.solve(sin_ic, dt=0.002, t_total=0.1)
            X, Y = np.meshgrid(r.grid_x, r.grid_y, indexing="ij")
            u_ex = analytical_solution_sin(X, Y, t=0.1, alpha=0.01)
            errors.append(l2_error(r.u, u_ex))
        for i in range(1, len(errors)):
            ratio = errors[i - 1] / errors[i]
            assert ratio > 1.5, f"Grid {grids[i-1]}->{grids[i]}: error ratio {ratio:.2f}"

    def test_gaussian_ic(self):
        def gaussian_ic(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            return np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)

        solver = CrankNicolson2D(nx=50, ny=50, alpha=0.01)
        result = solver.solve(gaussian_ic, dt=0.001, t_total=0.1)
        assert result.u[0, 0] == 0.0
        assert result.u.max() <= 1.0

    def test_euler_cn_match_small_dt(self):
        bc = BCConfig.dirichlet_all(0.0)
        dt = 0.0005
        euler = ExplicitEuler2D(nx=40, ny=40, alpha=0.01, bc=bc)
        cn = CrankNicolson2D(nx=40, ny=40, alpha=0.01, bc=bc)
        r_e = euler.solve(sin_ic, dt=dt, t_total=0.05)
        r_cn = cn.solve(sin_ic, dt=dt, t_total=0.05)
        diff = np.max(np.abs(r_e.u - r_cn.u))
        assert diff < 0.001, f"Euler vs CN diff {diff:.6f} > 0.001 at small dt"

    def test_cn_neumann_conserves_heat(self):
        """Insulated (all-Neumann) box must conserve total heat to round-off.

        Heat is measured with trapezoidal weights, since boundary nodes
        represent half a control volume and corners a quarter.
        """
        bc = BCConfig.neumann_all()
        solver = CrankNicolson2D(nx=50, ny=50, alpha=0.05, bc=bc)

        def gaussian_ic(X, Y):
            return np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)

        def weighted_mass(u):
            w = np.ones_like(u)
            w[0, :] *= 0.5
            w[-1, :] *= 0.5
            w[:, 0] *= 0.5
            w[:, -1] *= 0.5
            return float((u * w).sum())

        u0 = solver._init_grid(gaussian_ic)
        r = solver.solve(gaussian_ic, dt=0.005, t_total=2.0)
        rel_drift = abs(weighted_mass(r.u) - weighted_mass(u0)) / weighted_mass(u0)
        assert rel_drift < 1e-6, f"Neumann heat drift {rel_drift*100:.6f}% too large"

    def test_cn_neumann_flattens_to_uniform(self):
        """With no flux out, the field should relax toward its (uniform) mean."""
        bc = BCConfig.neumann_all()
        solver = CrankNicolson2D(nx=40, ny=40, alpha=0.1, bc=bc)

        def gaussian_ic(X, Y):
            return np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)

        r = solver.solve(gaussian_ic, dt=0.01, t_total=10.0)
        assert r.u.max() - r.u.min() < 1e-2, "Neumann field should flatten out"

    def test_cn_dirichlet_unaffected_by_rewrite(self):
        """Dirichlet zero edges stay pinned at zero throughout the solve."""
        bc = BCConfig.dirichlet_all(0.0)
        solver = CrankNicolson2D(nx=30, ny=30, alpha=0.01, bc=bc)
        r = solver.solve(sin_ic, dt=0.001, t_total=0.05)
        assert np.allclose(r.u[0, :], 0.0)
        assert np.allclose(r.u[-1, :], 0.0)
        assert np.allclose(r.u[:, 0], 0.0)
        assert np.allclose(r.u[:, -1], 0.0)


# ============================================================
# Analytical Solution Tests
# ============================================================

class TestAnalytics:
    def test_analytical_initial_condition(self):
        X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10), indexing="ij")
        u0 = analytical_solution_sin(X, Y, t=0.0)
        expected = np.sin(np.pi * X) * np.sin(np.pi * Y)
        assert np.allclose(u0, expected)

    def test_analytical_decay(self):
        X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10), indexing="ij")
        u_t = analytical_solution_sin(X, Y, t=0.1)
        u_0 = analytical_solution_sin(X, Y, t=0.0)
        assert u_t.max() < u_0.max()
        assert u_t.min() >= 0

    def test_l2_error_identical(self):
        u = np.random.random((10, 10))
        assert l2_error(u, u) == 0.0

    def test_max_error(self):
        u = np.ones((5, 5))
        v = np.zeros((5, 5))
        from src.analytics import max_error
        assert max_error(u, v) == 1.0

    def test_relative_l2_error(self):
        u = np.ones((10, 10)) * 2.0
        v = np.ones((10, 10))
        err = relative_l2_error(u, v)
        assert err == 1.0

    def test_relative_l2_error_zero_denom(self):
        u = np.ones((5, 5))
        v = np.zeros((5, 5))
        err = relative_l2_error(u, v)
        assert err == float("inf")


# ============================================================
# Reaction-Diffusion Tests
# ============================================================

class TestReactionDiffusion:
    def test_fisher_kpp_basic(self):
        ic = _circle_ic(1.0, 1.0, 0.2)
        solver = ReactionDiffusionSolver(
            nx=40, ny=40, Lx=2.0, Ly=2.0, D=0.01, r=1.0,
        )
        result = solver.solve(ic, t_total=1.0, dt=0.01)
        u = result.u
        assert 0.0 <= u.min() <= u.max() <= 1.5
        assert result.v is None
        assert len(result.times) > 0

    def test_gray_scott_basic(self):
        ic_u = _circle_ic(1.0, 1.0, 0.15)

        def ic_v(X, Y):
            return (_circle_ic(1.0, 1.0, 0.15)(X, Y) * 0.25).astype(float)

        solver = ReactionDiffusionSolver(
            nx=50, ny=50, Lx=2.0, Ly=2.0,
            reaction_type=ReactionType.GRAY_SCOTT,
            F=0.04, k=0.06, Du=0.001, Dv=0.0005,
        )
        result = solver.solve(ic_u, t_total=5.0, dt=0.1, ic_func_v=ic_v)
        u, v = result.u, result.v
        assert v is not None
        assert u.min() >= 0.0 and u.max() <= 1.0
        assert v.min() >= 0.0 and v.max() <= 1.0
        assert len(result.times) > 0
        assert len(result.u_history) > 0
        assert result.v_history is not None and len(result.v_history) > 0

    def test_gray_scott_spatial_variation(self):
        ic_u = _circle_ic(0.5, 0.5, 0.1)

        def ic_v(X, Y):
            return (_circle_ic(0.5, 0.5, 0.1)(X, Y) * 0.25).astype(float)

        solver = ReactionDiffusionSolver(
            nx=50, ny=50, Lx=1.0, Ly=1.0,
            reaction_type=ReactionType.GRAY_SCOTT,
            F=0.03, k=0.06, Du=0.001, Dv=0.0005,
        )
        result = solver.solve(ic_u, t_total=3.0, dt=0.05, ic_func_v=ic_v)
        u_min, u_max = result.u.min(), result.u.max()
        assert u_max - u_min > 0.01, "Gray-Scott should develop spatial structure"

    def test_record_every(self):
        ic = _circle_ic(1.0, 1.0, 0.2)
        solver = ReactionDiffusionSolver(nx=30, ny=30, Lx=2.0, Ly=2.0, D=0.01, r=1.0)
        result = solver.solve(ic, t_total=1.0, dt=0.01, record_every=5)
        assert len(result.times) > 0
        assert len(result.u_history) == len(result.times)


# ============================================================
# Integration Tests
# ============================================================

def test_cn_error_under_03_percent():
    bc = BCConfig.dirichlet_all(0.0)
    solver = CrankNicolson2D(nx=100, ny=100, Lx=1.0, Ly=1.0, alpha=0.01, bc=bc)
    result = solver.solve(sin_ic, dt=0.001, t_total=0.1)
    X, Y = np.meshgrid(result.grid_x, result.grid_y, indexing="ij")
    u_exact = analytical_solution_sin(X, Y, t=0.1, alpha=0.01)
    rel_err = relative_l2_error(result.u, u_exact)
    assert rel_err < 0.003, f"CN error {rel_err*100:.4f}% >= 0.3% on 100x100 grid"
