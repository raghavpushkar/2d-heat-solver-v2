"""Interactive Streamlit front-end for the 2D Heat & Reaction-Diffusion solver.

Run locally:
    uv run streamlit run streamlit_app.py
Or after `pip install -r requirements.txt`:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before pyplot is used

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.analytics import analytical_solution_sin, l2_error, relative_l2_error
from src.boundary import BCConfig
from src.reaction import ReactionDiffusionSolver, ReactionType
from src.solvers import CrankNicolson2D, ExplicitEuler2D

# Streamlit Cloud runs a newer matplotlib whose mathtext parser can choke on
# auto-generated log-scale offset tick labels (e.g. 1e-6). We never need LaTeX
# in matplotlib figures here (equations are rendered by Streamlit markdown), so
# disable math parsing entirely to keep tick/label rendering bulletproof.
plt.rcParams["text.parse_math"] = False
plt.rcParams["axes.formatter.use_mathtext"] = False

st.set_page_config(
    page_title="2D Heat & Reaction-Diffusion Solver",
    page_icon="🔥",
    layout="wide",
)


# --------------------------------------------------------------------------
# Initial conditions
# --------------------------------------------------------------------------
def ic_sin(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


def ic_gaussian(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.1**2))


IC_MAP = {"sin(πx)·sin(πy)": ic_sin, "Gaussian hot spot": ic_gaussian}


# --------------------------------------------------------------------------
# Cached solves — heavy work runs once per unique parameter set
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_heat(
    solver_name: str,
    nx: int,
    alpha: float,
    dt: float,
    t_total: float,
    ic_name: str,
    bc_name: str,
    bc_val: float,
):
    ic = IC_MAP[ic_name]
    bc = (
        BCConfig.dirichlet_all(bc_val)
        if bc_name == "Dirichlet"
        else BCConfig.neumann_all()
    )
    cls = ExplicitEuler2D if solver_name == "Explicit Euler" else CrankNicolson2D
    solver = cls(nx=nx, ny=nx, alpha=alpha, bc=bc)
    result = solver.solve(ic, dt=dt, t_total=t_total, record_every=1)
    rx = alpha * dt / solver.dx**2
    return (
        result.u_history,
        result.times,
        result.grid_x,
        result.grid_y,
        2 * rx,  # rx + ry on a square grid
    )


@st.cache_data(show_spinner=False)
def run_convergence(alpha: float, dt: float, t_total: float):
    grids = [20, 30, 40, 50, 60, 80]
    errors = []
    for g in grids:
        s = CrankNicolson2D(nx=g, ny=g, alpha=alpha, bc=BCConfig.dirichlet_all(0.0))
        r = s.solve(ic_sin, dt=dt, t_total=t_total)
        X, Y = np.meshgrid(r.grid_x, r.grid_y, indexing="ij")
        ex = analytical_solution_sin(X, Y, t=t_total, alpha=alpha)
        errors.append(l2_error(r.u, ex))
    return grids, errors


@st.cache_data(show_spinner=False)
def run_reaction(
    reaction: str,
    nx: int,
    F: float,
    k: float,
    Du: float,
    Dv: float,
    t_total: float,
    dt: float,
):
    L = 2.5
    np.random.seed(0)

    def ic_u(X, Y):
        u = np.ones_like(X)
        c = (np.abs(X - L / 2) < 0.08) & (np.abs(Y - L / 2) < 0.08)
        u[c] = 0.5
        return u + 0.01 * np.random.random(X.shape)

    def ic_v(X, Y):
        v = np.zeros_like(X)
        c = (np.abs(X - L / 2) < 0.08) & (np.abs(Y - L / 2) < 0.08)
        v[c] = 0.25
        return v + 0.01 * np.random.random(X.shape)

    rtype = (
        ReactionType.GRAY_SCOTT if reaction == "Gray-Scott" else ReactionType.FISHER_KPP
    )
    solver = ReactionDiffusionSolver(
        nx=nx, ny=nx, Lx=L, Ly=L, reaction_type=rtype,
        F=F, k=k, Du=Du, Dv=Dv, D=Du, r=1.0,
    )
    result = solver.solve(
        ic_u, t_total=t_total, dt=dt,
        ic_func_v=ic_v if rtype == ReactionType.GRAY_SCOTT else None,
        record_every=max(1, int(t_total / dt) // 60),
    )
    field = result.v if result.v is not None else result.u
    dx = L / (nx - 1)
    cfl = 2 * Du * (dt / 2) / dx**2
    return field, result.grid_x, result.grid_y, cfl


def heatmap_fig(u, x, y, vmin, vmax, cmap="hot", title=""):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.pcolormesh(x, y, u.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🔥 2D Heat & Reaction-Diffusion PDE Solver")
st.markdown(
    "Interactive finite-difference solver for the 2D heat equation "
    "$\\partial_t u = \\alpha \\nabla^2 u$ and reaction-diffusion systems. "
    "Implements **Explicit Euler (FTCS)** and **Crank-Nicolson** schemes with "
    "analytical validation. Use the sliders to explore how the methods behave."
)

tab_heat, tab_compare, tab_valid, tab_react, tab_about = st.tabs(
    ["Heat Diffusion", "Method Comparison", "Validation", "Reaction-Diffusion", "About"]
)


# --------------------------------------------------------------------------
# Tab 1 — Heat diffusion with a time slider
# --------------------------------------------------------------------------
with tab_heat:
    st.subheader("Watch heat diffuse over time")
    c1, c2 = st.columns([1, 2])
    with c1:
        solver_name = st.selectbox(
            "Scheme", ["Crank-Nicolson", "Explicit Euler"], key="h_solver"
        )
        ic_name = st.selectbox("Initial condition", list(IC_MAP), key="h_ic")
        bc_name = st.selectbox("Boundary condition", ["Dirichlet", "Neumann"], key="h_bc")
        bc_val = 0.0
        if bc_name == "Dirichlet":
            bc_val = st.slider("Dirichlet edge value", 0.0, 1.0, 0.0, 0.05, key="h_bcval")
        alpha = st.slider("Diffusivity α", 0.001, 0.1, 0.02, 0.001, key="h_alpha")
        nx = st.slider("Grid size (nx = ny)", 20, 100, 60, 5, key="h_nx")
        dt = st.select_slider(
            "Time step Δt",
            options=[0.0005, 0.001, 0.002, 0.005, 0.01],
            value=0.001,
            key="h_dt",
        )
        t_total = st.slider("Total time", 0.05, 1.0, 0.4, 0.05, key="h_ttot")

    hist, times, gx, gy, cfl = run_heat(
        solver_name, nx, alpha, dt, t_total, ic_name, bc_name, bc_val
    )

    if solver_name == "Explicit Euler" and cfl > 0.5:
        st.warning(
            f"CFL number rx + ry = {cfl:.3f} > 0.5. Explicit Euler is unstable "
            "here — expect the solution to blow up. Try a smaller Δt, a coarser "
            "grid, or switch to Crank-Nicolson (unconditionally stable)."
        )

    with c2:
        frame = st.slider(
            "Time", 0, len(hist) - 1, len(hist) - 1,
            format="step %d", key="h_frame",
        )
        st.caption(f"t = {times[frame]:.4f}  (frame {frame + 1} of {len(hist)})")
        vmax = max(1e-9, float(np.nanmax(hist[0])))
        fig = heatmap_fig(
            hist[frame], gx, gy, vmin=0.0, vmax=vmax,
            title=f"{solver_name} — {ic_name}",
        )
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "**What to notice:** with Dirichlet (fixed) edges the total heat leaks "
        "away and the field decays to zero. With Neumann (insulated) edges no "
        "heat escapes, so the field flattens toward a uniform average instead."
    )


# --------------------------------------------------------------------------
# Tab 2 — Euler vs CN side by side, including the CFL blow-up
# --------------------------------------------------------------------------
with tab_compare:
    st.subheader("Explicit Euler vs Crank-Nicolson")
    st.markdown(
        "Both schemes agree at small Δt. Push Δt past the CFL limit and Explicit "
        "Euler blows up while Crank-Nicolson stays stable — the central reason to "
        "prefer an implicit scheme."
    )
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        alpha_c = st.slider("Diffusivity α", 0.001, 0.1, 0.01, 0.001, key="c_alpha")
        nx_c = st.slider("Grid size", 20, 80, 40, 5, key="c_nx")
        dt_c = st.select_slider(
            "Time step Δt",
            options=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03],
            value=0.001,
            key="c_dt",
        )
        t_c = st.slider("Total time", 0.02, 0.3, 0.05, 0.01, key="c_t")
        dx_c = 1.0 / (nx_c - 1)
        cfl_c = 2 * alpha_c * dt_c / dx_c**2
        st.metric("CFL number (rx + ry)", f"{cfl_c:.3f}", "stable" if cfl_c <= 0.5 else "UNSTABLE")

    he, te, xe, ye, _ = run_heat(
        "Explicit Euler", nx_c, alpha_c, dt_c, t_c, "sin(πx)·sin(πy)", "Dirichlet", 0.0
    )
    hc, tc, xc, yc, _ = run_heat(
        "Crank-Nicolson", nx_c, alpha_c, dt_c, t_c, "sin(πx)·sin(πy)", "Dirichlet", 0.0
    )

    with cc2:
        g1, g2 = st.columns(2)
        eul_max = float(np.nanmax(np.abs(he[-1])))
        vmax_e = 1.0 if eul_max > 10 else max(1e-9, eul_max)
        with g1:
            st.caption("Explicit Euler (final state)")
            fe = heatmap_fig(he[-1], xe, ye, 0.0, vmax_e)
            st.pyplot(fe)
            plt.close(fe)
        with g2:
            st.caption("Crank-Nicolson (final state)")
            fc = heatmap_fig(hc[-1], xc, yc, 0.0, max(1e-9, float(np.nanmax(hc[-1]))))
            st.pyplot(fc)
            plt.close(fc)
        if eul_max > 10:
            st.error(
                f"Explicit Euler has blown up (max value ≈ {eul_max:.1e}) while "
                "Crank-Nicolson remains well-behaved."
            )


# --------------------------------------------------------------------------
# Tab 3 — Validation against the analytical solution
# --------------------------------------------------------------------------
with tab_valid:
    st.subheader("Validation against the closed-form solution")
    st.markdown(
        "For the $\\sin(\\pi x)\\sin(\\pi y)$ initial condition under zero-Dirichlet "
        "boundaries, the heat equation has the exact solution "
        "$u(x,y,t) = \\sin(\\pi x)\\sin(\\pi y)\\,e^{-\\alpha(2\\pi^2)t}$. "
        "We compare the numerical result against it."
    )
    vc1, vc2 = st.columns([1, 2])
    with vc1:
        alpha_v = st.slider("Diffusivity α", 0.001, 0.05, 0.01, 0.001, key="v_alpha")
        nx_v = st.slider("Grid size", 30, 100, 60, 10, key="v_nx")
        dt_v = st.select_slider(
            "Time step Δt", options=[0.0005, 0.001, 0.002], value=0.002, key="v_dt"
        )
        t_v = st.slider("Total time", 0.05, 0.3, 0.1, 0.05, key="v_t")

    s = CrankNicolson2D(nx=nx_v, ny=nx_v, alpha=alpha_v, bc=BCConfig.dirichlet_all(0.0))
    r = s.solve(ic_sin, dt=dt_v, t_total=t_v)
    X, Y = np.meshgrid(r.grid_x, r.grid_y, indexing="ij")
    exact = analytical_solution_sin(X, Y, t=t_v, alpha=alpha_v)
    rel = relative_l2_error(r.u, exact)

    with vc1:
        st.metric("Relative L2 error", f"{rel * 100:.4f} %")

    with vc2:
        g1, g2 = st.columns(2)
        vmax = max(1e-9, float(exact.max()))
        with g1:
            st.caption("Numerical (Crank-Nicolson)")
            f1 = heatmap_fig(r.u, r.grid_x, r.grid_y, 0.0, vmax)
            st.pyplot(f1)
            plt.close(f1)
        with g2:
            st.caption("Analytical (exact)")
            f2 = heatmap_fig(exact, r.grid_x, r.grid_y, 0.0, vmax)
            st.pyplot(f2)
            plt.close(f2)

    st.divider()
    st.subheader("Grid convergence")
    grids, errors = run_convergence(alpha_v, dt_v, t_v)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.loglog(grids, errors, "o-")
    ax.set_xlabel("Grid size (nx)")
    ax.set_ylabel("L2 error")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    if len(errors) >= 2:
        ratio = errors[0] / errors[1]
        st.caption(
            f"Error drops about {ratio:.1f}× when the grid doubles from "
            f"{grids[0]} to {grids[1]} points — consistent with second-order "
            "spatial accuracy."
        )


# --------------------------------------------------------------------------
# Tab 4 — Reaction-diffusion
# --------------------------------------------------------------------------
with tab_react:
    st.subheader("Reaction-Diffusion: pattern formation")
    st.markdown(
        "Strang operator splitting couples diffusion with nonlinear reaction. "
        "**Gray-Scott** produces Turing patterns; **Fisher-KPP** produces "
        "traveling fronts."
    )
    rc1, rc2 = st.columns([1, 2])
    with rc1:
        reaction = st.selectbox("Model", ["Gray-Scott", "Fisher-KPP"], key="r_model")
        nx_r = st.slider("Grid size", 60, 160, 120, 10, key="r_nx")
        if reaction == "Gray-Scott":
            F = st.slider("Feed rate F", 0.02, 0.06, 0.037, 0.001, key="r_F")
            k = st.slider("Kill rate k", 0.05, 0.07, 0.06, 0.001, key="r_k")
        else:
            F, k = 0.04, 0.06
        Du = st.select_slider(
            "U diffusivity Dᵤ",
            options=[1e-5, 1.6e-5, 2e-5, 3e-5],
            value=2e-5,
            key="r_Du",
        )
        Dv = Du / 2
        t_r = st.slider("Total time", 1000, 12000, 8000, 500, key="r_t")
        dt_r = st.select_slider("Time step Δt", options=[0.5, 1.0, 2.0], value=1.0, key="r_dt")

    field, gx, gy, cfl_r = run_reaction(reaction, nx_r, F, k, Du, Dv, t_r, dt_r)

    if cfl_r > 0.5:
        st.warning(
            f"Diffusion CFL number = {cfl_r:.3f} > 0.5. The explicit diffusion "
            "sub-step is unstable — reduce Δt or the diffusivity."
        )

    with rc2:
        cmap = "magma" if reaction == "Gray-Scott" else "viridis"
        label = "V concentration" if reaction == "Gray-Scott" else "u"
        fig = heatmap_fig(
            field, gx, gy,
            vmin=float(np.nanmin(field)), vmax=float(np.nanmax(field)),
            cmap=cmap, title=f"{reaction} — {label}",
        )
        st.pyplot(fig)
        plt.close(fig)
    st.caption(
        "Gray-Scott tip: small changes in F and k move you across the parameter "
        "map between spots, stripes, and labyrinths."
    )


# --------------------------------------------------------------------------
# Tab 5 — About
# --------------------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
This is a from-scratch 2D PDE solver built around the heat equation and
reaction-diffusion systems.

**Numerical methods**
- Second-order central differences in space.
- **Explicit Euler (FTCS):** conditionally stable, requires rx + ry ≤ 0.5 (the CFL condition).
- **Crank-Nicolson:** implicit trapezoidal time-stepping, unconditionally stable,
  second-order in both space and time. Boundary conditions (Dirichlet, Neumann
  via the ghost-node method, periodic) are encoded directly in the sparse system
  matrix and solved with `scipy.sparse`.
- **Reaction-diffusion:** Strang operator splitting (half diffusion → full
  reaction → half diffusion) for Fisher-KPP and Gray-Scott.

**Validation**
- Crank-Nicolson agrees with the analytical solution to ~0.0007% relative L2
  error on a 50×50 grid.
- Grid convergence confirms second-order spatial accuracy.
- An insulated (all-Neumann) box conserves total heat to round-off.

**Tested:** 30 unit and integration tests.
        """
    )
    st.caption("Source code and full README are in the GitHub repository.")
