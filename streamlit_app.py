"""Interactive front-end for the 2D Heat & Reaction-Diffusion solver.

Charts use Plotly so they render in the browser: the time slider scrubs through
a precomputed solution history client-side, with no server round-trip per move.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analytics import analytical_solution_sin, l2_error, relative_l2_error
from src.boundary import BCConfig
from src.solvers import CrankNicolson2D, ExplicitEuler2D

st.set_page_config(
    page_title="2D Heat & Reaction-Diffusion Solver",
    layout="wide",
)

# --------------------------------------------------------------------------
# Styling - theme-aware (works in both light and dark mode)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root { --accent: #c2410c; }

    html, body, [class*="css"], .stMarkdown, p, span, label, div {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main .block-container {
        max-width: 1160px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    /* Headings inherit the theme text color so they read in both modes */
    h1, h2, h3 {
        font-family: 'Source Serif 4', serif;
        letter-spacing: -0.01em;
        color: inherit;
    }
    h1 { font-weight: 600; font-size: 1.95rem; }
    h2 { font-weight: 600; font-size: 1.3rem; margin-top: 0.4rem; }
    h3 { font-weight: 600; font-size: 1.05rem; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid rgba(128,128,128,0.25);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem; font-weight: 500;
        padding: 0.4rem 0; background: transparent;
        opacity: 0.65;
    }
    .stTabs [aria-selected="true"] { opacity: 1; color: var(--accent); }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent); }

    /* Lead paragraph and help notes use the theme text color at reduced opacity
       so they stay readable on light OR dark backgrounds */
    .lead {
        font-size: 0.97rem; line-height: 1.55; max-width: 72ch;
        color: inherit; opacity: 0.82;
    }
    .help-note {
        font-size: 0.8rem; line-height: 1.4;
        margin: -0.3rem 0 1rem 0;
        color: inherit; opacity: 0.55;
    }
    .metric-mono { font-family: 'IBM Plex Mono', monospace; }

    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

ACCENT = "#c2410c"
HEAT_SCALE = "Viridis"
PLOT_FONT = "IBM Plex Sans, sans-serif"


def help_note(text: str) -> None:
    st.markdown(f'<div class="help-note">{text}</div>', unsafe_allow_html=True)


def ic_sin(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


def ic_gaussian(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.1**2))


def ic_noisy(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # High-frequency content: this is what reveals Explicit Euler's instability.
    rng = np.random.default_rng(1)
    return 0.5 + 0.5 * rng.random(X.shape)


IC_MAP = {
    "Smooth sine bump": ic_sin,
    "Gaussian hot spot": ic_gaussian,
    "Random noise": ic_noisy,
}


@st.cache_data(show_spinner=False)
def run_heat(solver_name, nx, alpha, dt, t_total, ic_name, bc_name, bc_val):
    ic = IC_MAP[ic_name]
    bc = (
        BCConfig.dirichlet_all(bc_val)
        if bc_name == "Fixed edges (Dirichlet)"
        else BCConfig.neumann_all()
    )
    cls = ExplicitEuler2D if solver_name == "Explicit Euler" else CrankNicolson2D
    solver = cls(nx=nx, ny=nx, alpha=alpha, bc=bc)
    n_steps = max(1, int(np.ceil(t_total / dt)))
    record_every = max(1, n_steps // 80)  # ~80 stored frames keeps animation smooth
    result = solver.solve(ic, dt=dt, t_total=t_total, record_every=record_every)
    rx = alpha * dt / solver.dx**2
    return (
        np.array(result.u_history),
        np.array(result.times),
        result.grid_x,
        result.grid_y,
        2 * rx,
    )


@st.cache_data(show_spinner=False)
def run_convergence(alpha, dt, t_total):
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
def run_validation(alpha, nx, dt, t_total):
    s = CrankNicolson2D(nx=nx, ny=nx, alpha=alpha, bc=BCConfig.dirichlet_all(0.0))
    r = s.solve(ic_sin, dt=dt, t_total=t_total)
    X, Y = np.meshgrid(r.grid_x, r.grid_y, indexing="ij")
    exact = analytical_solution_sin(X, Y, t=t_total, alpha=alpha)
    return r.u, exact, r.grid_x, r.grid_y, relative_l2_error(r.u, exact)


def base_layout(fig, height=440, title=None):
    # Transparent backgrounds let the chart sit on whatever theme the page uses;
    # a mid-grey font and gridline read acceptably on both light and dark.
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 16, b=10),
        font=dict(family=PLOT_FONT, size=13, color="#808080"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(size=15)) if title else None,
    )
    fig.update_xaxes(
        showline=True, linecolor="rgba(128,128,128,0.4)",
        zeroline=False, tickfont=dict(color="#808080"),
    )
    fig.update_yaxes(
        showline=True, linecolor="rgba(128,128,128,0.4)",
        zeroline=False, tickfont=dict(color="#808080"),
    )
    return fig


def heat_figure(u, x, y, zmin, zmax, colorscale=HEAT_SCALE, height=440, title=None):
    fig = go.Figure(
        go.Heatmap(
            z=u.T, x=x, y=y, zmin=zmin, zmax=zmax, colorscale=colorscale,
            colorbar=dict(
                thickness=12, outlinewidth=0, len=0.9,
                tickfont=dict(color="#808080"),
            ),
            hovertemplate="x=%{x:.2f}, y=%{y:.2f}<br>value=%{z:.3f}<extra></extra>",
        )
    )
    # fixedrange disables zoom/pan so a stray click can't get the view stuck.
    fig.update_xaxes(
        title="x", showgrid=False, constrain="domain", fixedrange=True
    )
    fig.update_yaxes(
        title="y", showgrid=False, scaleanchor="x", scaleratio=1, fixedrange=True
    )
    fig = base_layout(fig, height=height, title=title)
    fig.update_layout(dragmode=False)
    return fig


# Plotly config shared by every chart: no mode bar, no zoom-stuck, no scroll-zoom.
CHART_CONFIG = {"displayModeBar": False, "staticPlot": False, "scrollZoom": False}


st.title("2D Heat & Reaction-Diffusion Solver")
st.markdown(
    '<p class="lead">A finite-difference solver for heat flow and pattern-forming '
    "chemical systems in two dimensions. It implements two time-stepping schemes "
    "(explicit Euler and Crank-Nicolson), is validated against an exact solution, "
    "and is built from scratch in NumPy and SciPy. Use the controls to explore how "
    "the numerical methods behave.</p>",
    unsafe_allow_html=True,
)
st.write("")

tab_heat, tab_compare, tab_valid, tab_about = st.tabs(
    ["Heat flow", "Method comparison", "Validation", "About"]
)


with tab_heat:
    st.subheader("How heat spreads over time")
    st.markdown(
        '<p class="lead">A region starts hot in the middle and cools as heat '
        "spreads outward. Set up the scenario on the left, then drag the time "
        "slider to watch it evolve.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([1, 1.9], gap="large")
    with left:
        solver_name = st.selectbox("Numerical scheme", ["Crank-Nicolson", "Explicit Euler"])
        help_note(
            "The recipe for stepping forward in time. Crank-Nicolson is stable at "
            "any step size; Explicit Euler is simpler but can break down (see the "
            "comparison tab)."
        )

        ic_name = st.selectbox("Starting heat pattern", list(IC_MAP))
        help_note(
            "The temperature at the start. A Gaussian is a single concentrated "
            "hot spot; the sine bump is a smooth mound filling the square."
        )

        bc_name = st.selectbox(
            "Edge behaviour", ["Fixed edges (Dirichlet)", "Insulated edges (Neumann)"]
        )
        help_note(
            "What happens at the walls. Fixed edges are held at a set temperature, "
            "so heat escapes. Insulated edges let no heat out, so it spreads "
            "inward and evens out."
        )

        bc_val = 0.0
        if bc_name == "Fixed edges (Dirichlet)":
            bc_val = st.slider("Edge temperature", 0.0, 1.0, 0.0, 0.05)
            help_note("The temperature the walls are held at. Zero means cold walls.")

        alpha = st.slider("Diffusivity", 0.001, 0.1, 0.02, 0.001, format="%.3f")
        help_note(
            "How fast heat conducts through the material. Higher means it spreads "
            "and evens out more quickly."
        )

        nx = st.slider("Grid resolution", 20, 100, 60, 5)
        help_note(
            "How finely the square is divided. More points give a sharper picture "
            "but take longer to compute."
        )

    # Choose the animation time step so Explicit Euler stays stable (CFL <= 0.4)
    # while Crank-Nicolson, stable at any step, still solves quickly. This keeps
    # the heat-flow tab responsive without ever showing a diverged Euler field.
    dx = 1.0 / (nx - 1)
    euler_safe_dt = 0.4 * dx**2 / (2 * alpha)
    anim_dt = min(0.005, euler_safe_dt) if solver_name == "Explicit Euler" else 0.005
    hist, times, gx, gy, cfl = run_heat(
        solver_name, nx, alpha, anim_dt, 0.4, ic_name, bc_name, bc_val
    )

    with right:
        n_frames = len(hist)
        frame = st.slider("Time", 0, n_frames - 1, n_frames - 1, format="")
        help_note(
            f"Simulated time t = {times[frame]:.3f}. Drag to move through the "
            "evolution from start to finish."
        )

        if solver_name == "Explicit Euler" and cfl > 0.5:
            st.warning(
                f"This combination is numerically unstable (stability number "
                f"{cfl:.2f}, must stay under 0.5). The result will diverge - see "
                "the comparison tab for why."
            )

        zmax = max(1e-9, float(np.nanmax(hist[0])))
        fig = heat_figure(hist[frame], gx, gy, 0.0, zmax, height=460)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        '<p class="lead"><strong>What to look for:</strong> with fixed edges the '
        "heat steadily drains away and the whole region cools toward the edge "
        "temperature. With insulated edges nothing escapes, so the hot spot "
        "spreads out until the temperature is nearly uniform.</p>",
        unsafe_allow_html=True,
    )


with tab_compare:
    st.subheader("Why the choice of scheme matters")
    st.markdown(
        '<p class="lead">Both schemes solve the same equation. Starting from a '
        "rough, noisy temperature field, the simpler Explicit Euler scheme stays "
        "stable only while the stability number is at or below 0.50 - push the time "
        "step past that and it blows up into garbage, while Crank-Nicolson smooths "
        "the noise away correctly. This is the central reason to use an implicit "
        "scheme. The default settings are already past the limit; lower the time "
        "step to make Euler behave again.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([1, 1.9], gap="large")
    with left:
        alpha_c = st.slider("Diffusivity", 0.001, 0.1, 0.02, 0.001, format="%.3f", key="c_a")
        help_note("How fast heat conducts. Affects the stability threshold below.")

        nx_c = st.slider("Grid resolution", 20, 80, 40, 5, key="c_nx")
        help_note("Finer grids are more sensitive and destabilise at smaller time steps.")

        dt_c = st.select_slider(
            "Time step size",
            options=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03],
            value=0.01, key="c_dt",
        )
        help_note(
            "How far each step jumps forward in time. Larger is faster but riskier "
            "for Explicit Euler. Slide it up to 0.02 or beyond and watch Euler break."
        )

        t_c = st.slider("Total time", 0.02, 0.3, 0.15, 0.01, key="c_t")
        help_note("How long to simulate in total. Longer runs let instability grow.")

        dx_c = 1.0 / (nx_c - 1)
        cfl_c = 2 * alpha_c * dt_c / dx_c**2
        status = "stable" if cfl_c <= 0.5 else "UNSTABLE"
        color = "#2e7d32" if cfl_c <= 0.5 else ACCENT
        st.markdown(
            f'<p class="metric-mono" style="font-size:0.9rem;color:{color};">'
            f"Stability number: {cfl_c:.2f} &nbsp;({status})</p>",
            unsafe_allow_html=True,
        )
        help_note(
            "Explicit Euler is only stable when this stays at or below 0.50. "
            "Crank-Nicolson has no such limit."
        )

    he, _, xe, ye, _ = run_heat(
        "Explicit Euler", nx_c, alpha_c, dt_c, t_c, "Random noise",
        "Fixed edges (Dirichlet)", 0.0,
    )
    hc, _, xc, yc, _ = run_heat(
        "Crank-Nicolson", nx_c, alpha_c, dt_c, t_c, "Random noise",
        "Fixed edges (Dirichlet)", 0.0,
    )

    with right:
        eul_max = float(np.nanmax(np.abs(he[-1])))
        diverged = eul_max > 10
        over_limit = cfl_c > 0.5
        # Both panels share one color scale so genuine magnitude differences are
        # visible rather than hidden by independent rescaling.
        shared_max = 1.0 if diverged else max(1e-9, float(np.nanmax(hc[-1])))
        g1, g2 = st.columns(2, gap="medium")
        with g1:
            fig_e = heat_figure(he[-1], xe, ye, 0.0, shared_max, height=320, title="Explicit Euler")
            st.plotly_chart(fig_e, use_container_width=True, config=CHART_CONFIG)
        with g2:
            fig_c = heat_figure(hc[-1], xc, yc, 0.0, shared_max, height=320, title="Crank-Nicolson")
            st.plotly_chart(fig_c, use_container_width=True, config=CHART_CONFIG)

        diff = np.abs(he[-1] - hc[-1])
        max_diff = float(np.nanmax(diff))

        if diverged:
            # Regime 1: Euler has blown up. State it plainly.
            st.error(
                f"Explicit Euler has diverged — its peak value reached about "
                f"{eul_max:.0e}, pure numerical garbage, while Crank-Nicolson stayed "
                "well-behaved. This is exactly the failure the stability limit warns "
                "about. Lower the time step to bring Euler back."
            )
        elif over_limit:
            # Regime 2: past the limit but instability hasn't grown visibly yet.
            st.warning(
                f"The stability number is {cfl_c:.2f}, above the 0.50 limit, so "
                "Explicit Euler is formally unstable here — the error is growing but "
                "has not yet blown up over this run. Increase the total time and "
                "watch it erupt, or raise the time step further to trigger it faster."
            )
        else:
            # Regime 3: stable. Explain WHY the panels match, so an identical-looking
            # pair reads as a correct result rather than a broken tab.
            st.info(
                f"The stability number is {cfl_c:.2f}, comfortably below the 0.50 "
                "limit, so both schemes are stable and agree closely — that is why "
                "the two panels look the same. This is the safe regime. To see the "
                "schemes part ways, raise the time step (and total time) until the "
                "stability number climbs above 0.50."
            )
            fig_d = heat_figure(
                diff, xe, ye, 0.0, max(1e-9, max_diff),
                colorscale="Hot", height=300,
                title=f"Where they disagree (peak {max_diff:.1e})",
            )
            st.plotly_chart(fig_d, use_container_width=True, config=CHART_CONFIG)
            st.markdown(
                '<p class="help-note">The tiny remaining difference concentrates '
                "near the fixed edges, where the solution bends most sharply and the "
                "two schemes approximate it slightly differently. It shrinks as the "
                "time step shrinks.</p>",
                unsafe_allow_html=True,
            )


with tab_valid:
    st.subheader("Checking the solver against an exact answer")
    st.markdown(
        '<p class="lead">For one specific starting pattern the heat equation can be '
        "solved exactly with pen and paper. Comparing the solver against that known "
        "answer measures how accurate it really is.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([1, 1.9], gap="large")
    with left:
        alpha_v = st.slider("Diffusivity", 0.001, 0.05, 0.01, 0.001, format="%.3f", key="v_a")
        help_note("How fast heat conducts in this test.")

        nx_v = st.slider("Grid resolution", 30, 100, 60, 10, key="v_nx")
        help_note("Finer grids should track the exact answer more closely.")

        t_v = st.slider("Total time", 0.05, 0.3, 0.1, 0.05, key="v_t")
        help_note("How long to run before comparing.")

        u_num, exact, gxv, gyv, rel = run_validation(alpha_v, nx_v, 0.002, t_v)
        st.markdown(
            f'<p class="metric-mono" style="font-size:1.4rem;color:{ACCENT};margin-top:1rem;">'
            f"{rel * 100:.4f}%</p>"
            '<p class="help-note">Average disagreement with the exact solution. '
            "Below a fraction of a percent is excellent.</p>",
            unsafe_allow_html=True,
        )

    with right:
        zmax = max(1e-9, float(exact.max()))
        g1, g2 = st.columns(2, gap="medium")
        with g1:
            f1 = heat_figure(u_num, gxv, gyv, 0.0, zmax, height=360, title="Solver result")
            st.plotly_chart(f1, use_container_width=True, config=CHART_CONFIG)
        with g2:
            f2 = heat_figure(exact, gxv, gyv, 0.0, zmax, height=360, title="Exact solution")
            st.plotly_chart(f2, use_container_width=True, config=CHART_CONFIG)

        # The two panels look identical because the solver is accurate; the
        # error map below is where the (tiny) disagreement is actually visible.
        err_map = np.abs(u_num - exact)
        peak = float(np.nanmax(err_map))
        f3 = heat_figure(
            err_map, gxv, gyv, 0.0, max(1e-9, peak),
            colorscale="Hot", height=300,
            title=f"Error map (peak {peak:.1e})",
        )
        st.plotly_chart(f3, use_container_width=True, config=CHART_CONFIG)
        st.markdown(
            '<p class="help-note">The first two panels look the same because the '
            "solver is accurate. The error map magnifies the difference: it is "
            "largest where the solution curves most, and shrinks as the grid is "
            "refined.</p>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Accuracy improves as the grid gets finer")
    grids, errors = run_convergence(alpha_v, 0.002, t_v)
    conv = go.Figure(
        go.Scatter(
            x=grids, y=errors, mode="lines+markers",
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=9, color=ACCENT),
            hovertemplate="grid %{x} x %{x}<br>error %{y:.2e}<extra></extra>",
        )
    )
    conv.update_xaxes(
        title="Grid resolution (points per side)", type="log",
        gridcolor="rgba(128,128,128,0.15)", fixedrange=True,
    )
    conv.update_yaxes(
        title="Error vs exact solution", type="log",
        gridcolor="rgba(128,128,128,0.15)", fixedrange=True,
    )
    base_layout(conv, height=420)
    conv.update_layout(dragmode=False)
    st.plotly_chart(conv, use_container_width=True, config=CHART_CONFIG)
    if len(errors) >= 2:
        ratio = errors[0] / errors[1]
        st.markdown(
            f'<p class="help-note">Both axes are logarithmic. The error falls by '
            f"about {ratio:.1f} times when the grid resolution increases by 1.5x, "
            "which is the hallmark of a second-order accurate method.</p>",
            unsafe_allow_html=True,
        )


with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
This is a from-scratch numerical solver for the two-dimensional heat equation
and reaction-diffusion systems, written in NumPy and SciPy.

**Numerical methods.** Space is discretised with second-order central
differences. Two time-stepping schemes are implemented: Explicit Euler, which is
simple but only stable below a step-size limit (the CFL condition), and
Crank-Nicolson, an implicit scheme that is stable at any step size and
second-order accurate in both space and time. Its boundary conditions - fixed,
insulated, and periodic - are built directly into the sparse linear system that
is solved at each step.

**Validation.** The Crank-Nicolson solver agrees with the exact analytical
solution to roughly 0.0007% on a 50x50 grid, and a grid-refinement study
confirms the expected second-order accuracy. In a fully insulated domain the
total heat is conserved to numerical round-off.

The full source, test suite, and documentation are in the project repository.
        """
    )
