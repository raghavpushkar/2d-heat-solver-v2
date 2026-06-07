# 2D Heat & Reaction-Diffusion PDE Solver

A from-scratch finite-difference solver for the two-dimensional heat equation
and reaction-diffusion systems, written in NumPy and SciPy. It implements two
time-stepping schemes (Explicit Euler and Crank-Nicolson), validates them
against a closed-form analytical solution, and ships with a test suite, a
command-line interface, and an interactive browser app.

![Heat diffusion](assets/heat_diffusion_panel.png)

*A Gaussian hot spot cooling under zero-Dirichlet boundaries, solved with Crank-Nicolson.*

## Contents

- [Interactive app](#interactive-app)
- [Quick start](#quick-start)
- [The governing equations](#the-governing-equations)
- [Discretization](#discretization)
- [Time-stepping schemes](#time-stepping-schemes)
- [Boundary conditions](#boundary-conditions)
- [Reaction-diffusion](#reaction-diffusion)
- [Validation and results](#validation-and-results)
- [Command-line interface](#command-line-interface)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Requirements](#requirements)

## Interactive app

An interactive [Streamlit](https://streamlit.io) app exposes the solver in the
browser. It has four tabs:

- **Heat flow** — set up an initial temperature field, boundary type, and
  diffusivity, then drag a time slider to watch the field evolve. The slider
  scrubs through a precomputed history, so it responds instantly.
- **Method comparison** — run Explicit Euler and Crank-Nicolson side by side
  from a noisy initial field. The panel reports the stability number and adapts
  its explanation to the regime: stable (the schemes agree), past the limit
  (Euler is formally unstable), or diverged (Euler has blown up).
- **Validation** — compare the numerical solution against the exact analytical
  solution, with an error map and a grid-convergence plot.
- **About** — a short summary of the methods.

**Live demo:** [2d-heat-solver-v2.streamlit.app](https://2d-heat-solver-v2.streamlit.app/)

Run it locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Quick start

```bash
# Install (choose one)
pip install -r requirements.txt        # or: uv sync

# Solve the heat equation with Crank-Nicolson and validate vs analytical
python main.py --solver cn --nx 50 --ic sin --validate

# Run a grid-convergence study
python main.py --solver cn --convergence

# Solve an insulated (Neumann) box and save an animation
python main.py --solver cn --bc neumann --ic gaussian --save heat.gif
```

## The governing equations

The **heat equation** (equivalently the diffusion equation) on a rectangular
domain is

```
∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²) = α ∇²u
```

where `u(x, y, t)` is temperature (or concentration) and `α` is the thermal
diffusivity. The solver works on a rectangle `[0, Lx] × [0, Ly]`.

The **reaction-diffusion** systems add a nonlinear reaction term `R(u)`:

```
∂u/∂t = D ∇²u + R(u)
```

## Discretization

Space is discretized on a uniform grid of `nx × ny` points with spacings
`dx = Lx/(nx-1)` and `dy = Ly/(ny-1)`. The Laplacian uses the standard
**second-order central difference** stencil. In one dimension,

```
∂²u/∂x²  ≈  (u[i+1] - 2 u[i] + u[i-1]) / dx²
```

Two dimensionless diffusion numbers control everything:

```
rx = α dt / dx²        ry = α dt / dy²
```

## Time-stepping schemes

### Explicit Euler (FTCS)

Forward-Time, Central-Space. Each new value is computed directly from the
current step:

```
u[i,j]^{n+1} = u[i,j]^n
             + rx (u[i+1,j] - 2 u[i,j] + u[i-1,j])^n
             + ry (u[i,j+1] - 2 u[i,j] + u[i,j-1])^n
```

This is cheap (no linear solve) but only **conditionally stable**. Stability
requires the CFL condition

```
rx + ry ≤ 0.5
```

The solver emits a warning when this is violated. Past the limit, high-frequency
error components are amplified each step and the solution diverges. The
`explicit_diffusion_step` function performs one such step in place using
vectorized NumPy slicing.

### Crank-Nicolson

An implicit scheme that averages the spatial operator across the current and
next time levels (the trapezoidal rule in time). It is **unconditionally
stable** and **second-order accurate in both space and time** — O(Δt² + Δx²).
Each step requires solving a linear system

```
A u^{n+1} = B u^n + b
```

where `A` and `B` are sparse matrices assembled once per time-step size. For an
interior node the stencil coefficients are

```
A: diagonal 1 + rx + ry,  off-diagonals -rx/2 (x), -ry/2 (y)
B: diagonal 1 - rx - ry,  off-diagonals +rx/2 (x), +ry/2 (y)
```

The system is built over the **full grid** (every node gets an explicit row),
stored as a SciPy CSR sparse matrix, and solved each step with
`scipy.sparse.linalg.spsolve`. The matrices are cached and rebuilt only when the
time step changes between calls.

## Boundary conditions

Three boundary types are supported, each configured per edge through `BCConfig`.
In Crank-Nicolson they are encoded **directly in the system matrix**, so the
condition is enforced inside the implicit solve rather than patched on
afterwards:

- **Dirichlet** (fixed value): the boundary row is the identity and the
  prescribed value is injected through the right-hand-side vector each step.
  Time-dependent values are supported via per-edge callables.
- **Neumann** (zero-flux / insulated): implemented with the **ghost-node
  method**. The ghost point outside the wall mirrors the interior neighbour,
  which preserves the second-order stencil at the boundary and keeps the scheme
  conservative. This was a deliberate fix — a one-sided copy would have been
  only first-order and would have leaked heat.
- **Periodic**: opposite edges are tied to each other's interior rows.

Explicit Euler applies the same boundary conditions through a separate
`apply_boundary` routine after each update step.

## Reaction-diffusion

Two nonlinear systems are integrated using **Strang operator splitting**, which
alternates a half diffusion step, a full reaction step, and a second half
diffusion step to retain second-order accuracy in time:

```
diffuse(dt/2)  →  react(dt)  →  diffuse(dt/2)
```

- **Fisher-KPP** (single species): reaction term `r u (1 - u)`, producing
  traveling wave fronts.
- **Gray-Scott** (two species U and V): a feed-and-kill system,
  `-uv² + F(1-u)` for U and `uv² - (F+k)v` for V, producing Turing patterns
  (spots, stripes, labyrinths).

The diffusion sub-steps are explicit, so the reaction-diffusion solver is
subject to the same CFL limit as Explicit Euler: `D (dt/2) / dx² · 2 ≤ 0.5` per
half-step. Choose the time step, grid spacing, and diffusion coefficients
accordingly.

## Validation and results

### Accuracy against the analytical solution

For the initial condition `u(x,y,0) = sin(πx) sin(πy)` under zero-Dirichlet
boundaries, the heat equation has the exact solution

```
u(x, y, t) = sin(πx) sin(πy) · exp(-α (2π²) t)
```

Crank-Nicolson reproduces it to a relative L2 error well under 0.001%
(`α = 0.01`, `t = 0.1`):

| Grid | dt | Relative L2 error |
|---|---|---|
| 50 × 50 | 0.002 | 0.0007 % |
| 60 × 60 | 0.002 | 0.0005 % |
| 100 × 100 | 0.001 | 0.0002 % |

### Grid convergence

Refining the grid (Crank-Nicolson, `dt = 0.002`) drops the L2 error by ~4×
each time the resolution doubles — the signature of **second-order** spatial
accuracy:

| Grid | L2 error | Ratio |
|---|---|---|
| 20 × 20 | 2.09 × 10⁻⁵ | — |
| 40 × 40 | 5.10 × 10⁻⁶ | 4.10× |
| 80 × 80 | 1.26 × 10⁻⁶ | 4.05× |

### Heat conservation under Neumann boundaries

In a fully insulated (all-Neumann) box, total heat must be conserved. Measured
with trapezoidal weights (boundary nodes count as half a control volume, corners
a quarter), the relative drift is ~10⁻¹³% — i.e. round-off. The field correctly
relaxes toward its uniform mean.

### Stability demonstration

Starting from a noisy field, Explicit Euler is stable while `rx + ry ≤ 0.5` and
diverges once the time step pushes it past that limit, whereas Crank-Nicolson
remains stable at every step size. This contrast is the central reason to prefer
an implicit scheme and is reproduced in the test suite.

## Command-line interface

`main.py` exposes the solver through `argparse`:

| Flag | Default | Description |
|---|---|---|
| `--solver` | `cn` | `euler`, `cn`, or `reaction` |
| `--nx`, `--ny` | 50 | Grid points per side |
| `--Lx`, `--Ly` | 1.0 | Domain size |
| `--alpha` | 0.01 | Thermal diffusivity |
| `--dt` | 0.001 | Time step |
| `--t-total` | 0.1 | Total simulation time |
| `--ic` | `sin` | Initial condition: `sin` or `gaussian` |
| `--bc` | `dirichlet` | Boundary type: `dirichlet` or `neumann` |
| `--bc-val` | 0.0 | Dirichlet boundary value |
| `--validate` | off | Compare against the analytical solution |
| `--convergence` | off | Run a grid-convergence study |
| `--save` | None | Save output as `*.png` or `*.gif` |
| `--no-display` | off | Headless mode (no interactive window) |
| `--reaction` | `fisher` | `fisher` or `gray-scott` |
| `--r` | 1.0 | Fisher-KPP growth rate |
| `--F` | 0.04 | Gray-Scott feed rate |
| `--k` | 0.06 | Gray-Scott kill rate |
| `--D` | 0.001 | Reaction diffusion coefficient |
| `--Du`, `--Dv` | 0.001 / 0.0005 | Gray-Scott species diffusivities |

## Project structure

```
.
├── src/
│   ├── __init__.py          # Public API exports
│   ├── _types.py            # Shared ICFunc type alias
│   ├── boundary.py          # BCType, BCConfig, apply_boundary
│   ├── solvers.py           # explicit_diffusion_step, ExplicitEuler2D, CrankNicolson2D
│   ├── reaction.py          # ReactionDiffusionSolver (Fisher-KPP, Gray-Scott)
│   ├── analytics.py         # Analytical solution and error metrics
│   └── visualization.py     # Heatmaps, animations, convergence plots
├── tests/
│   └── test_solvers.py      # 30 tests
├── notebooks/
│   └── demo.ipynb           # Interactive demo notebook
├── assets/                  # Rendered figures and animations
├── main.py                  # Command-line entry point
├── streamlit_app.py         # Interactive browser app
├── pyproject.toml           # Project metadata and tooling config
├── requirements.txt         # Dependencies (for the app / pip installs)
└── uv.lock                  # Locked dependency versions
```

### Module overview

- **`solvers.py`** — the core. `BaseSolver2D` handles grid setup, initial-
  condition evaluation, and snapshot recording. `ExplicitEuler2D` and
  `CrankNicolson2D` implement the two schemes. `SolverResult` is the returned
  dataclass (final field, time history, grid coordinates, step metadata).
- **`boundary.py`** — `BCType` (Dirichlet / Neumann / Periodic), the per-edge
  `BCConfig` with factory presets and time-dependent callables, and
  `apply_boundary`.
- **`reaction.py`** — `ReactionDiffusionSolver` with Strang splitting, plus the
  Fisher-KPP and Gray-Scott reaction terms.
- **`analytics.py`** — the closed-form sinusoidal solution and the L2, max, and
  relative-L2 error metrics (relative error guards against a zero denominator).
- **`visualization.py`** — Matplotlib heatmaps, GIF animation, and the log-log
  convergence plot.

## Testing

```bash
pytest                       # 30 tests
ruff check src/ main.py tests/ streamlit_app.py
```

The suite covers boundary conditions (Dirichlet, Neumann, periodic, mixed,
time-dependent), Explicit Euler stability inside and outside the CFL limit,
Crank-Nicolson accuracy and convergence against the analytical solution,
Neumann heat conservation, the analytical metrics, and the reaction-diffusion
systems.

## Requirements

- Python ≥ 3.12
- numpy ≥ 2.0, scipy ≥ 1.14, matplotlib ≥ 3.9
- plotly ≥ 5.20, streamlit ≥ 1.40 (for the interactive app)
- pytest ≥ 8.0 (for the test suite)
