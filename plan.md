# Numerical PDE Solver — Implementation Plan

## Overview
2D Heat & Reaction-Diffusion PDE solver using Python, NumPy, Matplotlib.
Explicit Euler + Crank-Nicolson schemes, analytical validation, animated viz.

## Project Setup
- **Package Manager:** `uv` (Astral) — venv + dep management
- **Python:** 3.13.3
- **Deps:** numpy, scipy, matplotlib

## Project Structure
```
Numerical solver/
├── src/
│   ├── __init__.py
│   ├── boundary.py         # BC handlers (Dirichlet, Neumann, mixed)
│   ├── solvers.py          # ExplicitEuler2D, CrankNicolson2D
│   ├── reaction.py         # Fisher-KPP, Gray-Scott reaction terms
│   ├── analytics.py        # Analytical solutions, error metrics, convergence
│   └── visualization.py    # Static heatmaps, FuncAnimation, convergence plots
├── notebooks/
│   └── demo.ipynb          # Interactive demo
├── tests/
│   └── test_solvers.py     # Unit tests
├── main.py                 # CLI entry point
├── pyproject.toml
├── plan.md
├── BUILD_LOG.md            # Build + error resolution diary
└── README.md               # Final documentation
```

## Implementation Order (each step tested)

1. **pyproject.toml + uv init** — project skeleton, deps, venv
2. **`src/boundary.py`** — BC enums, apply function, test
3. **`src/solvers.py`** — Euler (explicit) + Crank-Nicolson (implicit/sparse), test
4. **`src/reaction.py`** — Fisher-KPP reaction term, operator-split solve, test
5. **`src/analytics.py`** — analytical heat solution (sinusoidal), L2/max-error, test
6. **`src/visualization.py`** — heatmap, animation, test with dummy data
7. **`main.py`** — argparse CLI, end-to-end solve → viz pipeline, test
8. **`notebooks/demo.ipynb`** — interactive walkthrough
9. **`tests/test_solvers.py`** — comprehensive: stability, convergence, BC, reaction
10. **`README.md`** — full docs: usage, theory, examples

## Key Algorithms

### Explicit Euler (FTCS)
```
u^{n+1}_{i,j} = u^n_{i,j} + r_x (u_{i+1,j} - 2u_{i,j} + u_{i-1,j})
                           + r_y (u_{i,j+1} - 2u_{i,j} + u_{i,j-1})
r_x = α Δt / Δx²,  r_y = α Δt / Δy²
Stability: r_x + r_y ≤ 0.5
```

### Crank-Nicolson (Implicit)
```
(I - (r_x/2) δ²_x - (r_y/2) δ²_y) u^{n+1} = (I + (r_x/2) δ²_x + (r_y/2) δ²_y) u^n
→ sparse linear system: A u^{n+1} = B u^n
Unconditionally stable.
```

### Reaction-Diffusion (Operator Splitting)
1. Diffuse half-step: u^* = u^n + (Δt/2) D ∇²u^n
2. React step: u^{**} = u^* + Δt f(u^*)
3. Diffuse half-step: u^{n+1} = u^{**} + (Δt/2) D ∇²u^{**}
Uses Strang splitting for O(Δt²) accuracy.

### Analytical Solution (Sinusoidal IC, Dirichlet BCs)
```
u(x,y,t) = sin(πx/Lx) sin(πy/Ly) exp(-α π² (1/Lx² + 1/Ly²) t)
```

## Validation Criteria
- CN < 0.3% error on 100×100 vs analytical
- Euler stable within CFL, unstable beyond
- Reaction-diffusion produces pattern formation (Gray-Scott spots/stripes)
