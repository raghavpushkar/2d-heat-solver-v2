import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.analytics import analytical_solution_sin, l2_error, relative_l2_error
from src.boundary import BCConfig
from src.reaction import ReactionDiffusionSolver, ReactionType
from src.solvers import CrankNicolson2D, ExplicitEuler2D
from src.visualization import animate_diffusion, plot_convergence, plot_heatmap

IC_MAP = {
    "sin": lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y),
    "gaussian": lambda X, Y: np.exp(
        -((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.1**2)
    ),
}


def main():
    p = argparse.ArgumentParser(description="2D Heat & Reaction-Diffusion PDE Solver")
    p.add_argument("--solver", choices=["euler", "cn", "reaction"], default="cn")
    p.add_argument("--nx", type=int, default=50)
    p.add_argument("--ny", type=int, default=50)
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.01, help="Thermal diffusivity")
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--t-total", type=float, default=0.1)
    p.add_argument(
        "--ic", choices=["sin", "gaussian"], default="sin", help="Initial condition"
    )
    p.add_argument(
        "--bc",
        choices=["dirichlet", "neumann"],
        default="dirichlet",
        help="Boundary condition type",
    )
    p.add_argument("--bc-val", type=float, default=0.0)
    p.add_argument("--reaction", choices=["fisher", "gray-scott"], default="fisher")
    p.add_argument("--r", type=float, default=1.0, help="Fisher-KPP growth rate")
    p.add_argument("--F", type=float, default=0.04, help="Gray-Scott feed rate")
    p.add_argument("--k", type=float, default=0.06, help="Gray-Scott kill rate")
    p.add_argument("--D", type=float, default=0.001, help="Diffusion coeff (reaction)")
    p.add_argument("--Du", type=float, default=0.001, help="Gray-Scott U diffusivity")
    p.add_argument("--Dv", type=float, default=0.0005, help="Gray-Scott V diffusivity")
    p.add_argument("--save", type=str, default=None, help="Save animation or image to file")
    p.add_argument("--no-display", action="store_true", help="Headless mode")
    p.add_argument("--validate", action="store_true", help="Validate vs analytical")
    p.add_argument("--convergence", action="store_true", help="Run convergence study")

    args = p.parse_args()

    if args.no_display:
        matplotlib.use("Agg")

    bc = (
        BCConfig.dirichlet_all(args.bc_val)
        if args.bc == "dirichlet"
        else BCConfig.neumann_all()
    )

    ic_func = IC_MAP[args.ic]

    if args.solver == "reaction":
        rtype = (
            ReactionType.FISHER_KPP
            if args.reaction == "fisher"
            else ReactionType.GRAY_SCOTT
        )
        solver = ReactionDiffusionSolver(
            nx=args.nx,
            ny=args.ny,
            Lx=args.Lx,
            Ly=args.Ly,
            D=args.D,
            Du=args.Du,
            Dv=args.Dv,
            F=args.F,
            k=args.k,
            r=args.r,
            reaction_type=rtype,
            bc=bc,
        )
        result = solver.solve(ic_func, t_total=args.t_total, dt=args.dt)
        u_final = result.u
        u_hist = result.u_history
        times = result.times
        x, y = result.grid_x, result.grid_y
    else:
        if args.solver == "euler":
            solver = ExplicitEuler2D(
                nx=args.nx, ny=args.ny, Lx=args.Lx, Ly=args.Ly,
                alpha=args.alpha, bc=bc,
            )
        else:
            solver = CrankNicolson2D(
                nx=args.nx, ny=args.ny, Lx=args.Lx, Ly=args.Ly,
                alpha=args.alpha, bc=bc,
            )
        result = solver.solve(ic_func, dt=args.dt, t_total=args.t_total)
        u_final = result.u
        u_hist = result.u_history
        times = result.times
        x, y = result.grid_x, result.grid_y

    if args.validate:
        if args.solver == "reaction":
            print("Validation not available for reaction-diffusion solvers.")
        else:
            X, Y = np.meshgrid(x, y, indexing="ij")
            u_exact = analytical_solution_sin(X, Y, t=args.t_total, alpha=args.alpha)
            err = l2_error(u_final, u_exact)
            rel = relative_l2_error(u_final, u_exact)
            print(f"L2 error: {err:.6e}")
            print(f"Relative L2 error: {rel*100:.4f}%")

    if args.convergence:
        if args.solver == "reaction":
            print("Convergence study not available for reaction-diffusion solvers.")
        else:
            grid_sizes = [20, 30, 40, 50, 60, 80]
            errors = []
            for g in grid_sizes:
                if args.solver == "euler":
                    s = ExplicitEuler2D(
                        nx=g, ny=g, Lx=args.Lx, Ly=args.Ly,
                        alpha=args.alpha, bc=bc,
                    )
                else:
                    s = CrankNicolson2D(
                        nx=g, ny=g, Lx=args.Lx, Ly=args.Ly,
                        alpha=args.alpha, bc=bc,
                    )
                r = s.solve(ic_func, dt=args.dt, t_total=args.t_total)
                X, Y = np.meshgrid(r.grid_x, r.grid_y, indexing="ij")
                u_ex = analytical_solution_sin(X, Y, t=args.t_total, alpha=args.alpha)
                errors.append(l2_error(r.u, u_ex))
                print(f"  nx={g}: error={errors[-1]:.6e}")
            plot_convergence(grid_sizes, errors)
            out = Path("convergence.png")
            plt.savefig(out)
            print(f"Convergence plot saved to {out}")

    if args.save:
        ext = Path(args.save).suffix.lower()
        if ext in (".gif",):
            animate_diffusion(
                u_hist, x, y, times, save_path=args.save, interval=100,
            )
            print(f"Animation saved to {args.save}")
        else:
            plot_heatmap(u_final, x, y, title=f"Final state — t={args.t_total}")
            plt.savefig(args.save)
            print(f"Saved to {args.save}")
    elif not args.convergence:
        plot_heatmap(u_final, x, y, title=f"Final state — t={args.t_total}")
        out = Path("final_state.png")
        plt.savefig(out)
        print(f"Final state saved to {out}")

    if not args.no_display:
        plt.show()


if __name__ == "__main__":
    main()
