from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes


def plot_heatmap(
    u: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    title: str = "",
    ax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "hot",
    cbar: bool = True,
) -> Axes:
    """Plot a 2-D field as a filled heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    _x: np.ndarray = x if x is not None else np.arange(u.shape[0])
    _y: np.ndarray = y if y is not None else np.arange(u.shape[1])
    im = ax.pcolormesh(_x, _y, u.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    if cbar:
        plt.colorbar(im, ax=ax)
    return ax


def animate_diffusion(
    u_history: list[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    times: list[float],
    title: str = "Heat Diffusion",
    cmap: str = "hot",
    interval: int = 50,
    save_path: str | None = None,
) -> FuncAnimation:
    """Animate a sequence of 2-D fields; optionally save as GIF.

    The returned *FuncAnimation* object must be kept alive by the caller
    (e.g. assigned to a variable) to prevent garbage collection from stopping
    the animation in interactive backends.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    vmin = min(snap.min() for snap in u_history)
    vmax = max(snap.max() for snap in u_history)

    im = ax.pcolormesh(x, y, u_history[0].T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    t_text = ax.set_title(f"{title} — t = {times[0]:.4f}")

    def update(frame: int):
        im.set_array(u_history[frame].T.ravel())
        t_text.set_text(f"{title} — t = {times[frame]:.4f}")
        return (im, t_text)

    anim = FuncAnimation(fig, update, frames=len(u_history), interval=interval, blit=True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer="pillow", fps=1000 // interval)

    return anim


def plot_convergence(
    grid_sizes: list[int],
    errors: list[float],
    label: str = "CN",
    ax: Axes | None = None,
) -> Axes:
    """Log-log convergence plot of error vs grid size."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(grid_sizes, errors, "o-", label=label)
    ax.set_xlabel("Grid size (nx)")
    ax.set_ylabel("L2 error")
    ax.set_title("Grid Convergence")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()
    return ax
