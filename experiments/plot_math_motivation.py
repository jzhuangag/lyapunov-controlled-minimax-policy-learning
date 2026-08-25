"""Draw the two-panel geometric motivation figure for the ICC paper.

The figure uses the bilinear field F(x,y)=(y,-x).  It is a mechanism
illustration, not an empirical result, and deliberately precedes LCMPL.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "figures"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 6.8,
        "axes.labelsize": 7.0,
        "legend.fontsize": 5.8,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

GDA = "#D84A2B"
PPM = "#228B45"
EGM = "#2A6FBB"
FIELD = "#D84A2B"
CURVATURE = "#2A6FBB"
CONTOUR = "#718593"


def field(z):
    """Bilinear saddle field F=Az with A skew-symmetric."""
    return np.array([z[1], -z[0]])


def trajectories(initial, step, iterations):
    matrix = np.array([[0.0, 1.0], [-1.0, 0.0]])
    identity = np.eye(2)
    updates = {
        "GDA": identity - step * matrix,
        "PPM": np.linalg.inv(identity + step * matrix),
        "EGM": identity - step * matrix + step**2 * matrix @ matrix,
    }
    paths = {}
    for name, update in updates.items():
        path = [np.asarray(initial, dtype=float)]
        for _ in range(iterations):
            path.append(update @ path[-1])
        paths[name] = np.asarray(path)
    return paths


def draw_energy(axis, limit):
    grid = np.linspace(-limit, limit, 300)
    x_grid, y_grid = np.meshgrid(grid, grid)
    energy = 0.5 * (x_grid**2 + y_grid**2)
    levels = np.linspace(0.18, 2.0, 8)
    axis.contourf(
        x_grid,
        y_grid,
        energy,
        levels=np.r_[0.0, levels, 4.0],
        cmap="Blues",
        alpha=0.34,
    )
    axis.contour(
        x_grid,
        y_grid,
        energy,
        levels=levels,
        colors=CONTOUR,
        linewidths=0.45,
        alpha=0.68,
    )
    axis.axhline(0.0, color="#9AA7AE", lw=0.35, zorder=1)
    axis.axvline(0.0, color="#9AA7AE", lw=0.35, zorder=1)


def trajectory_panel(axis):
    draw_energy(axis, 2.05)
    styles = {
        "GDA": (GDA, "-", "o"),
        "PPM": (PPM, "--", "s"),
        "EGM": (EGM, "-.", "^"),
    }
    paths = trajectories(initial=[1.05, 0.52], step=0.32, iterations=12)
    for name, path in paths.items():
        color, linestyle, marker = styles[name]
        axis.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            ls=linestyle,
            lw=1.35,
            marker=marker,
            ms=2.3,
            markevery=2,
            label=name,
            zorder=5,
        )
    axis.plot(0.0, 0.0, marker="*", ms=7.5, color="#F2C94C", mec="#273746", mew=0.55, zorder=8)
    axis.plot(*paths["GDA"][0], marker="o", ms=3.6, color="#202A31", zorder=8)
    axis.text(paths["GDA"][0, 0] + 0.07, paths["GDA"][0, 1] + 0.05, r"$z_0$", fontsize=6.2)
    axis.legend(loc="upper left", frameon=True, framealpha=0.90, borderpad=0.28, handlelength=2.2)
    axis.set(xlim=(-2.05, 2.05), ylim=(-2.05, 2.05), xlabel=r"$x$", ylabel=r"$y$")
    axis.set_box_aspect(1)
    axis.text(0.5, -0.21, "(a) Bilinear trajectories", transform=axis.transAxes, ha="center", va="top", fontsize=7.2)


def direction_panel(axis):
    draw_energy(axis, 1.55)
    angles = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)
    radius = 1.08
    for angle in angles:
        point = radius * np.array([np.cos(angle), np.sin(angle)])
        field_direction = -field(point)
        curvature_direction = -point  # G=A^2 z=-z for the bilinear field.
        axis.plot(*point, "o", ms=2.2, color="#1D252A", zorder=7)
        for vector, color in ((field_direction, FIELD), (curvature_direction, CURVATURE)):
            unit = vector / np.linalg.norm(vector)
            axis.annotate(
                "",
                xy=point + 0.34 * unit,
                xytext=point,
                arrowprops={"arrowstyle": "-|>", "lw": 1.05, "color": color, "mutation_scale": 7.0},
                zorder=6,
            )
    axis.plot(0.0, 0.0, marker="*", ms=7.5, color="#F2C94C", mec="#273746", mew=0.55, zorder=8)
    handles = [
        Line2D([0], [0], color=FIELD, lw=1.25, marker=">", ms=3.5, label=r"$-F$: rotational"),
        Line2D([0], [0], color=CURVATURE, lw=1.25, marker=">", ms=3.5, label=r"$+G$: inward"),
    ]
    axis.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.90, borderpad=0.28, handlelength=2.1)
    axis.set(xlim=(-1.55, 1.55), ylim=(-1.55, 1.55), xlabel=r"$x$", ylabel=r"$y$")
    axis.set_box_aspect(1)
    axis.text(0.5, -0.21, "(b) Directions on field energy", transform=axis.transAxes, ha="center", va="top", fontsize=7.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(3.48, 1.82))
    trajectory_panel(axes[0])
    direction_panel(axes[1])
    figure.subplots_adjust(left=0.12, right=0.985, top=0.98, bottom=0.22, wspace=0.32)
    figure.savefig(args.output / "mathematical_motivation.pdf", bbox_inches="tight", pad_inches=0.015)
    figure.savefig(args.output / "mathematical_motivation.png", dpi=420, bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)


if __name__ == "__main__":
    main()
