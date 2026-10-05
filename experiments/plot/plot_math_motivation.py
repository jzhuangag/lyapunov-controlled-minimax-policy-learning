"""Draw the ICC-specific coupling-adaptation motivation figure.

The figure uses the scaled bilinear saddle field F_omega(z)=omega Jz to show
why the field-to-curvature step-size ratio must respond to local coupling.
Both panels are analytic mechanism illustrations rather than benchmark results.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "figures"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 6.6,
        "axes.labelsize": 6.8,
        "legend.fontsize": 5.5,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

FIXED = "#7B4AB5"
ADAPTIVE = "#111111"
NEUTRAL = "#6E7B83"
FIELD = "#1F77B4"
CURVATURE = "#2A8C64"
STAR = "#F2C94C"


def amplification(omega, ratio, beta):
    """One-step norm factor for z_+=z-beta F_omega+gamma G_omega."""
    gamma = beta * ratio
    return np.sqrt((1.0 - gamma * omega**2) ** 2 + (beta * omega) ** 2)


def draw_energy(axis, limit=1.55):
    grid = np.linspace(-limit, limit, 240)
    x_grid, y_grid = np.meshgrid(grid, grid)
    energy = 0.5 * (x_grid**2 + y_grid**2)
    levels = np.linspace(0.15, 2.0, 8)
    axis.contourf(
        x_grid,
        y_grid,
        energy,
        levels=np.r_[0.0, levels, 3.0],
        cmap="Blues",
        alpha=0.30,
    )
    axis.contour(
        x_grid,
        y_grid,
        energy,
        levels=levels,
        colors="#78909C",
        linewidths=0.40,
        alpha=0.65,
    )
    axis.axhline(0.0, color="#AAB4BA", lw=0.35)
    axis.axvline(0.0, color="#AAB4BA", lw=0.35)


def direction_panel(axis, curvature=False):
    draw_energy(axis)
    coordinates = np.linspace(-1.25, 1.25, 9)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    if curvature:
        u_grid = -x_grid
        v_grid = -y_grid
        color = CURVATURE
        label = r"$+G$: inward"
        panel = "(b) Curvature action"
    else:
        u_grid = -y_grid
        v_grid = x_grid
        color = FIELD
        label = r"$-F$: rotational"
        panel = "(a) Field action"
    norm = np.sqrt(u_grid**2 + v_grid**2)
    nonzero = norm > 0.0
    u_grid = np.divide(u_grid, norm, out=np.zeros_like(u_grid), where=nonzero)
    v_grid = np.divide(v_grid, norm, out=np.zeros_like(v_grid), where=nonzero)
    axis.quiver(
        x_grid,
        y_grid,
        u_grid,
        v_grid,
        color=color,
        angles="xy",
        scale_units="xy",
        scale=5.2,
        width=0.010,
        headwidth=3.6,
        headlength=4.4,
        zorder=4,
    )
    axis.plot(0.0, 0.0, marker="*", ms=7.0, color=STAR, mec="#273746", mew=0.55, zorder=6)
    axis.legend(
        [Line2D([0], [0], color=color, lw=1.4)],
        [label],
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        borderpad=0.22,
        handlelength=1.7,
    )
    axis.set(xlim=(-1.55, 1.55), ylim=(-1.55, 1.55), xlabel=r"$x$", ylabel=r"$y$")
    axis.set_box_aspect(1)
    axis.xaxis.labelpad = 0.3
    axis.text(0.5, -0.22, panel, transform=axis.transAxes, ha="center", va="top", fontsize=7.0)


def ratio_map_panel(axis, beta):
    omega = np.linspace(1.2, 4.2, 260)
    ratio = np.linspace(0.0, 12.0, 260)
    omega_grid, ratio_grid = np.meshgrid(omega, ratio)
    factor = amplification(omega_grid, ratio_grid, beta)
    axis.pcolormesh(omega, ratio, factor, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0.05, vcenter=1.0, vmax=2.0), shading="auto", rasterized=True)
    axis.contour(omega, ratio, factor, levels=[1.0], colors="white", linewidths=1.1, linestyles="--")
    fixed_ratio = 4.0
    adaptive_ratio = np.clip(1.0 / (beta * omega**2), ratio.min(), ratio.max())
    axis.plot(omega, np.full_like(omega, fixed_ratio), color=FIXED, lw=1.25, ls="--")
    axis.plot(omega, adaptive_ratio, color=ADAPTIVE, lw=1.35)
    axis.text(3.72, 10.5, r"$q>1$", color="#7A1E1E", ha="center", fontsize=6.0)
    axis.text(2.12, 1.05, r"$q<1$", color="#173F6A", ha="center", fontsize=6.0)
    handles = [
        Line2D([0], [0], color=FIXED, lw=1.25, ls="--", label=r"fixed $\gamma/\beta=4$"),
        Line2D([0], [0], color=ADAPTIVE, lw=1.35, label="local minimum"),
        Line2D([0], [0], color="white", markeredgecolor=NEUTRAL, marker="_", lw=1.0, ls="--", label=r"$q=1$ boundary"),
    ]
    axis.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.92, borderpad=0.24, handlelength=1.8)
    axis.set(xlim=(1.2, 4.2), ylim=(0.0, 12.0), xlabel=r"local coupling $\omega$", ylabel=r"step-size ratio $\gamma/\beta$")
    axis.set_box_aspect(1)
    axis.xaxis.labelpad = 0.5
    axis.text(0.5, -0.22, "(c) Coupling-dependent contraction", transform=axis.transAxes, ha="center", va="top", fontsize=7.0)


def switching_panel(axis, beta):
    segment_length = 18
    mode_couplings = [1.4, 2.8, 4.2, 2.0]
    omega = np.repeat(mode_couplings, segment_length)
    updates = np.arange(omega.size)
    field_only = amplification(omega, np.zeros_like(omega), beta)
    fixed = amplification(omega, np.full_like(omega, 4.0), beta)
    adaptive_ratio = np.clip(1.0 / (beta * omega**2), 0.0, 12.0)
    adaptive = amplification(omega, adaptive_ratio, beta)
    axis.axhline(1.0, color=NEUTRAL, lw=0.85, ls=":", zorder=1)
    axis.step(updates, field_only, where="post", color=FIELD, lw=1.15, label="field only")
    axis.step(updates, fixed, where="post", color=FIXED, lw=1.2, ls="--", label="fixed ratio")
    axis.step(updates, adaptive, where="post", color=ADAPTIVE, lw=1.3, label="local ratio")
    for index, coupling in enumerate(mode_couplings):
        start = index * segment_length
        if index > 0:
            axis.axvline(start, color="#B5BEC3", lw=0.45)
    axis.text(69.5, 1.03, r"$q=1$", ha="right", va="bottom", color=NEUTRAL, fontsize=5.7)
    axis.legend(loc="upper left", ncol=1, frameon=True, framealpha=0.92, borderpad=0.24, handlelength=1.8)
    axis.set(xlim=(0, omega.size - 1), ylim=(0.0, 2.0), xlabel="joint update", ylabel=r"one-step factor $q$")
    axis.set_xticks([0, 18, 36, 54, 71])
    top_axis = axis.secondary_xaxis("top")
    top_axis.set_xticks([9, 27, 45, 63])
    top_axis.set_xticklabels([rf"$\omega={value:.1f}$" for value in mode_couplings])
    top_axis.tick_params(axis="x", length=0, pad=1.5, labelsize=5.7)
    axis.set_box_aspect(1)
    axis.xaxis.labelpad = 0.5
    axis.text(0.5, -0.22, "(d) Switching local coupling", transform=axis.transAxes, ha="center", va="top", fontsize=7.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    beta = 0.04
    figure, axes = plt.subplots(2, 2, figsize=(3.48, 3.20))
    direction_panel(axes[0, 0], curvature=False)
    direction_panel(axes[0, 1], curvature=True)
    ratio_map_panel(axes[1, 0], beta)
    switching_panel(axes[1, 1], beta)
    figure.subplots_adjust(left=0.12, right=0.985, top=0.97, bottom=0.10, wspace=0.38, hspace=0.40)
    figure.savefig(args.output / "mathematical_motivation.pdf", bbox_inches="tight", pad_inches=0.015)
    figure.savefig(args.output / "mathematical_motivation.png", dpi=420, bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)


if __name__ == "__main__":
    main()
