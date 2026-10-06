"""Create the queue-aware LCMPL framework figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


BLUE = "#315FA8"
GREEN = "#2E8B57"
RED = "#D73A31"
GRAY = "#555555"


def rounded(axis, xy, width, height, text, edge, face, fontsize=5.3, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        edgecolor=edge,
        facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=edge,
        weight=weight,
    )


def arrow(axis, start, end, color=GRAY, label=None, offset=(0.0, 0.025)):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.75,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )
    if label:
        axis.text(
            (start[0] + end[0]) / 2 + offset[0],
            (start[1] + end[1]) / 2 + offset[1],
            label,
            ha="center",
            va="center",
            fontsize=4.8,
            color=color,
        )


def draw(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(3.5, 2.05))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(Rectangle((0.015, 0.08), 0.40, 0.84, fill=False, edgecolor="#888888", linewidth=0.8, linestyle="--"))
    axis.add_patch(Rectangle((0.47, 0.08), 0.515, 0.84, fill=False, edgecolor=BLUE, linewidth=0.9, linestyle="--"))
    axis.text(0.215, 0.955, "Queue-aware DSA game", ha="center", va="center", fontsize=6.0, weight="bold")
    axis.text(0.728, 0.955, "LCMPL step-size controller", ha="center", va="center", fontsize=6.0, weight="bold", color=BLUE)

    rounded(axis, (0.055, 0.69), 0.32, 0.15, "Markov state\n$s_t=(h_t,x_t,q_t,a_{t-1})$", GRAY, "#F7F7F7", 5.0, "bold")
    rounded(axis, (0.055, 0.43), 0.32, 0.16, "Transmitter $\\pi_\\theta$\nselects channel $a_t$", GREEN, "#F1FBF5", 5.4, "bold")
    rounded(axis, (0.055, 0.18), 0.32, 0.16, "Jammer $\\nu_\\psi$\nselects attack $b_t$", RED, "#FFF3F2", 5.4, "bold")
    arrow(axis, (0.215, 0.69), (0.215, 0.59), GRAY)
    arrow(axis, (0.215, 0.43), (0.215, 0.34), GRAY)

    rounded(axis, (0.51, 0.77), 0.20, 0.10, "Field $\\mathbf{F}_k$", BLUE, "#F4F7FD", 5.2, "bold")
    rounded(axis, (0.75, 0.77), 0.20, 0.10, "Curvature $\\mathbf{G}_k$", BLUE, "#F4F7FD", 5.2, "bold")
    rounded(axis, (0.56, 0.59), 0.34, 0.10, "Communication-aware merit $V(\\mathbf{z})$", BLUE, "#F4F7FD", 5.0)
    rounded(axis, (0.56, 0.42), 0.34, 0.10, "Conditional drift $\\mathcal{D}_k$\nand local model $m_k$", BLUE, "#F4F7FD", 4.9)
    rounded(axis, (0.56, 0.24), 0.34, 0.11, "Box QP selects $(\\beta_k,\\gamma_k)$\nmerit/utility acceptance", BLUE, "#F4F7FD", 4.9, "bold")
    rounded(axis, (0.56, 0.09), 0.34, 0.08, "$\\mathbf{z}_{k+1}=\\mathbf{z}_k-\\beta_k\\mathbf{F}_k+\\gamma_k\\mathbf{G}_k$", BLUE, "#F4F7FD", 4.8)
    arrow(axis, (0.61, 0.77), (0.67, 0.69), BLUE)
    arrow(axis, (0.85, 0.77), (0.80, 0.69), BLUE)
    arrow(axis, (0.73, 0.59), (0.73, 0.52), BLUE)
    arrow(axis, (0.73, 0.42), (0.73, 0.35), BLUE)
    arrow(axis, (0.73, 0.24), (0.73, 0.17), BLUE)
    arrow(axis, (0.375, 0.51), (0.51, 0.82), GRAY)
    arrow(axis, (0.56, 0.13), (0.375, 0.26), BLUE)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"algorithm_framework_queue.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    draw(arguments.output)
