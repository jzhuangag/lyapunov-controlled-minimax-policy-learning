"""Create the queue-aware adversarial DSA system-model figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


BLUE = "#315FA8"
GREEN = "#2E8B57"
RED = "#D73A31"
ORANGE = "#D9822B"
GRAY = "#555555"


def box(axis, xy, width, height, text, edge, face="#FFFFFF", fontsize=6.0, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
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


def arrow(axis, start, end, color=GRAY, text=None, text_offset=(0.0, 0.035), style="-|>"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=0.8,
        color=color,
        shrinkA=1.5,
        shrinkB=1.5,
    )
    axis.add_patch(patch)
    if text:
        axis.text(
            (start[0] + end[0]) / 2 + text_offset[0],
            (start[1] + end[1]) / 2 + text_offset[1],
            text,
            ha="center",
            va="center",
            fontsize=5.4,
            color=color,
        )


def draw(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(3.5, 1.48))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    box(axis, (0.03, 0.62), 0.19, 0.22, "Radio mode $h_t$\n(fading, occupancy)", BLUE, "#F4F7FD", 5.2, "bold")
    box(axis, (0.30, 0.64), 0.18, 0.18, "Channel profiles\n$g_t, I_t^{\\rm pri}, g_t^J$", BLUE, "#F4F7FD", 5.2)
    arrow(axis, (0.22, 0.73), (0.30, 0.73), BLUE)

    box(axis, (0.02, 0.18), 0.12, 0.19, "Arrivals\n$A_t$", ORANGE, "#FFF7EC", 5.9, "bold")
    box(axis, (0.20, 0.15), 0.15, 0.25, "Finite queue\n$q_t\\in[0,Q_{\\max}]$", ORANGE, "#FFF7EC", 5.1, "bold")
    arrow(axis, (0.14, 0.275), (0.20, 0.275), ORANGE)

    box(axis, (0.42, 0.16), 0.15, 0.23, "Tx policy $\\pi_\\theta$\nselects $a_t$", GREEN, "#F1FBF5", 5.9, "bold")
    arrow(axis, (0.35, 0.275), (0.42, 0.275), GREEN)
    arrow(axis, (0.39, 0.64), (0.49, 0.39), BLUE, "$g_t,I_t$")

    box(axis, (0.63, 0.59), 0.14, 0.20, "Jammer $\\nu_\\psi$\nselects $b_t$", RED, "#FFF3F2", 5.2, "bold")
    box(axis, (0.67, 0.16), 0.16, 0.23, "Service $S_t$\n$\\Pr(S_t=1)=p_{\\rm s}$\n$(h_t,a_t,b_t)$", GRAY, "#F7F7F7", 4.9, "bold")
    arrow(axis, (0.57, 0.275), (0.67, 0.275), GREEN, "$a_t$")
    arrow(axis, (0.70, 0.59), (0.74, 0.39), RED, "$b_t$")
    arrow(axis, (0.48, 0.73), (0.67, 0.34), BLUE)

    box(axis, (0.85, 0.14), 0.14, 0.27, "Queue update\n$q_{t+1}=f(q_t,S_t,A_t)$", ORANGE, "#FFF7EC", 5.0, "bold")
    arrow(axis, (0.83, 0.275), (0.86, 0.275), ORANGE, "$S_t$")
    arrow(axis, (0.92, 0.14), (0.275, 0.15), ORANGE, "action-dependent next state", (0.0, -0.045), "->")

    for suffix in ("pdf", "png"):
        fig.savefig(output / f"system_model_queue.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    draw(arguments.output)
