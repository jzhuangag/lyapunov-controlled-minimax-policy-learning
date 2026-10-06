"""Plot and summarize the locked six-method queue-aware DSA benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


METHODS = ("QP+G", "noG", "Minimax-PPO", "GDA", "EGM", "PPM-3")
LABELS = {
    "QP+G": "LCMPL",
    "noG": "LCMPL-F",
    "Minimax-PPO": "Minimax PPO",
    "GDA": "GDA",
    "EGM": "EGM",
    "PPM-3": "PPM-3",
}
COLORS = {
    "QP+G": "#111111",
    "noG": "#009E9A",
    "Minimax-PPO": "#CC3311",
    "GDA": "#EE7733",
    "EGM": "#0077BB",
    "PPM-3": "#AA4499",
}
LINESTYLES = {
    "QP+G": "-",
    "noG": "--",
    "Minimax-PPO": (0, (4, 1)),
    "GDA": ":",
    "EGM": "-.",
    "PPM-3": (0, (3, 1, 1, 1)),
}
MARKERS = {
    "QP+G": "o",
    "noG": "s",
    "Minimax-PPO": "P",
    "GDA": "^",
    "EGM": "v",
    "PPM-3": "D",
}
TICK_LABELS = {
    "QP+G": "LCMPL",
    "noG": "LCMPL-F",
    "Minimax-PPO": "M-PPO",
    "GDA": "GDA",
    "EGM": "EGM",
    "PPM-3": "PPM-3",
}
FINAL_METRICS = (
    ("robust_rate", 1.0),
    ("hard_exploitability", -1.0),
    ("br_goodput", 1.0),
    ("br_backlog", -1.0),
    ("br_drop_probability", -1.0),
)


def mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, 0.0
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return mean, half


def plot_curve(axis: plt.Axes, data: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    for method in METHODS:
        selected = data[data["method"] == method]
        pivot = selected.pivot(index="seed", columns="step", values=metric).sort_index(axis=1)
        mean, sem = mean_sem(pivot.to_numpy(dtype=float))
        steps = pivot.columns.to_numpy(dtype=float)
        axis.plot(
            steps,
            mean,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.15,
            label=LABELS[method],
        )
        axis.fill_between(steps, mean - sem, mean + sem, color=COLORS[method], alpha=0.08, linewidth=0)
    axis.set_xlabel("joint updates", fontsize=7.3)
    axis.set_ylabel(ylabel, fontsize=7.3)
    axis.set_title(title, fontsize=7.6, y=-0.35)
    axis.tick_params(labelsize=6.8)
    axis.grid(True, alpha=0.25, linewidth=0.45)
    axis.set_box_aspect(1)


def plot_final(axis: plt.Axes, final: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    means: List[float] = []
    errors: List[float] = []
    for method in METHODS:
        values = final[final["method"] == method][metric].to_numpy(dtype=float)
        mean, half = mean_ci(values)
        means.append(mean)
        errors.append(half)
    y = np.arange(len(METHODS))
    for index, method in enumerate(METHODS):
        axis.errorbar(
            means[index],
            y[index],
            xerr=errors[index],
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=3.5,
            capsize=2.0,
            linewidth=1.0,
        )
    axis.set_yticks(y, [TICK_LABELS[method] for method in METHODS])
    axis.invert_yaxis()
    axis.set_xlabel(ylabel, fontsize=7.3)
    axis.set_title(title, fontsize=7.6, y=-0.35)
    axis.tick_params(labelsize=6.0)
    axis.grid(axis="x", alpha=0.25, linewidth=0.45)
    axis.set_box_aspect(1)


def paired_record(final: pd.DataFrame, comparator: str, metric: str, direction: float) -> Dict[str, object]:
    indexed = final.set_index(["seed", "method"])
    proposed = indexed.xs("QP+G", level="method")[metric]
    baseline = indexed.xs(comparator, level="method")[metric]
    improvement = direction * (proposed - baseline)
    mean, half = mean_ci(improvement.to_numpy(dtype=float))
    return {
        "comparator": LABELS[comparator],
        "metric": metric,
        "improvement_direction": "higher" if direction > 0 else "lower",
        "paired_mean_improvement": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "seed_wins": int(np.sum(improvement.to_numpy(dtype=float) > 0.0)),
        "seed_count": int(len(improvement)),
    }


def plot(input_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_csv)
    final_step = int(data["step"].max())
    final = data[data["step"] == final_step].copy()
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.05), constrained_layout=True)
    plot_curve(axes[0], data, "robust_rate", "(a) Worst-case queue utility", "utility")
    plot_curve(axes[1], data, "hard_exploitability", "(b) Policy exploitability", "exploitability")
    plot_final(axes[2], final, "br_goodput", "(c) Delivered goodput", "packet/slot")
    plot_final(axes[3], final, "br_backlog", "(d) Queue backlog", "packet")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
        fontsize=6.8,
        handlelength=1.8,
        columnspacing=0.8,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"queue_nominal_benchmark.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)

    summary: List[Dict[str, object]] = []
    for method in METHODS:
        subset = final[final["method"] == method]
        record: Dict[str, object] = {
            "method": LABELS[method],
            "seed_count": int(len(subset)),
        }
        for metric, _ in FINAL_METRICS:
            mean, half = mean_ci(subset[metric].to_numpy(dtype=float))
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci95_low"] = mean - half
            record[f"{metric}_ci95_high"] = mean + half
        summary.append(record)
    comparisons = [
        paired_record(final, comparator, metric, direction)
        for comparator in METHODS
        if comparator != "QP+G"
        for metric, direction in FINAL_METRICS
    ]
    statistics = {
        "source": str(input_csv),
        "final_step": final_step,
        "summary": summary,
        "paired_comparisons": comparisons,
    }
    (input_csv.parent / "queue_nominal_statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/queue_nominal/queue_nominal_results.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    plot(arguments.input, arguments.output)
