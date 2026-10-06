"""Plot and summarize the locked queue-aware DSA experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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


METHODS = ("QP+G", "noG", "PPM-3")
LABELS = {"QP+G": "LCMPL", "noG": "LCMPL-F", "PPM-3": "PPM-3"}
COLORS = {"QP+G": "#111111", "noG": "#009E9A", "PPM-3": "#B24AA7"}
MARKERS = {"QP+G": "o", "noG": "s", "PPM-3": "^"}
LOAD_LABELS = {0.20: "light", 0.45: "moderate", 0.70: "near-cap."}
METRICS = (
    ("robust_rate", "worst-case utility", "(a) Queue-aware utility", 1.0),
    ("br_goodput", "goodput (packet/slot)", "(b) Delivered goodput", 1.0),
    ("br_backlog", "average backlog (packet)", "(c) Queue backlog", -1.0),
    ("br_drop_probability", "drop probability", "(d) Packet dropping", -1.0),
)


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, 0.0
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return mean, half


def paired_record(
    selected: pd.DataFrame,
    arrival_rate: float,
    comparator: str,
    metric: str,
    direction: float,
) -> Dict[str, float | str]:
    indexed = selected[selected["arrival_rate"] == arrival_rate].set_index(["seed", "method"])
    proposed = indexed.xs("QP+G", level="method")[metric]
    baseline = indexed.xs(comparator, level="method")[metric]
    improvement = direction * (proposed - baseline)
    mean, half = mean_ci(improvement.to_numpy(dtype=float))
    return {
        "arrival_rate": float(arrival_rate),
        "load": LOAD_LABELS[round(float(arrival_rate), 2)],
        "comparator": LABELS[comparator],
        "metric": metric,
        "improvement_direction": "higher" if direction > 0 else "lower",
        "paired_mean_improvement": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "seed_count": int(len(improvement)),
    }


def plot(input_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_csv)
    selected = data[data["step"] == data["step"].max()].copy()
    loads = sorted(float(value) for value in selected["arrival_rate"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.88), constrained_layout=True)
    for axis, (metric, ylabel, title, _) in zip(axes, METRICS):
        for method in METHODS:
            means: List[float] = []
            errors: List[float] = []
            for load in loads:
                values = selected[
                    (selected["arrival_rate"] == load) & (selected["method"] == method)
                ][metric].to_numpy(dtype=float)
                mean, half = mean_ci(values)
                means.append(mean)
                errors.append(half)
            axis.errorbar(
                loads,
                means,
                yerr=errors,
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=3.2,
                linewidth=1.15,
                capsize=2.0,
                label=LABELS[method],
            )
        axis.set_xticks(loads, [LOAD_LABELS[round(load, 2)] for load in loads])
        axis.tick_params(axis="x", labelrotation=0, labelsize=6.6)
        axis.tick_params(axis="y", labelsize=6.8)
        axis.grid(True, alpha=0.25, linewidth=0.5)
        axis.set_ylabel(ylabel, fontsize=7.3)
        axis.set_title(title, fontsize=7.8, y=-0.36)
        axis.set_box_aspect(1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
        fontsize=7.5,
        handlelength=1.8,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"queue_load_results.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)
    summaries: List[Dict[str, object]] = []
    for load in loads:
        for method in METHODS:
            subset = selected[
                (selected["arrival_rate"] == load) & (selected["method"] == method)
            ]
            record: Dict[str, object] = {
                "arrival_rate": float(load),
                "load": LOAD_LABELS[round(load, 2)],
                "method": LABELS[method],
                "seed_count": int(len(subset)),
            }
            for metric, _, _, _ in METRICS:
                mean, half = mean_ci(subset[metric].to_numpy(dtype=float))
                record[f"{metric}_mean"] = mean
                record[f"{metric}_ci95_low"] = mean - half
                record[f"{metric}_ci95_high"] = mean + half
            summaries.append(record)
    paired = [
        paired_record(selected, load, comparator, metric, direction)
        for load in loads
        for comparator in ("noG", "PPM-3")
        for metric, _, _, direction in METRICS
    ]
    statistics = {
        "source": str(input_csv),
        "final_step": int(selected["step"].max()),
        "loads": [LOAD_LABELS[round(load, 2)] for load in loads],
        "summary": summaries,
        "paired_comparisons": paired,
    }
    (input_csv.parent / "queue_statistics.json").write_text(
        json.dumps(statistics, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/queue_aware/queue_results.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    plot(arguments.input, arguments.output)
