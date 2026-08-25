"""Regenerate the theory-aligned ICC figures from locked and new raw data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import plot_neural_benchmark as bench
import plot_neural_communication_stress as stress


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCKED = ROOT / "results" / "exogenous_ph_v1_20260813_tuned"
DEFAULT_ANALYSIS = ROOT / "results" / "theory_aligned_v1_20260813"


def ci(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    half = float(student_t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, half


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 6.3,
            "axes.titlesize": 6.8,
            "axes.labelsize": 6.3,
            "legend.fontsize": 5.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def oracle_panel(axis: plt.Axes, frame: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for metric, label, color, marker in (
        ("relative_f_error", r"field $\widehat F$", "#0077BB", "o"),
        ("relative_g_error", r"JVP $\widehat G$", "#CC3311", "s"),
    ):
        means, halves = [], []
        for batch in sorted(frame.batch_trajectories.unique()):
            values = frame.loc[frame.batch_trajectories == batch, metric].to_numpy()
            mean, half = ci(values)
            means.append(mean)
            halves.append(half)
        axis.errorbar(
            sorted(frame.batch_trajectories.unique()), means, yerr=halves,
            color=color, marker=marker, linewidth=1.15, markersize=2.8,
            elinewidth=0.65, capsize=1.5, label=label,
        )
        summary[metric] = {"means": means, "ci_half_widths": halves}
    axis.set_yscale("log")
    axis.set_xlabel("trajectories per batch")
    axis.set_ylabel("relative oracle error")
    axis.set_title("(e) Same-batch oracle accuracy")
    axis.grid(axis="y", color="#B8C1C8", alpha=0.32, linewidth=0.4)
    axis.legend(frameon=False, loc="upper right")
    return summary


def neural_figure(locked: Path, analysis: Path, output: Path) -> dict[str, object]:
    hard = pd.read_csv(locked / "icc_neural_hard" / "hard_neural_results.csv")
    trajectory = pd.read_csv(locked / "icc_trajectory_sampled" / "trajectory_results.csv")
    diagnostics = pd.read_csv(analysis / "icc_neural_stress_12seed" / "neural_stress_diagnostics.csv")
    oracle = pd.read_csv(analysis / "oracle_quality_v2" / "oracle_quality_raw.csv")

    fig, axes = plt.subplots(2, 4, figsize=(7.12, 3.55))
    bench.plot_convergence(axes[0, 0], hard, "robust_rate", "(a) Worst-case rate", "bit/s/Hz")
    bench.plot_convergence(axes[0, 1], hard, "hard_exploitability", "(b) Policy exploitability", "return gap")
    bench.plot_convergence(axes[0, 2], hard, "field_norm", "(c) Saddle-field norm", r"$\|F\|$")
    bench.plot_activation(axes[0, 3], diagnostics)
    oracle_summary = oracle_panel(axes[1, 0], oracle)

    channels = [6, 8, 10, 12]
    configs = ["Channels-C6-H64", "Channels-C8-H64", "Stress-C10-H64", "Channels-C12-H64"]
    channel_rate = bench.scale_values(hard, configs, channels, "robust_rate")
    bench.plot_scale(axes[1, 1], channels, channel_rate, "(f) Channel scaling", "channels", "final bit/s/Hz")
    bench.plot_sampled(axes[1, 2], trajectory, "robust_rate", "(g) Sampled-oracle rate", "bit/s/Hz")
    bench.plot_sampled(axes[1, 3], trajectory, "hard_exploitability", "(h) Sampled-oracle exploitability", "return gap")
    axes[1, 1].set_xticks(channels)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), pad=0.52, w_pad=0.55, h_pad=0.66)
    fig.savefig(output / "neural_benchmark_theory.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output / "neural_benchmark_theory.png", dpi=360, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return oracle_summary


def stress_figure(analysis: Path, output: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(analysis / "icc_neural_stress_12seed" / "neural_stress_results.csv")
    fig, axes = plt.subplots(1, 4, figsize=(7.12, 1.68))
    for column, axis_name in enumerate(("jammer", "switch", "leakage", "persistence")):
        stress.plot_metric(axes[column], frame, axis_name, "robust_rate", chr(ord("a") + column), "final bit/s/Hz")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), pad=0.48, w_pad=0.55)
    fig.savefig(output / "wireless_stress_rate.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output / "wireless_stress_rate.png", dpi=360, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return stress.paired_records(frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-root", type=Path, default=DEFAULT_LOCKED)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output is not None else args.analysis_root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    style()
    audit = {
        "oracle_quality": neural_figure(args.locked_root, args.analysis_root, output),
        "stress_paired_results": stress_figure(args.analysis_root, output),
        "figure_2_3_modified": False,
    }
    (output / "figure_statistics.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "stress_cells": len(audit["stress_paired_results"])}, indent=2))


if __name__ == "__main__":
    main()
