"""Create the eight-panel neural-policy benchmark figure and statistics audit."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = Path(__import__("os").environ.get("ICC_RESULT_ROOT", ROOT / "results"))
SCALE_DIR = RESULT_ROOT / "icc_neural_scale"
HARD_DIR = RESULT_ROOT / "icc_neural_hard"
LARGE_DIR = RESULT_ROOT / "icc_neural_large"
STRESS_DIR = RESULT_ROOT / "icc_neural_stress"
TRAJECTORY_DIR = RESULT_ROOT / "icc_trajectory_sampled"
OUT_DIR = RESULT_ROOT / "icc_neural_benchmark"
METHODS = ("QP+G", "noG", "Minimax-PPO", "GDA", "EGM", "PPM-3")
SCALE_METHODS = ("QP+G", "noG", "PPM-3")
COLORS = {
    "QP+G": "#111111",
    "noG": "#009988",
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
LABELS = {
    "QP+G": "LCMPL",
    "noG": "LCMPL-F",
    "Minimax-PPO": "Minimax PPO",
    "GDA": "SimPG",
    "EGM": "ExtraPG",
    "PPM-3": "PPM-3",
}


def mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    half = float(student_t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean, half


def paired_record(frame: pd.DataFrame, configuration: str, comparator: str) -> dict[str, object]:
    subset = frame[frame["config"] == configuration]
    final = subset.sort_values("step").groupby(["seed", "method"], as_index=False).tail(1)
    wide = final.pivot(index="seed", columns="method", values=["robust_rate", "hard_exploitability"])
    rate = (wide["robust_rate"]["QP+G"] - wide["robust_rate"][comparator]).to_numpy()
    gap = (wide["hard_exploitability"][comparator] - wide["hard_exploitability"]["QP+G"]).to_numpy()
    rate_mean, rate_half = mean_ci(rate)
    gap_mean, gap_half = mean_ci(gap)
    return {
        "configuration": configuration,
        "comparator": comparator,
        "seed_count": len(rate),
        "rate_gain_mean": rate_mean,
        "rate_gain_95ci": [rate_mean - rate_half, rate_mean + rate_half],
        "rate_seed_wins": int(np.sum(rate > 0.0)),
        "gap_reduction_mean": gap_mean,
        "gap_reduction_95ci": [gap_mean - gap_half, gap_mean + gap_half],
        "gap_seed_wins": int(np.sum(gap > 0.0)),
    }


def plot_convergence(axis: plt.Axes, frame: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    subset = frame[frame["config"] == "Hard-C8-H64"]
    for method in METHODS:
        selected = subset[subset["method"] == method]
        pivot = selected.pivot(index="seed", columns="step", values=metric).sort_index(axis=1)
        mean, sem = mean_sem(pivot.to_numpy())
        steps = pivot.columns.to_numpy()
        axis.plot(
            steps,
            mean,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.25,
            label=LABELS[method],
        )
        axis.fill_between(steps, mean - sem, mean + sem, color=COLORS[method], alpha=0.09, linewidth=0)
    axis.set_title(title)
    axis.set_xlabel("joint updates")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#B8C1C8", alpha=0.32, linewidth=0.4)


def plot_activation(axis: plt.Axes, diagnostics: pd.DataFrame) -> None:
    records = []
    for leakage in (0.20, 0.40, 0.60, 0.80):
        configuration = f"Leakage-{leakage:.2f}"
        selected = diagnostics[
            (diagnostics["config"] == configuration) & (diagnostics["method"] == "QP+G")
        ].copy()
        selected["active"] = (selected["gamma"] > 1.0e-10).astype(float)
        per_seed = selected.groupby("seed")["active"].mean().to_numpy()
        mean, half = mean_ci(per_seed)
        records.append((leakage, mean, half))
    x_values = np.array([record[0] for record in records])
    means = np.array([record[1] for record in records])
    halves = np.array([record[2] for record in records])
    axis.errorbar(
        x_values,
        means,
        yerr=halves,
        color="#6A51A3",
        marker="o",
        markersize=3.0,
        linewidth=1.25,
        elinewidth=0.7,
        capsize=1.7,
    )
    axis.set_ylim(-0.04, 1.04)
    axis.set_title("(d) Curvature activation")
    axis.set_xlabel(r"adjacent leakage $\lambda$")
    axis.set_ylabel("active-update fraction")
    axis.grid(axis="y", color="#B8C1C8", alpha=0.32, linewidth=0.4)


def scale_values(
    frame: pd.DataFrame,
    configurations: list[str],
    x_values: list[int],
    metric: str,
    methods: tuple[str, ...] = SCALE_METHODS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in methods:
        means = []
        halves = []
        for configuration in configurations:
            selected = frame[(frame["config"] == configuration) & (frame["method"] == method)]
            final = selected.sort_values("step").groupby("seed", as_index=False).tail(1)[metric].to_numpy()
            mean, half = mean_ci(final)
            means.append(mean)
            halves.append(half)
        output[method] = np.asarray(means), np.asarray(halves)
    return output


def plot_scale(
    axis: plt.Axes,
    x_values: list[int],
    values: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    xlabel: str,
    ylabel: str,
    methods: tuple[str, ...] = SCALE_METHODS,
) -> None:
    for method in methods:
        means, halves = values[method]
        axis.errorbar(
            x_values,
            means,
            yerr=halves,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            markersize=3.0,
            linewidth=1.2,
            elinewidth=0.7,
            capsize=1.7,
            label=LABELS[method],
        )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#B8C1C8", alpha=0.32, linewidth=0.4)


def plot_sampled(
    axis: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    for method in ("QP+G", "noG", "GDA", "EGM", "PPM-3"):
        selected = frame[frame["method"] == method].copy()
        selected["interactions_k"] = selected["cumulative_transitions"] / 1000.0
        pivot = selected.pivot(index="seed", columns="interactions_k", values=metric).sort_index(axis=1)
        mean, sem = mean_sem(pivot.to_numpy())
        interactions = pivot.columns.to_numpy()
        axis.plot(
            interactions,
            mean,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.25,
            label=LABELS[method],
        )
        axis.fill_between(
            interactions,
            mean - sem,
            mean + sem,
            color=COLORS[method],
            alpha=0.09,
            linewidth=0,
        )
    axis.set_title(title)
    axis.set_xlabel(r"interactions ($\times 10^3$)")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#B8C1C8", alpha=0.32, linewidth=0.4)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scale = pd.read_csv(SCALE_DIR / "neural_results.csv")
    hard = pd.read_csv(HARD_DIR / "hard_neural_results.csv")
    large = pd.read_csv(LARGE_DIR / "hard_neural_results.csv")
    diagnostics = pd.read_csv(HARD_DIR / "hard_neural_diagnostics.csv")
    stress_diagnostics = pd.read_csv(STRESS_DIR / "neural_stress_diagnostics.csv")
    scale_diagnostics = pd.read_csv(SCALE_DIR / "neural_diagnostics.csv")
    trajectory = pd.read_csv(TRAJECTORY_DIR / "trajectory_results.csv")
    trajectory_statistics = json.loads(
        (TRAJECTORY_DIR / "trajectory_statistics.json").read_text(encoding="utf-8")
    )
    combined = pd.concat((scale, hard, large), ignore_index=True)
    audit = {
        "metric_direction": {
            "robust_rate": "larger is better",
            "hard_exploitability": "smaller is better",
        },
        "paired_confirmatory_results": [
            paired_record(combined, configuration, comparator)
            for configuration in ("Hard-C8-H64", "Stress-C10-H64", "Channels-C12-H64")
            for comparator in ("noG", "PPM-3")
        ]
        + [paired_record(combined, "Hard-C8-H64", "Minimax-PPO")],
        "curvature_activation_by_width": {
            configuration: float(
                (
                    scale_diagnostics[
                        (scale_diagnostics["config"] == configuration)
                        & (scale_diagnostics["method"] == "QP+G")
                    ]["gamma"]
                    > 1.0e-10
                ).mean()
            )
            for configuration in ("Width-C8-H8", "Width-C8-H16", "Width-C8-H32", "Hard-C8-H64")
        },
        "maximum_best_response_residual": float(combined["maximum_br_residual"].max()),
        "trajectory_sampled": trajectory_statistics,
    }
    (OUT_DIR / "neural_statistics.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 6.4,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.5,
            "legend.fontsize": 6.0,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(7.12, 3.62))
    plot_convergence(axes[0, 0], hard, "robust_rate", "(a) Worst-case rate", "bit/s/Hz")
    plot_convergence(axes[0, 1], hard, "hard_exploitability", "(b) Policy exploitability", "return gap")
    plot_convergence(axes[0, 2], hard, "field_norm", "(c) Saddle-field norm", r"$\|F\|$")
    plot_activation(axes[0, 3], stress_diagnostics)

    widths = [8, 16, 32, 64]
    width_configs = [f"Width-C8-H{width}" if width < 64 else "Hard-C8-H64" for width in widths]
    ablations = ("QP+G", "noG")
    width_rate = scale_values(combined, width_configs, widths, "robust_rate", methods=ablations)
    plot_scale(axes[1, 0], widths, width_rate, "(e) Policy-width scaling", "hidden units", "final bit/s/Hz", methods=ablations)

    channels = [6, 8, 10, 12]
    channel_configs = ["Channels-C6-H64", "Channels-C8-H64", "Stress-C10-H64", "Channels-C12-H64"]
    channel_rate = scale_values(combined, channel_configs, channels, "robust_rate")
    plot_scale(axes[1, 1], channels, channel_rate, "(f) Channel scaling", "channels", "final bit/s/Hz")
    plot_sampled(axes[1, 2], trajectory, "robust_rate", "(g) Sampled-oracle rate", "bit/s/Hz")
    plot_sampled(
        axes[1, 3],
        trajectory,
        "hard_exploitability",
        "(h) Sampled-oracle exploitability",
        "return gap",
    )
    axes[1, 0].set_xticks(widths)
    axes[1, 1].set_xticks(channels)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.002))
    fig.tight_layout(rect=(0.0, 0.065, 1.0, 1.0), pad=0.52, w_pad=0.56, h_pad=0.64)
    fig.savefig(OUT_DIR / "neural_benchmark.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_DIR / "neural_benchmark.png", dpi=360, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
