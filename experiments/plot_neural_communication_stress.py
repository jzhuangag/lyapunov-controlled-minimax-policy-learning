"""Plot and audit the neural communication-parameter stress sweep."""

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
RESULT_DIR = RESULT_ROOT / "icc_neural_stress"
METHODS = ("QP+G", "noG", "PPM-3")
COLORS = {"QP+G": "#111111", "noG": "#009988", "PPM-3": "#AA4499"}
LINESTYLES = {"QP+G": "-", "noG": "--", "PPM-3": (0, (3, 1, 1, 1))}
MARKERS = {"QP+G": "o", "noG": "s", "PPM-3": "D"}
LABELS = {"QP+G": "LCMPL", "noG": "LCMPL-F", "PPM-3": "PPM-3"}
AXES = {
    "jammer": {
        "prefix": "Jammer-",
        "values": [12.0, 16.0, 20.0, 24.0],
        "labels": ["12", "16", "20", "24"],
        "xlabel": r"jammer-to-noise ratio $\rho_J$",
        "title": "Jammer strength",
    },
    "switch": {
        "prefix": "Switch-",
        "values": [0.02, 0.06, 0.10, 0.14],
        "labels": ["0.02", "0.06", "0.10", "0.14"],
        "xlabel": r"switching cost $c_{\rm sw}$",
        "title": "Switching overhead",
    },
    "leakage": {
        "prefix": "Leakage-",
        "values": [0.20, 0.40, 0.60, 0.80],
        "labels": ["0.2", "0.4", "0.6", "0.8"],
        "xlabel": r"adjacent leakage $\lambda$",
        "title": "Adjacent interference",
    },
    "persistence": {
        "prefix": "Persistence-",
        "values": [0.40, 0.55, 0.70, 0.85],
        "labels": ["0.40", "0.55", "0.70", "0.85"],
        "xlabel": r"radio-mode persistence $\rho_h$",
        "title": "Mode persistence",
    },
}


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, half


def final_values(frame: pd.DataFrame, configuration: str, method: str, metric: str) -> np.ndarray:
    selected = frame[(frame["config"] == configuration) & (frame["method"] == method)]
    return selected.sort_values("step").groupby("seed", as_index=False).tail(1)[metric].to_numpy()


def configuration_name(axis_name: str, value: float) -> str:
    if axis_name == "jammer":
        return f"Jammer-{value:.0f}"
    return f"{AXES[axis_name]['prefix']}{value:.2f}"


def plot_metric(axis: plt.Axes, frame: pd.DataFrame, axis_name: str, metric: str, panel: str, ylabel: str) -> None:
    specification = AXES[axis_name]
    x_values = np.arange(len(specification["values"]))
    for method in METHODS:
        means = []
        halves = []
        for value in specification["values"]:
            samples = final_values(frame, configuration_name(axis_name, value), method, metric)
            mean, half = mean_ci(samples)
            means.append(mean)
            halves.append(half)
        axis.errorbar(
            x_values,
            means,
            yerr=halves,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            markersize=2.6,
            linewidth=1.15,
            elinewidth=0.65,
            capsize=1.5,
            label=LABELS[method],
        )
    axis.set_title(f"({panel}) {specification['title']}")
    axis.set_xticks(x_values, specification["labels"])
    axis.set_xlabel(specification["xlabel"])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#B8C1C8", alpha=0.32, linewidth=0.4)


def paired_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for axis_name, specification in AXES.items():
        for value in specification["values"]:
            configuration = configuration_name(axis_name, value)
            for comparator in ("noG", "PPM-3"):
                qp_rate = final_values(frame, configuration, "QP+G", "robust_rate")
                comparison_rate = final_values(frame, configuration, comparator, "robust_rate")
                qp_gap = final_values(frame, configuration, "QP+G", "hard_exploitability")
                comparison_gap = final_values(frame, configuration, comparator, "hard_exploitability")
                rate = qp_rate - comparison_rate
                gap = comparison_gap - qp_gap
                rate_mean, rate_half = mean_ci(rate)
                gap_mean, gap_half = mean_ci(gap)
                records.append(
                    {
                        "axis": axis_name,
                        "value": value,
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
                )
    return records


def main() -> None:
    frame = pd.read_csv(RESULT_DIR / "neural_stress_results.csv")
    diagnostics = pd.read_csv(RESULT_DIR / "neural_stress_diagnostics.csv")
    records = paired_records(frame)
    audit = {
        "metric_direction": {"robust_rate": "larger is better", "hard_exploitability": "smaller is better"},
        "paired_results": records,
        "qp_curvature_activation_fraction": float(
            (diagnostics[diagnostics["method"] == "QP+G"]["gamma"] > 1.0e-10).mean()
        ),
        "maximum_best_response_residual": float(frame["maximum_br_residual"].max()),
    }
    (RESULT_DIR / "stress_statistics.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 6.4,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.3,
            "legend.fontsize": 6.0,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(7.12, 3.15))
    for column, axis_name in enumerate(("jammer", "switch", "leakage", "persistence")):
        plot_metric(axes[0, column], frame, axis_name, "robust_rate", chr(ord("a") + column), "final bit/s/Hz")
        plot_metric(axes[1, column], frame, axis_name, "hard_exploitability", chr(ord("e") + column), "final return gap")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.002))
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 1.0), pad=0.50, w_pad=0.58, h_pad=0.65)
    fig.savefig(RESULT_DIR / "neural_communication_stress.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(RESULT_DIR / "neural_communication_stress.png", dpi=360, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
