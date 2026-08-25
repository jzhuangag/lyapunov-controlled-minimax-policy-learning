"""Validate fixed learning rates on disjoint seeds for exogenous-ph-v1.

The primary selection criterion is final mean worst-case spectral efficiency.
Final mean hard exploitability breaks an exact tie.  Test seeds are never used.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import run_neural_scale_game as core


SEEDS = (0, 1)
STEPS = 30
GRIDS = {
    "GDA": (0.01, 0.03, 0.06),
    "EGM": (0.01, 0.03, 0.06),
    "PPM-3": (0.01, 0.03, 0.06),
    "Minimax-PPO": (0.0003, 0.001, 0.003),
}


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing output: {}".format(output))
    output.mkdir(parents=True)
    spec = core.GameSpec("Validation-C8-H64", channels=8, hidden_width=64)
    game = core.build_game(spec)
    rows = []
    selections = {}
    for method, grid in GRIDS.items():
        candidates = []
        for learning_rate in grid:
            seed_metrics = []
            for seed in SEEDS:
                z = core.initialize(seed, game)
                for _ in range(STEPS):
                    z = core.classical_update(method, z, game, learning_rate)
                measured = core.metrics(z, game)
                seed_metrics.append(measured)
                rows.append(
                    {
                        "method": method,
                        "learning_rate": learning_rate,
                        "seed": seed,
                        "final_worst_case_rate": measured["robust_rate"],
                        "final_hard_exploitability": measured["hard_exploitability"],
                        "final_field_norm": measured["field_norm"],
                    }
                )
            candidates.append(
                {
                    "learning_rate": learning_rate,
                    "mean_worst_case_rate": float(np.mean([x["robust_rate"] for x in seed_metrics])),
                    "mean_hard_exploitability": float(np.mean([x["hard_exploitability"] for x in seed_metrics])),
                }
            )
        ranked = sorted(candidates, key=lambda x: (-x["mean_worst_case_rate"], x["mean_hard_exploitability"]))
        selections[method] = {"selected_learning_rate": ranked[0]["learning_rate"], "candidates": candidates}
    with (output / "per_seed_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "model_version": "exogenous-ph-v1",
        "configuration": {"channels": 8, "hidden_width": 64, "updates": STEPS},
        "validation_seeds": list(SEEDS),
        "test_seeds": list(range(400, 412)),
        "selection_rule": "maximize final mean worst-case spectral efficiency; break exact ties by lower final mean hard exploitability",
        "selections": selections,
    }
    with (output / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    test_rows = []
    for method in ("GDA", "EGM", "PPM-3"):
        learning_rate = selections[method]["selected_learning_rate"]
        for seed in range(400, 412):
            z = core.initialize(seed, game)
            for _ in range(STEPS):
                z = core.classical_update(method, z, game, learning_rate)
            measured = core.metrics(z, game)
            test_rows.append(
                {
                    "method": method,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "final_worst_case_rate": measured["robust_rate"],
                    "final_hard_exploitability": measured["hard_exploitability"],
                    "final_field_norm": measured["field_norm"],
                }
            )
    with (output / "locked_test_selected_rates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    report["locked_test_summary"] = {
        method: {
            "mean_worst_case_rate": float(np.mean([x["final_worst_case_rate"] for x in test_rows if x["method"] == method])),
            "mean_hard_exploitability": float(np.mean([x["final_hard_exploitability"] for x in test_rows if x["method"] == method])),
            "mean_field_norm": float(np.mean([x["final_field_norm"] for x in test_rows if x["method"] == method])),
        }
        for method in ("GDA", "EGM", "PPM-3")
    }
    with (output / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(selections, indent=2))
    print(json.dumps(report["locked_test_summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output)


if __name__ == "__main__":
    main()
