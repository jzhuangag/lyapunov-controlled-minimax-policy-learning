"""Confirmatory large neural-policy spectrum-access experiments.

The script extends the scale study to 64-unit two-layer policies and up to ten
channels.  Seed ranges, update budgets, and method sets are fixed below before
the confirmatory run.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

from run_neural_scale_game import GameSpec, parameter_count, run_configuration


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path, smoke: bool, large_only: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if large_only:
        configurations = [
            (
                GameSpec("Large-C12-H96", channels=12, hidden_width=96),
                tuple(range(440, 446)),
                25,
                ("QP+G", "noG", "PPM-3"),
                5,
            )
        ]
    elif smoke:
        configurations = [
            (
                GameSpec("Smoke-C10-H64", channels=10, hidden_width=64),
                (420, 421),
                5,
                ("QP+G", "noG", "Minimax-PPO"),
                1,
            )
        ]
    else:
        configurations = [
            (
                GameSpec("Hard-C8-H64", channels=8, hidden_width=64),
                tuple(range(400, 412)),
                30,
                ("QP+G", "noG", "Minimax-PPO", "GDA", "EGM", "PPM-3"),
                5,
            ),
            (
                GameSpec("Stress-C10-H64", channels=10, hidden_width=64),
                tuple(range(420, 428)),
                25,
                ("QP+G", "noG", "PPM-3"),
                5,
            ),
            (
                GameSpec("Channels-C12-H64", channels=12, hidden_width=64),
                tuple(range(420, 428)),
                25,
                ("QP+G", "noG", "PPM-3"),
                5,
            ),
            *[
                (
                    GameSpec(f"Width-C8-H{hidden}", channels=8, hidden_width=hidden),
                    tuple(range(400, 412)),
                    25,
                    ("QP+G", "noG", "PPM-3"),
                    5,
                )
                for hidden in (8, 16, 32)
            ],
            *[
                (
                    GameSpec(f"Channels-C{channels}-H64", channels=channels, hidden_width=64),
                    tuple(range(420, 428)),
                    25,
                    ("QP+G", "noG", "PPM-3"),
                    5,
                )
                for channels in (6, 8)
            ],
        ]
    rows: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    for arguments in configurations:
        new_rows, new_diagnostics = run_configuration(*arguments)
        rows.extend(new_rows)
        diagnostics.extend(new_diagnostics)
        write_csv(output / "hard_neural_results.csv", rows)
        write_csv(output / "hard_neural_diagnostics.csv", diagnostics)
    summary = {
        "device": "CPU",
        "smoke": smoke,
        "large_only": large_only,
        "elapsed_seconds": time.time() - started,
        "configurations": [
            {
                "spec": asdict(arguments[0]),
                "seeds": list(arguments[1]),
                "steps": arguments[2],
                "methods": list(arguments[3]),
                "checkpoint_every": arguments[4],
                "parameters_per_player": parameter_count(arguments[0].channels, arguments[0].hidden_width),
            }
            for arguments in configurations
        ],
        "maximum_br_residual": max(float(row["maximum_br_residual"]) for row in rows),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/icc_neural_hard"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--large-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.output, arguments.smoke, arguments.large_only)
