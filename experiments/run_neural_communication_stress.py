"""Communication-parameter stress sweep for the nonlinear neural Markov game.

The protocol is fixed before execution.  Each physical operating point is
trained independently with matched seeds and simultaneous transmitter--jammer
updates.  The sweep varies jammer strength, channel-switching cost,
adjacent-channel leakage, and exogenous radio-mode persistence while keeping
the neural architecture and optimizer settings unchanged.
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


CHANNELS = 6
HIDDEN_WIDTH = 32
SEEDS = tuple(range(500, 504))
STEPS = 25
METHODS = ("QP+G", "noG", "PPM-3")
CHECKPOINT_EVERY = 5


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def configurations() -> list[GameSpec]:
    specs: list[GameSpec] = []
    for jammer_to_noise in (12.0, 16.0, 20.0, 24.0):
        specs.append(
            GameSpec(
                f"Jammer-{jammer_to_noise:.0f}",
                channels=CHANNELS,
                hidden_width=HIDDEN_WIDTH,
                jammer_to_noise=jammer_to_noise,
            )
        )
    for switch_cost in (0.02, 0.06, 0.10, 0.14):
        specs.append(
            GameSpec(
                f"Switch-{switch_cost:.2f}",
                channels=CHANNELS,
                hidden_width=HIDDEN_WIDTH,
                switch_cost=switch_cost,
            )
        )
    for adjacent_leakage in (0.20, 0.40, 0.60, 0.80):
        specs.append(
            GameSpec(
                f"Leakage-{adjacent_leakage:.2f}",
                channels=CHANNELS,
                hidden_width=HIDDEN_WIDTH,
                adjacent_leakage=adjacent_leakage,
            )
        )
    for rho_h in (0.40, 0.55, 0.70, 0.85):
        specs.append(
            GameSpec(
                f"Persistence-{rho_h:.2f}",
                channels=CHANNELS,
                hidden_width=HIDDEN_WIDTH,
                rho_h=rho_h,
            )
        )
    return specs


def run(output: Path, seeds: Sequence[int]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    rows: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    specs = configurations()
    for spec in specs:
        new_rows, new_diagnostics = run_configuration(
            spec,
            seeds,
            STEPS,
            METHODS,
            CHECKPOINT_EVERY,
        )
        rows.extend(new_rows)
        diagnostics.extend(new_diagnostics)
        write_csv(output / "neural_stress_results.csv", rows)
        write_csv(output / "neural_stress_diagnostics.csv", diagnostics)
    summary = {
        "date": "2026-08-07",
        "device": "CPU",
        "training_mode": "independent retraining at every communication operating point",
        "simultaneous_joint_updates": True,
        "warmup": "none",
        "channels": CHANNELS,
        "states": CHANNELS**2,
        "hidden_width": HIDDEN_WIDTH,
        "parameters_per_player": parameter_count(CHANNELS, HIDDEN_WIDTH),
        "seeds": list(seeds),
        "seed_selection": "prespecified before execution; no outcome-based filtering",
        "steps": STEPS,
        "methods": list(METHODS),
        "fixed_step_size": 0.01,
        "configurations": [asdict(spec) for spec in specs],
        "maximum_br_residual": max(float(row["maximum_br_residual"]) for row in rows),
        "elapsed_seconds": time.time() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/icc_neural_stress"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.output, tuple(arguments.seeds))
