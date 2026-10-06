"""Locked six-method queue-aware adversarial DSA benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import scipy
import torch

from run_queue_aware_stress import QueueGameSpec, manifest as common_manifest, run_configuration, write_csv


MODEL_VERSION = "queue-aware-dsa-nominal-v2"
SEEDS = tuple(range(1600, 1612))
STEPS = 60
ARRIVAL_RATE = 0.45
METHODS = ("QP+G", "noG", "Minimax-PPO", "GDA", "EGM", "PPM-3")
CHECKPOINT_EVERY = 5


def run(output: Path, seeds: Sequence[int], steps: int) -> None:
    output.mkdir(parents=True, exist_ok=False)
    spec = QueueGameSpec("Queue-nominal", arrival_rate=ARRIVAL_RATE)
    manifest = {
        "model_version": MODEL_VERSION,
        "protocol_locked_before_confirmatory_execution": True,
        "seed_selection": "fixed queue-aware seeds; no outcome-based filtering or reordering",
        "seeds": list(seeds),
        "methods": list(METHODS),
        "joint_updates": steps,
        "checkpoint_every": CHECKPOINT_EVERY,
        "configuration": spec.__dict__,
        "arrival_process": "independent Bernoulli arrivals in every slot",
        "service_process": "Bernoulli service whose probability depends on both channel actions",
        "joint_action_dependent_transition": True,
        "algorithm_parameters": common_manifest(seeds, steps, (ARRIVAL_RATE,))["algorithm_parameters"],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    failures_path = output / "failures.jsonl"
    failures_path.write_text("", encoding="utf-8")
    rows: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    started = time.time()
    for seed in seeds:
        try:
            new_rows, new_diagnostics = run_configuration(
                spec,
                (seed,),
                steps,
                METHODS,
                CHECKPOINT_EVERY,
            )
            rows.extend(new_rows)
            diagnostics.extend(new_diagnostics)
        except Exception as error:
            failure = {
                "seed": int(seed),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            with failures_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, sort_keys=True) + "\n")
        write_csv(output / "queue_nominal_results.csv", rows)
        write_csv(output / "queue_nominal_diagnostics.csv", diagnostics)
    summary = {
        **manifest,
        "elapsed_seconds": time.time() - started,
        "completed_seed_count": len({int(row["seed"]) for row in rows}),
        "expected_seed_count": len(seeds),
        "failure_count": len(failures),
        "maximum_br_residual": max(
            (float(row["maximum_br_residual"]) for row in rows),
            default=None,
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/queue_nominal"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--steps", type=int, default=STEPS)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.output, tuple(arguments.seeds), arguments.steps)
