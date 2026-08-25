"""Pre-specified stochastic-oracle diagnostic for the ICC trajectory benchmark.

The script deterministically replays the locked LCMPL trajectory to recover
checkpoints 0, 15, and 30.  It does not retrain or select checkpoints by
outcome.  Independent evaluation batches compare empirical F/G with exact
population F/G at four fixed trajectory counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

import run_neural_scale_game as core
import run_trajectory_sampled_neural as sampled


SEEDS = tuple(range(400, 412))
CHECKPOINTS = (0, 15, 30)
BATCH_TRAJECTORIES = (32, 64, 128, 256)
REPLICATES = 4
HORIZON = 64
UPDATES = 30


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def replay_checkpoints(seed: int, game: core.Game) -> dict[int, torch.Tensor]:
    z = core.initialize(seed, game)
    scales = core.initial_scales(z, game)
    states = {0: z.clone()}
    for step in range(UPDATES):
        batch_seed = 120_000_000 + seed * 100_003 + step
        batch = sampled.sample_batch(z, game, sampled.TRANSITIONS_PER_UPDATE, batch_seed)
        z, _ = sampled.sampled_adaptive_update("QP+G", z, game, scales, batch)
        if step + 1 in CHECKPOINTS:
            states[step + 1] = z.clone()
    if set(states) != set(CHECKPOINTS):
        raise RuntimeError(f"missing checkpoint for seed {seed}: {sorted(states)}")
    return states


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    return float(torch.dot(left, right) / max(denominator, 1.0e-15))


def run(output: Path, smoke: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=False)
    game = core.build_game(core.GameSpec("Hard-C8-H64", channels=8, hidden_width=64))
    seeds = SEEDS[:1] if smoke else SEEDS
    checkpoints_to_evaluate = (0,) if smoke else CHECKPOINTS
    batch_trajectories = (32,) if smoke else BATCH_TRAJECTORIES
    replicates = 1 if smoke else REPLICATES
    rows: list[dict[str, object]] = []
    for seed in seeds:
        checkpoints = replay_checkpoints(seed, game)
        for checkpoint, z in checkpoints.items():
            if checkpoint not in checkpoints_to_evaluate:
                continue
            exact_f, exact_g, _, _ = core.field_and_curvature(z, game)
            f_norm = float(torch.linalg.vector_norm(exact_f))
            g_norm = float(torch.linalg.vector_norm(exact_g))
            for trajectories in batch_trajectories:
                for replicate in range(replicates):
                    batch_seed = (
                        910_000_000
                        + seed * 100_003
                        + checkpoint * 1_009
                        + trajectories * 17
                        + replicate
                    )
                    batch = sampled.sample_batch(
                        z,
                        game,
                        trajectories * HORIZON,
                        batch_seed,
                        horizon=HORIZON,
                    )
                    estimate_f, estimate_g, _, _ = sampled.sampled_oracles(z, batch, game)
                    rows.append(
                        {
                            "training_seed": seed,
                            "checkpoint": checkpoint,
                            "batch_trajectories": trajectories,
                            "batch_transitions": trajectories * HORIZON,
                            "replicate": replicate,
                            "batch_seed": batch_seed,
                            "exact_f_norm": f_norm,
                            "exact_g_norm": g_norm,
                            "relative_f_error": float(torch.linalg.vector_norm(estimate_f - exact_f))
                            / max(f_norm, 1.0e-12),
                            "relative_g_error": float(torch.linalg.vector_norm(estimate_g - exact_g))
                            / max(g_norm, 1.0e-12),
                            "f_cosine": cosine(estimate_f, exact_f),
                            "g_cosine": cosine(estimate_g, exact_g),
                        }
                    )
        print(f"oracle diagnostic seed={seed} complete", flush=True)
    write_csv(output / "oracle_quality_raw.csv", rows)
    protocol = {
        "status": "smoke_complete" if smoke else "complete",
        "training_seeds": list(seeds),
        "checkpoint_rule": "fixed joint updates 0, 15, and 30 on deterministic replay of locked QP+G runs",
        "checkpoints": list(checkpoints_to_evaluate),
        "batch_trajectories": list(batch_trajectories),
        "horizon": HORIZON,
        "replicates_per_seed_checkpoint_batch": replicates,
        "evaluation_batch_seed_formula": "910000000 + training_seed*100003 + checkpoint*1009 + trajectories*17 + replicate",
        "population_oracle": "exact finite-game autograd saddle field and exact JVP",
        "sampled_oracle": "same-batch likelihood-ratio field and autograd JVP",
        "outcome_based_selection": False,
        "failed_runs": 0,
        "raw_rows": len(rows),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.output, smoke=args.smoke)


if __name__ == "__main__":
    main()
