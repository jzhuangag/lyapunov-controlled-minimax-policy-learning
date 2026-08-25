"""Trajectory-sampled saddle-field experiment for the ICC neural benchmark.

The experiment keeps the Hard-C8-H64 game, policies, objective, Lyapunov
merit, safeguards, and fixed-rule learning rate used by the population study.
Only the saddle-field oracle is replaced by a finite-trajectory estimator.
The same on-policy batch supplies both players, the field, and the JVP.  PPM-3
reuses that batch through per-decision likelihood ratios, which equalizes the
environment-interaction budget across all three methods.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import t as student_t

import run_neural_scale_game as core


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "icc_trajectory_sampled"
METHODS = ("QP+G", "noG", "GDA", "EGM", "PPM-3")
FORMAL_SEEDS = tuple(range(400, 412))
UPDATES = 30
CHECKPOINT_EVERY = 5
ROLLOUT_HORIZON = 64
TRANSITIONS_PER_UPDATE = 8192
PPM_LEARNING_RATE = 0.01
FIXED_LEARNING_RATE = 0.01


@dataclass(frozen=True)
class TrajectoryBatch:
    states: torch.Tensor
    transmitter_actions: torch.Tensor
    jammer_actions: torch.Tensor
    rewards: torch.Tensor
    behavior_log_joint: torch.Tensor
    requested_transitions: int

    @property
    def trajectories(self) -> int:
        return int(self.states.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.states.shape[1])

    @property
    def transitions(self) -> int:
        return int(self.states.numel())


def vector_categorical(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    draws = rng.random(probabilities.shape[0])
    return np.sum(draws[:, None] > np.cumsum(probabilities, axis=1), axis=1).astype(np.int64)


def sample_batch(
    z: torch.Tensor,
    game: core.Game,
    requested_transitions: int,
    seed: int,
    horizon: int = ROLLOUT_HORIZON,
) -> TrajectoryBatch:
    rng = np.random.default_rng(seed)
    trajectories = max(1, math.ceil(requested_transitions / horizon))
    with torch.no_grad():
        transmitter, jammer = core.policies(z, game)
    transmitter_np = transmitter.cpu().numpy()
    jammer_np = jammer.cpu().numpy()
    transitions = game.transitions.cpu().numpy()
    rewards = game.rewards.cpu().numpy()
    rho = game.rho.cpu().numpy()
    states = np.empty((trajectories, horizon), dtype=np.int64)
    actions_a = np.empty_like(states)
    actions_b = np.empty_like(states)
    sampled_rewards = np.empty((trajectories, horizon), dtype=np.float64)
    behavior_log_joint = np.empty_like(sampled_rewards)
    current = rng.choice(len(rho), size=trajectories, p=rho / rho.sum())
    rows = np.arange(trajectories)
    for step in range(horizon):
        action_a = vector_categorical(transmitter_np[current], rng)
        action_b = vector_categorical(jammer_np[current], rng)
        states[:, step] = current
        actions_a[:, step] = action_a
        actions_b[:, step] = action_b
        sampled_rewards[:, step] = rewards[current, action_a, action_b]
        behavior_log_joint[:, step] = (
            np.log(np.maximum(transmitter_np[current, action_a], 1.0e-300))
            + np.log(np.maximum(jammer_np[current, action_b], 1.0e-300))
        )
        next_probabilities = transitions[current, action_a, action_b]
        current = vector_categorical(next_probabilities, rng)
    return TrajectoryBatch(
        states=torch.tensor(states, dtype=torch.long),
        transmitter_actions=torch.tensor(actions_a, dtype=torch.long),
        jammer_actions=torch.tensor(actions_b, dtype=torch.long),
        rewards=torch.tensor(sampled_rewards),
        behavior_log_joint=torch.tensor(behavior_log_joint),
        requested_transitions=requested_transitions,
    )


def sampled_objective(
    z: torch.Tensor,
    batch: TrajectoryBatch,
    game: core.Game,
) -> Tuple[torch.Tensor, torch.Tensor]:
    transmitter, jammer = core.policies(z, game)
    state = batch.states
    action_a = batch.transmitter_actions
    action_b = batch.jammer_actions
    log_joint = (
        torch.log(transmitter[state, action_a].clamp_min(1.0e-15))
        + torch.log(jammer[state, action_b].clamp_min(1.0e-15))
    )
    prefix_ratio = torch.exp(torch.cumsum(log_joint - batch.behavior_log_joint, dim=1))
    entropy_a = -(transmitter * torch.log(transmitter.clamp_min(1.0e-15))).sum(dim=1)[state]
    entropy_b = -(jammer * torch.log(jammer.clamp_min(1.0e-15))).sum(dim=1)[state]
    reward = batch.rewards + core.ENTROPY_TAU * (entropy_a - entropy_b)
    discount = torch.pow(
        torch.tensor(core.DISCOUNT, dtype=z.dtype),
        torch.arange(batch.horizon, dtype=z.dtype),
    ).reshape(1, -1)
    value = (1.0 - core.DISCOUNT) * torch.mean(
        torch.sum(discount * prefix_ratio * reward, dim=1)
    )
    return value, prefix_ratio


def sign_vector(game: core.Game) -> torch.Tensor:
    return torch.cat((-torch.ones(game.block_size), torch.ones(game.block_size)))


def empirical_field(z_value: torch.Tensor, batch: TrajectoryBatch, game: core.Game) -> torch.Tensor:
    z = z_value.detach().clone().requires_grad_(True)
    value, _ = sampled_objective(z, batch, game)
    gradient = torch.autograd.grad(value, z)[0]
    return (sign_vector(game) * gradient).detach()


def sampled_oracles(
    z_value: torch.Tensor,
    batch: TrajectoryBatch,
    game: core.Game,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    z = z_value.detach().clone().requires_grad_(True)
    value, _ = sampled_objective(z, batch, game)
    gradient = torch.autograd.grad(value, z, create_graph=True)[0]
    signs = sign_vector(game)
    field = signs * gradient
    hessian_field = torch.autograd.grad(gradient, z, grad_outputs=field.detach())[0]
    curvature = signs * hessian_field
    field = field.detach()
    curvature = curvature.detach()
    f_norm = float(torch.linalg.vector_norm(field))
    g_norm = float(torch.linalg.vector_norm(curvature))
    ratio = g_norm / max(f_norm, 1.0e-15)
    cosine = float(torch.dot(field, curvature) / max(f_norm * g_norm, 1.0e-15))
    return field, curvature, ratio, cosine


def sampled_adaptive_update(
    method: str,
    z: torch.Tensor,
    game: core.Game,
    scales: Tuple[float, float, float],
    batch: TrajectoryBatch,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    use_curvature = method == "QP+G"
    if use_curvature:
        field, curvature, ratio, cosine = sampled_oracles(z, batch, game)
    else:
        field = empirical_field(z, batch, game)
        curvature = None
        ratio = 0.0
        cosine = 0.0
    value_before = core.merit(z, game, scales)
    beta, gamma, beta_no_g, model_diagnostics = core.local_coefficients(z, field, curvature, game, scales)
    candidate, beta_used, gamma_used, backtracks = core.safeguard(
        z, field, curvature, beta, gamma, game, scales
    )
    rejected_curvature = 0.0
    if use_curvature:
        no_g_candidate, _, _, _ = core.safeguard(
            z, field, None, beta_no_g, 0.0, game, scales
        )
        if (
            core.metrics(candidate, game)["robust_rate"] + 1.0e-10
            < core.metrics(no_g_candidate, game)["robust_rate"]
            or core.merit(candidate, game, scales)
            > core.merit(no_g_candidate, game, scales) + 1.0e-10
        ):
            candidate = no_g_candidate
            beta_used = beta_no_g
            gamma_used = 0.0
            rejected_curvature = float(gamma > 0.0)
    realized_change = core.merit(candidate, game, scales) - value_before
    return candidate, {
        "beta": beta_used,
        "gamma": gamma_used,
        "g_over_f": ratio,
        "cosine_fg": cosine,
        "sampled_field_norm": float(torch.linalg.vector_norm(field)),
        "sampled_curvature_norm": 0.0
        if curvature is None
        else float(torch.linalg.vector_norm(curvature)),
        "curvature_candidate_accepted": float(use_curvature and gamma_used > 0.0),
        "curvature_candidate_rejected": rejected_curvature,
        "backtracking_steps": float(backtracks),
        "predicted_model_change": float(model_diagnostics["predicted_model_change"]),
        "realized_merit_change": float(realized_change),
        "probe_model_error": float(realized_change - model_diagnostics["predicted_model_change"]),
        "merit_probe_count": float(model_diagnostics["probe_count"]),
        "field_oracle_count": 1.0,
        "jvp_count": float(use_curvature),
    }


def sampled_ppm_update(
    z: torch.Tensor,
    game: core.Game,
    batch: TrajectoryBatch,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    initial_field = empirical_field(z, batch, game)
    iterate = z.detach()
    for _ in range(3):
        implicit_field = empirical_field(iterate, batch, game)
        iterate = (z - PPM_LEARNING_RATE * implicit_field).detach()
    return iterate, {
        "beta": PPM_LEARNING_RATE,
        "gamma": 0.0,
        "g_over_f": 0.0,
        "cosine_fg": 0.0,
        "sampled_field_norm": float(torch.linalg.vector_norm(initial_field)),
        "sampled_curvature_norm": 0.0,
        "curvature_candidate_accepted": 0.0,
        "curvature_candidate_rejected": 0.0,
        "backtracking_steps": 0.0,
        "predicted_model_change": float("nan"),
        "realized_merit_change": float("nan"),
        "probe_model_error": float("nan"),
        "merit_probe_count": 0.0,
        "field_oracle_count": 3.0,
        "jvp_count": 0.0,
    }


def sampled_fixed_update(
    method: str,
    z: torch.Tensor,
    game: core.Game,
    batch: TrajectoryBatch,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    field = empirical_field(z, batch, game)
    if method == "GDA":
        candidate = (z - FIXED_LEARNING_RATE * field).detach()
        oracle_count = 1.0
    elif method == "EGM":
        predictor = (z - FIXED_LEARNING_RATE * field).detach()
        predictor_field = empirical_field(predictor, batch, game)
        candidate = (z - FIXED_LEARNING_RATE * predictor_field).detach()
        oracle_count = 2.0
    else:
        raise ValueError(method)
    return candidate, {
        "beta": FIXED_LEARNING_RATE,
        "gamma": 0.0,
        "g_over_f": 0.0,
        "cosine_fg": 0.0,
        "sampled_field_norm": float(torch.linalg.vector_norm(field)),
        "sampled_curvature_norm": 0.0,
        "curvature_candidate_accepted": 0.0,
        "curvature_candidate_rejected": 0.0,
        "backtracking_steps": 0.0,
        "predicted_model_change": float("nan"),
        "realized_merit_change": float("nan"),
        "probe_model_error": float("nan"),
        "merit_probe_count": 0.0,
        "field_oracle_count": oracle_count,
        "jvp_count": 0.0,
    }


def paired_interval(values: np.ndarray) -> List[float]:
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return [mean - half, mean + half]


def summarize(
    rows: Sequence[Dict[str, object]],
    diagnostics: Sequence[Dict[str, object]],
    final_step: int,
) -> Dict[str, object]:
    final = [row for row in rows if int(row["step"]) == final_step]
    by_method: Dict[str, Dict[int, Dict[str, object]]] = {method: {} for method in METHODS}
    for row in final:
        by_method[str(row["method"])][int(row["seed"])] = row
    qpg = by_method["QP+G"]
    nog = by_method["noG"]
    seeds = sorted(set(qpg) & set(nog))
    rate_gain = np.asarray([float(qpg[s]["robust_rate"]) - float(nog[s]["robust_rate"]) for s in seeds])
    exploitability_reduction = np.asarray([
        float(nog[s]["hard_exploitability"]) - float(qpg[s]["hard_exploitability"])
        for s in seeds
    ])
    method_summary = {}
    for method in METHODS:
        selected = list(by_method[method].values())
        method_summary[method] = {
            "final_rate_mean": float(np.mean([float(row["robust_rate"]) for row in selected])),
            "final_exploitability_mean": float(np.mean([float(row["hard_exploitability"]) for row in selected])),
            "final_population_field_norm_mean": float(np.mean([float(row["field_norm"]) for row in selected])),
            "total_transitions_per_seed": int(max(int(row["cumulative_transitions"]) for row in selected)),
        }
    qpg_diagnostics = [row for row in diagnostics if row["method"] == "QP+G"]
    return {
        "methods": method_summary,
        "paired_lcmpl_minus_lcmpl_f_rate_mean": float(rate_gain.mean()),
        "paired_rate_gain_95ci": paired_interval(rate_gain),
        "paired_lcmpl_f_minus_lcmpl_exploitability_mean": float(exploitability_reduction.mean()),
        "paired_exploitability_reduction_95ci": paired_interval(exploitability_reduction),
        "rate_seed_wins": int(np.sum(rate_gain > 0.0)),
        "exploitability_seed_wins": int(np.sum(exploitability_reduction > 0.0)),
        "curvature_activation_fraction": float(np.mean([float(row["gamma"]) > 1.0e-10 for row in qpg_diagnostics])),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_outputs(
    output: Path,
    summary: Dict[str, object],
    rows: Sequence[Dict[str, object]],
    diagnostics: Sequence[Dict[str, object]],
) -> None:
    method_rows = []
    for method, record in summary["methods"].items():
        method_rows.append({"method": method, **record})
    write_csv(output / "trajectory_summary.csv", method_rows)
    expected_seeds = set(FORMAL_SEEDS)
    observed_seeds = {int(row["seed"]) for row in rows}
    final_rows = [row for row in rows if int(row["step"]) == UPDATES]
    budget_by_method_seed = {
        (str(row["method"]), int(row["seed"])): int(row["cumulative_transitions"])
        for row in final_rows
    }
    integrity = {
        "expected_seed_count": len(expected_seeds),
        "observed_seed_count": len(observed_seeds),
        "all_locked_seeds_present": observed_seeds == expected_seeds,
        "expected_methods": list(METHODS),
        "final_method_seed_cells": len(budget_by_method_seed),
        "expected_final_method_seed_cells": len(METHODS) * len(expected_seeds),
        "all_cells_present": len(budget_by_method_seed) == len(METHODS) * len(expected_seeds),
        "unique_final_budgets": sorted(set(budget_by_method_seed.values())),
        "equal_locked_transition_budget": set(budget_by_method_seed.values()) == {UPDATES * TRANSITIONS_PER_UPDATE},
        "diagnostic_rows": len(diagnostics),
        "expected_diagnostic_rows": len(METHODS) * len(expected_seeds) * UPDATES,
        "all_updates_present": len(diagnostics) == len(METHODS) * len(expected_seeds) * UPDATES,
        "finite_primary_metrics": all(
            math.isfinite(float(row[metric]))
            for row in rows
            for metric in ("robust_rate", "hard_exploitability", "field_norm")
        ),
    }
    integrity["status"] = "pass" if all(
        integrity[key]
        for key in (
            "all_locked_seeds_present",
            "all_cells_present",
            "equal_locked_transition_budget",
            "all_updates_present",
            "finite_primary_metrics",
        )
    ) else "fail"
    (output / "formal_integrity.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )


def postprocess_existing(output: Path) -> None:
    with (output / "trajectory_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (output / "trajectory_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        diagnostics = list(csv.DictReader(handle))
    summary = summarize(rows, diagnostics, UPDATES)
    (output / "trajectory_statistics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_aggregate_outputs(output, summary, rows, diagnostics)
    print((output / "formal_integrity.json").read_text(encoding="utf-8"))


def run(output: Path, seeds: Sequence[int], updates: int, transitions_per_update: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    game = core.build_game(core.GameSpec("Hard-C8-H64", channels=8, hidden_width=64))
    rows: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    started = time.time()
    for seed in seeds:
        initial = core.initialize(seed, game)
        scales = core.initial_scales(initial, game)
        for method in METHODS:
            z = initial.clone()
            cumulative_transitions = 0
            cumulative_oracle_calls = 0.0
            cumulative_jvp_calls = 0.0
            method_started = time.perf_counter()
            last = {
                "beta": 0.0,
                "gamma": 0.0,
                "g_over_f": 0.0,
                "cosine_fg": 0.0,
                "sampled_field_norm": float("nan"),
                "sampled_curvature_norm": float("nan"),
                "curvature_candidate_accepted": 0.0,
                "curvature_candidate_rejected": 0.0,
                "backtracking_steps": 0.0,
                "predicted_model_change": float("nan"),
                "realized_merit_change": float("nan"),
                "probe_model_error": float("nan"),
                "merit_probe_count": 0.0,
                "field_oracle_count": 0.0,
                "jvp_count": 0.0,
            }
            for step in range(updates + 1):
                if step % CHECKPOINT_EVERY == 0 or step == updates:
                    rows.append({
                        "config": "Hard-C8-H64-Trajectory",
                        "seed": seed,
                        "method": method,
                        "step": step,
                        "cumulative_transitions": cumulative_transitions,
                        "cumulative_oracle_calls": cumulative_oracle_calls,
                        "cumulative_jvp_calls": cumulative_jvp_calls,
                        "wall_clock_seconds": time.perf_counter() - method_started,
                        **core.metrics(z, game),
                        **last,
                    })
                    write_csv(output / "trajectory_results.partial.csv", rows)
                if step == updates:
                    break
                batch_seed = 120_000_000 + seed * 100_003 + step
                batch = sample_batch(z, game, transitions_per_update, batch_seed)
                if method == "PPM-3":
                    z, last = sampled_ppm_update(z, game, batch)
                elif method in ("GDA", "EGM"):
                    z, last = sampled_fixed_update(method, z, game, batch)
                else:
                    z, last = sampled_adaptive_update(method, z, game, scales, batch)
                cumulative_transitions += batch.transitions
                cumulative_oracle_calls += float(last.get("field_oracle_count", 0.0))
                cumulative_jvp_calls += float(last.get("jvp_count", 0.0))
                diagnostics.append({
                    "config": "Hard-C8-H64-Trajectory",
                    "seed": seed,
                    "method": method,
                    "step": step + 1,
                    "batch_seed": batch_seed,
                    "trajectories": batch.trajectories,
                    "transitions_used": batch.transitions,
                    "cumulative_transitions": cumulative_transitions,
                    "cumulative_oracle_calls": cumulative_oracle_calls,
                    "cumulative_jvp_calls": cumulative_jvp_calls,
                    "wall_clock_seconds": time.perf_counter() - method_started,
                    **last,
                })
                write_csv(output / "trajectory_diagnostics.partial.csv", diagnostics)
        print(f"seed={seed} complete elapsed={time.time() - started:.1f}s", flush=True)
    write_csv(output / "trajectory_results.csv", rows)
    write_csv(output / "trajectory_diagnostics.csv", diagnostics)
    summary = summarize(rows, diagnostics, updates)
    protocol = {
        "configuration": "Hard-C8-H64",
        "methods": list(METHODS),
        "seeds": list(seeds),
        "joint_updates": updates,
        "rollout_horizon": ROLLOUT_HORIZON,
        "trajectories_per_update": math.ceil(transitions_per_update / ROLLOUT_HORIZON),
        "transitions_per_update": math.ceil(transitions_per_update / ROLLOUT_HORIZON) * ROLLOUT_HORIZON,
        "total_transitions_per_method_seed": updates * math.ceil(transitions_per_update / ROLLOUT_HORIZON) * ROLLOUT_HORIZON,
        "same_batch_for_players_field_and_jvp": True,
        "same_batch_reused_for_ppm_inner_steps": True,
        "critic_or_baseline": "none",
        "field_oracle": "per-decision likelihood-ratio finite-horizon policy gradient",
        "curvature_oracle": "same-batch autograd JVP of the empirical saddle field",
        "controller_merit": "exact finite-game composite Lyapunov merit, unchanged from the population study",
        "training_safeguard": "exact finite-game merit and hard-BR rate, unchanged from the population study",
        "checkpoint_evaluation": "exact finite-game hard best responses and population field norm",
        "fully_model_free": False,
        "sample_budget_alignment": "equal environment transitions across methods",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (output / "trajectory_statistics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_aggregate_outputs(output, summary, rows, diagnostics)
    print(json.dumps({"protocol": protocol, "summary": summary, "elapsed_seconds": time.time() - started}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal", "postprocess"), default="smoke")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.phase == "smoke":
        run(args.output / "smoke", (9391,), 1, 512)
    elif args.phase == "formal":
        run(args.output, FORMAL_SEEDS, UPDATES, TRANSITIONS_PER_UPDATE)
    else:
        postprocess_existing(args.output)


if __name__ == "__main__":
    main()
