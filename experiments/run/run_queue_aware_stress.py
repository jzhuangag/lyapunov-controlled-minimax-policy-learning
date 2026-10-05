"""Locked queue-aware adversarial DSA experiment for the ICC paper.

The finite-buffer Markov game makes the joint channel actions affect future
states through packet service.  The transmitter and jammer share the same
neural architecture, initialization, update count, and radio/traffic process
for every compared learning rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy
import torch

from run_neural_scale_game import (
    DISCOUNT,
    FIXED_LEARNING_RATE,
    Game,
    adaptive_update,
    classical_update,
    initial_scales,
    initialize,
    metrics,
    policies_numpy,
)


MODEL_VERSION = "queue-aware-dsa-v1"
CHANNELS = 4
MODES = 4
QUEUE_CAPACITY = 4
HIDDEN_WIDTH = 32
ARRIVAL_RATES = (0.20, 0.45, 0.70)
SEEDS = tuple(range(600, 612))
STEPS = 25
METHODS = ("QP+G", "noG", "PPM-3")
CHECKPOINT_EVERY = 5
PACKET_RATE = 1.0
JAMMER_TO_NOISE = 18.0
MODE_PERSISTENCE = 0.75
SWITCH_COST = 0.05
ADJACENT_LEAKAGE = 0.50
QUEUE_PENALTY = 0.12
DROP_PENALTY = 0.60


@dataclass(frozen=True)
class QueueGameSpec:
    name: str
    arrival_rate: float
    channels: int = CHANNELS
    modes: int = MODES
    queue_capacity: int = QUEUE_CAPACITY
    hidden_width: int = HIDDEN_WIDTH
    packet_rate: float = PACKET_RATE
    jammer_to_noise: float = JAMMER_TO_NOISE
    rho_h: float = MODE_PERSISTENCE
    switch_cost: float = SWITCH_COST
    adjacent_leakage: float = ADJACENT_LEAKAGE
    queue_penalty: float = QUEUE_PENALTY
    drop_penalty: float = DROP_PENALTY


def state_index(mode: int, queue: int, previous_channel: int, spec: QueueGameSpec) -> int:
    return (mode * (spec.queue_capacity + 1) + queue) * spec.channels + previous_channel


def queue_parameter_count(spec: QueueGameSpec) -> int:
    input_dim = 2 * spec.channels + 1
    return (
        spec.hidden_width * input_dim
        + spec.hidden_width
        + spec.channels * spec.hidden_width
        + spec.channels
    )


def build_queue_game(spec: QueueGameSpec) -> Tuple[Game, Dict[str, torch.Tensor]]:
    channels = spec.channels
    modes = spec.modes
    queue_levels = spec.queue_capacity + 1
    input_dim = 2 * channels + 1
    state_count = modes * queue_levels * channels
    base_gains = np.geomspace(1.9, 0.30, channels)
    gain_table = np.stack([np.roll(base_gains, shift) for shift in range(modes)])
    mode_transition = np.full((modes, modes), (1.0 - spec.rho_h) / (modes - 1))
    np.fill_diagonal(mode_transition, spec.rho_h)
    features = np.zeros((state_count, input_dim), dtype=float)
    rewards = np.zeros((state_count, channels, channels), dtype=float)
    transitions = np.zeros((state_count, channels, channels, state_count), dtype=float)
    goodput = np.zeros((state_count, channels, channels), dtype=float)
    backlog = np.zeros((state_count, channels, channels), dtype=float)
    drops = np.zeros((state_count, channels, channels), dtype=float)
    success_probability = np.zeros((state_count, channels, channels), dtype=float)
    threshold = 2.0**spec.packet_rate - 1.0
    for mode in range(modes):
        gains = gain_table[mode]
        normalized_gains = 2.0 * (gains - base_gains.min()) / (base_gains.max() - base_gains.min()) - 1.0
        for queue in range(queue_levels):
            normalized_queue = 2.0 * queue / spec.queue_capacity - 1.0
            for previous_channel in range(channels):
                state = state_index(mode, queue, previous_channel, spec)
                features[state, :channels] = normalized_gains
                features[state, channels] = normalized_queue
                features[state, channels + 1 + previous_channel] = 1.0
                for transmit_channel in range(channels):
                    switching_penalty = spec.switch_cost * float(transmit_channel != previous_channel)
                    for jammed_channel in range(channels):
                        cyclic_distance = min(
                            (transmit_channel - jammed_channel) % channels,
                            (jammed_channel - transmit_channel) % channels,
                        )
                        leakage = (
                            1.0
                            if cyclic_distance == 0
                            else spec.adjacent_leakage
                            if cyclic_distance == 1
                            else 0.08
                        )
                        mean_sinr = 10.0 * gains[transmit_channel] / (
                            1.0 + spec.jammer_to_noise * leakage
                        )
                        packet_success = 0.0 if queue == 0 else math.exp(-threshold / mean_sinr)
                        expected_goodput = spec.packet_rate * packet_success
                        expected_drop = (
                            spec.arrival_rate * (1.0 - packet_success)
                            if queue == spec.queue_capacity
                            else 0.0
                        )
                        rewards[state, transmit_channel, jammed_channel] = (
                            expected_goodput
                            - spec.queue_penalty * queue
                            - spec.drop_penalty * expected_drop
                            - switching_penalty
                        )
                        goodput[state, transmit_channel, jammed_channel] = expected_goodput
                        backlog[state, transmit_channel, jammed_channel] = queue
                        drops[state, transmit_channel, jammed_channel] = expected_drop
                        success_probability[state, transmit_channel, jammed_channel] = packet_success
                        service_outcomes = ((0, 1.0),) if queue == 0 else (
                            (0, 1.0 - packet_success),
                            (1, packet_success),
                        )
                        for served, service_probability in service_outcomes:
                            if service_probability <= 0.0:
                                continue
                            remaining = queue - served
                            for arrival, arrival_probability in (
                                (0, 1.0 - spec.arrival_rate),
                                (1, spec.arrival_rate),
                            ):
                                new_queue = min(spec.queue_capacity, remaining + arrival)
                                traffic_probability = service_probability * arrival_probability
                                for new_mode in range(modes):
                                    new_state = state_index(new_mode, new_queue, transmit_channel, spec)
                                    transitions[state, transmit_channel, jammed_channel, new_state] += (
                                        mode_transition[mode, new_mode] * traffic_probability
                                    )
    rho = np.zeros(state_count, dtype=float)
    for mode in range(modes):
        for previous_channel in range(channels):
            rho[state_index(mode, 0, previous_channel, spec)] = 1.0 / (modes * channels)
    game = Game(
        spec=spec,
        features=torch.tensor(features),
        rewards=torch.tensor(rewards),
        transitions=torch.tensor(transitions),
        rho=torch.tensor(rho),
        input_dim=input_dim,
        block_size=queue_parameter_count(spec),
    )
    physical = {
        "goodput": torch.tensor(goodput),
        "backlog": torch.tensor(backlog),
        "drop_probability": torch.tensor(drops),
        "success_probability": torch.tensor(success_probability),
    }
    validate_queue_game(game, physical)
    return game, physical


def validate_queue_game(game: Game, physical: Dict[str, torch.Tensor]) -> None:
    transitions = game.transitions.detach().cpu().numpy()
    spec = game.spec
    if not np.allclose(transitions.sum(axis=-1), 1.0, atol=1.0e-12):
        raise RuntimeError("Queue-aware transition rows do not sum to one.")
    action_sensitive = False
    jammer_sensitive = False
    for state in range(transitions.shape[0]):
        rows = transitions[state]
        if not np.allclose(rows[0, 0], rows[-1, 0], atol=1.0e-12):
            action_sensitive = True
        if not np.allclose(rows[0, 0], rows[0, -1], atol=1.0e-12):
            jammer_sensitive = True
        for transmit_channel in range(spec.channels):
            reshaped = rows[transmit_channel].reshape(
                spec.channels,
                spec.modes,
                spec.queue_capacity + 1,
                spec.channels,
            )
            invalid_previous = np.delete(reshaped, transmit_channel, axis=3)
            if not np.allclose(invalid_previous, 0.0, atol=1.0e-12):
                raise RuntimeError("Next-state previous-channel index is inconsistent.")
    if not action_sensitive or not jammer_sensitive:
        raise RuntimeError("The queue transition must depend on both players' actions.")
    for name, tensor in physical.items():
        if tuple(tensor.shape) != tuple(game.rewards.shape):
            raise RuntimeError(f"Metric tensor {name} has an invalid shape.")


def minimizing_best_response(transmitter: np.ndarray, game: Game) -> np.ndarray:
    rewards = game.rewards.detach().cpu().numpy()
    transitions = game.transitions.detach().cpu().numpy()
    action_reward = np.einsum("sa,sab->sb", transmitter, rewards)
    action_transition = np.einsum("sa,sabn->sbn", transmitter, transitions)
    value = np.zeros(len(game.rho), dtype=float)
    for _ in range(2000):
        q_values = action_reward + DISCOUNT * (action_transition @ value)
        updated = q_values.min(axis=1)
        if np.max(np.abs(updated - value)) <= 1.0e-12:
            value = updated
            break
        value = updated
    actions = (action_reward + DISCOUNT * (action_transition @ value)).argmin(axis=1)
    response = np.zeros((len(game.rho), game.spec.channels), dtype=float)
    response[np.arange(len(game.rho)), actions] = 1.0
    return response


def discounted_metric(
    transmitter: np.ndarray,
    jammer: np.ndarray,
    game: Game,
    stage_metric: torch.Tensor,
) -> float:
    transition = np.einsum(
        "sa,sabn,sb->sn",
        transmitter,
        game.transitions.detach().cpu().numpy(),
        jammer,
    )
    stage = np.einsum(
        "sa,sab,sb->s",
        transmitter,
        stage_metric.detach().cpu().numpy(),
        jammer,
    )
    value = np.linalg.solve(np.eye(len(game.rho)) - DISCOUNT * transition, stage)
    return float((1.0 - DISCOUNT) * (game.rho.detach().cpu().numpy() @ value))


def queue_metrics(
    z: torch.Tensor,
    game: Game,
    physical: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    transmitter, _ = policies_numpy(z, game)
    jammer_br = minimizing_best_response(transmitter, game)
    return {
        "br_goodput": discounted_metric(transmitter, jammer_br, game, physical["goodput"]),
        "br_backlog": discounted_metric(transmitter, jammer_br, game, physical["backlog"]),
        "br_drop_probability": discounted_metric(
            transmitter,
            jammer_br,
            game,
            physical["drop_probability"],
        ),
    }


def run_configuration(
    spec: QueueGameSpec,
    seeds: Iterable[int],
    steps: int,
    methods: Sequence[str],
    checkpoint_every: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    game, physical = build_queue_game(spec)
    result_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    for seed in seeds:
        initial = initialize(seed, game)
        scales = initial_scales(initial, game)
        for method in methods:
            z = initial.clone()
            cumulative_oracle_calls = 0.0
            cumulative_jvp_calls = 0.0
            started = time.perf_counter()
            last_diagnostics: Dict[str, float] = {
                "beta": 0.0,
                "gamma": 0.0,
                "g_over_f": 0.0,
                "cosine_fg": 0.0,
            }
            for step in range(steps + 1):
                if step % checkpoint_every == 0 or step == steps:
                    result_rows.append(
                        {
                            "config": spec.name,
                            "arrival_rate": spec.arrival_rate,
                            "channels": spec.channels,
                            "queue_capacity": spec.queue_capacity,
                            "hidden_width": spec.hidden_width,
                            "parameters_per_player": game.block_size,
                            "seed": seed,
                            "method": method,
                            "step": step,
                            "cumulative_oracle_calls": cumulative_oracle_calls,
                            "cumulative_jvp_calls": cumulative_jvp_calls,
                            "wall_clock_seconds": time.perf_counter() - started,
                            **metrics(z, game),
                            **queue_metrics(z, game, physical),
                        }
                    )
                if step == steps:
                    break
                if method == "QP+G":
                    z, last_diagnostics = adaptive_update(z, game, scales, use_curvature=True)
                elif method == "noG":
                    z, last_diagnostics = adaptive_update(z, game, scales, use_curvature=False)
                else:
                    z = classical_update(method, z, game, learning_rate=FIXED_LEARNING_RATE)
                    last_diagnostics = {
                        "beta": FIXED_LEARNING_RATE,
                        "gamma": 0.0,
                        "field_oracle_count": 3.0 if method == "PPM-3" else 1.0,
                        "jvp_count": 0.0,
                    }
                cumulative_oracle_calls += float(last_diagnostics.get("field_oracle_count", 0.0))
                cumulative_jvp_calls += float(last_diagnostics.get("jvp_count", 0.0))
                diagnostic_rows.append(
                    {
                        "config": spec.name,
                        "arrival_rate": spec.arrival_rate,
                        "seed": seed,
                        "method": method,
                        "step": step + 1,
                        "cumulative_oracle_calls": cumulative_oracle_calls,
                        "cumulative_jvp_calls": cumulative_jvp_calls,
                        **last_diagnostics,
                    }
                )
        print(
            f"CONFIG={spec.name} seed={seed} states={len(game.rho)} "
            f"parameters={game.block_size}",
            flush=True,
        )
    return result_rows, diagnostic_rows


def configurations(arrival_rates: Sequence[float]) -> Tuple[QueueGameSpec, ...]:
    return tuple(
        QueueGameSpec(name=f"Queue-load-{arrival_rate:.2f}", arrival_rate=arrival_rate)
        for arrival_rate in arrival_rates
    )


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def manifest(seeds: Sequence[int], steps: int, arrival_rates: Sequence[float]) -> Dict[str, object]:
    specs = configurations(arrival_rates)
    return {
        "model_version": MODEL_VERSION,
        "protocol_locked_before_confirmatory_execution": True,
        "seed_selection": "fixed queue-aware seeds; no outcome-based filtering or reordering",
        "seeds": list(seeds),
        "methods": list(METHODS),
        "joint_updates": steps,
        "checkpoint_every": CHECKPOINT_EVERY,
        "state": ["radio_mode", "queue_length", "previous_transmitter_channel"],
        "joint_action_dependent_transition": True,
        "arrival_process": "independent Bernoulli arrivals",
        "service_process": "one packet with Rayleigh-outage success probability exp(-(2^R-1)/mean_SINR)",
        "initial_state": "empty queue; uniform radio mode and previous channel",
        "primary_metrics": [
            "worst-case discounted queue-aware utility",
            "goodput against the utility best response",
            "average backlog against the utility best response",
            "drop probability against the utility best response",
        ],
        "configurations": [asdict(spec) for spec in specs],
        "parameters_per_player": queue_parameter_count(specs[0]),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
    }


def run(
    output: Path,
    seeds: Sequence[int],
    steps: int,
    arrival_rates: Sequence[float],
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    locked_manifest = manifest(seeds, steps, arrival_rates)
    (output / "manifest.json").write_text(json.dumps(locked_manifest, indent=2), encoding="utf-8")
    failures_path = output / "failures.jsonl"
    failures_path.write_text("", encoding="utf-8")
    started = time.time()
    rows: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for spec in configurations(arrival_rates):
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
                    "config": spec.name,
                    "seed": int(seed),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                with failures_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure, sort_keys=True) + "\n")
            write_csv(output / "queue_results.csv", rows)
            write_csv(output / "queue_diagnostics.csv", diagnostics)
    summary = {
        **locked_manifest,
        "elapsed_seconds": time.time() - started,
        "completed_config_seed_pairs": len(
            {(str(row["config"]), int(row["seed"])) for row in rows}
        ),
        "expected_config_seed_pairs": len(arrival_rates) * len(seeds),
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--arrival-rates", type=float, nargs="+", default=list(ARRIVAL_RATES))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke:
        run(arguments.output, tuple(arguments.seeds[:1]), min(arguments.steps, 2), tuple(arguments.arrival_rates[:1]))
    else:
        run(arguments.output, tuple(arguments.seeds), arguments.steps, tuple(arguments.arrival_rates))
