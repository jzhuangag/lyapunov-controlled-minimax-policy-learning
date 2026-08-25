"""Neural-scale adversarial spectrum-access benchmark.

The benchmark generalizes the four-channel linear-softmax experiment to a
variable number of channels and two-layer tanh policy networks.  It records a
main eight-channel learning run and two controlled scaling studies: channel
count and hidden-layer width.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from scipy.special import logsumexp


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DISCOUNT = 0.95
ENTROPY_TAU = 0.02
MERIT_GAP_WEIGHT = 0.35
MERIT_DEFICIENCY_WEIGHT = 1.50
BETA_CAP = 0.08
BACKTRACKS = 8
LR_GRID = (0.01, 0.03, 0.06)
PPO_CLIP = 0.20
PPO_EPOCHS = 3
PPO_LEARNING_RATE = 0.001
FIXED_LEARNING_RATE = 0.01


@dataclass(frozen=True)
class GameSpec:
    name: str
    channels: int
    hidden_width: int
    jammer_to_noise: float = 20.0
    rho_h: float = 0.75
    switch_cost: float = 0.10
    adjacent_leakage: float = 0.80


@dataclass
class Game:
    spec: GameSpec
    features: torch.Tensor
    rewards: torch.Tensor
    transitions: torch.Tensor
    rho: torch.Tensor
    input_dim: int
    block_size: int


def parameter_count(channels: int, hidden_width: int) -> int:
    input_dim = 2 * channels
    return hidden_width * input_dim + hidden_width + channels * hidden_width + channels


def build_game(spec: GameSpec) -> Game:
    channels = spec.channels
    modes = channels
    input_dim = 2 * channels
    state_count = modes * channels
    base_gains = np.geomspace(1.9, 0.18, channels)
    gain_table = np.stack([np.roll(base_gains, shift) for shift in range(modes)])
    mode_transition = np.full((modes, modes), (1.0 - spec.rho_h) / (modes - 1))
    np.fill_diagonal(mode_transition, spec.rho_h)
    features = np.zeros((state_count, input_dim), dtype=float)
    rewards = np.zeros((state_count, channels, channels), dtype=float)
    transitions = np.zeros((state_count, channels, channels, state_count), dtype=float)
    for mode in range(modes):
        gains = gain_table[mode]
        normalized_gains = 2.0 * (gains - base_gains.min()) / (base_gains.max() - base_gains.min()) - 1.0
        for previous_channel in range(channels):
            state = mode * channels + previous_channel
            features[state, :channels] = normalized_gains
            features[state, channels + previous_channel] = 1.0
            for transmit_channel in range(channels):
                switching_penalty = spec.switch_cost * float(transmit_channel != previous_channel)
                for jammed_channel in range(channels):
                    cyclic_distance = min(
                        (transmit_channel - jammed_channel) % channels,
                        (jammed_channel - transmit_channel) % channels,
                    )
                    if cyclic_distance == 0:
                        leakage = 1.0
                    elif cyclic_distance == 1:
                        leakage = spec.adjacent_leakage
                    elif cyclic_distance == 2:
                        leakage = 0.12
                    else:
                        leakage = 0.025
                    interference = spec.jammer_to_noise * leakage
                    rewards[state, transmit_channel, jammed_channel] = math.log2(
                        1.0 + 10.0 * gains[transmit_channel] / (1.0 + interference)
                    ) - switching_penalty
                    for new_mode in range(modes):
                        new_state = new_mode * channels + transmit_channel
                        transitions[state, transmit_channel, jammed_channel, new_state] = mode_transition[mode, new_mode]
    game = Game(
        spec=spec,
        features=torch.tensor(features),
        rewards=torch.tensor(rewards),
        transitions=torch.tensor(transitions),
        rho=torch.full((state_count,), 1.0 / state_count),
        input_dim=input_dim,
        block_size=parameter_count(channels, spec.hidden_width),
    )
    validate_transition_model(game)
    return game


def validate_transition_model(game: Game) -> None:
    """Verify the exogenous radio-mode kernel and previous-channel state update."""
    transitions = game.transitions.detach().cpu().numpy()
    channels = game.spec.channels
    if not np.allclose(transitions.sum(axis=-1), 1.0, atol=1.0e-12):
        raise RuntimeError("Transition rows do not sum to one.")
    for state in range(transitions.shape[0]):
        mode = state // channels
        reference_by_action = []
        for transmit_channel in range(channels):
            reference = transitions[state, transmit_channel, 0].reshape(channels, channels).sum(axis=1)
            reference_by_action.append(reference)
            for jammed_channel in range(channels):
                row = transitions[state, transmit_channel, jammed_channel].reshape(channels, channels)
                if not np.allclose(row.sum(axis=1), reference, atol=1.0e-12):
                    raise RuntimeError("Radio-mode transition depends on jammer action.")
                invalid_previous = np.delete(row, transmit_channel, axis=1)
                if not np.allclose(invalid_previous, 0.0, atol=1.0e-12):
                    raise RuntimeError("Stored next previous-channel index differs from transmitter action.")
        if not all(np.allclose(reference_by_action[0], value, atol=1.0e-12) for value in reference_by_action[1:]):
            raise RuntimeError(f"Radio-mode transition depends on transmitter action at mode {mode}.")


def split_blocks(z: torch.Tensor, game: Game) -> Tuple[torch.Tensor, torch.Tensor]:
    return z[: game.block_size], z[game.block_size :]


def policy_from_block(block: torch.Tensor, game: Game) -> torch.Tensor:
    channels = game.spec.channels
    hidden_width = game.spec.hidden_width
    offset = 0
    first_size = hidden_width * game.input_dim
    w1 = block[offset : offset + first_size].reshape(hidden_width, game.input_dim)
    offset += first_size
    b1 = block[offset : offset + hidden_width]
    offset += hidden_width
    second_size = channels * hidden_width
    w2 = block[offset : offset + second_size].reshape(channels, hidden_width)
    offset += second_size
    b2 = block[offset : offset + channels]
    hidden = torch.tanh(game.features @ w1.T + b1)
    return torch.softmax(hidden @ w2.T + b2, dim=1)


def policies(z: torch.Tensor, game: Game) -> Tuple[torch.Tensor, torch.Tensor]:
    transmitter, jammer = split_blocks(z, game)
    return policy_from_block(transmitter, game), policy_from_block(jammer, game)


def return_from_policies(
    transmitter: torch.Tensor,
    jammer: torch.Tensor,
    game: Game,
    regularized: bool,
) -> torch.Tensor:
    reward = torch.einsum("sa,sab,sb->s", transmitter, game.rewards, jammer)
    transition = torch.einsum("sa,sabn,sb->sn", transmitter, game.transitions, jammer)
    if regularized:
        h_transmitter = -(transmitter * torch.log(transmitter.clamp_min(1.0e-15))).sum(dim=1)
        h_jammer = -(jammer * torch.log(jammer.clamp_min(1.0e-15))).sum(dim=1)
        reward = reward + ENTROPY_TAU * (h_transmitter - h_jammer)
    identity = torch.eye(len(game.rho), dtype=transition.dtype)
    value = torch.linalg.solve(identity - DISCOUNT * transition, reward)
    return (1.0 - DISCOUNT) * (game.rho @ value)


def objective(z: torch.Tensor, game: Game, regularized: bool = True) -> torch.Tensor:
    return return_from_policies(*policies(z, game), game, regularized)


def saddle_field(z_value: torch.Tensor, game: Game, create_graph: bool = False) -> torch.Tensor:
    z = z_value if z_value.requires_grad else z_value.detach().requires_grad_(True)
    value = objective(z, game, regularized=True)
    gradient = torch.autograd.grad(value, z, create_graph=create_graph)[0]
    signs = torch.cat((-torch.ones(game.block_size), torch.ones(game.block_size)))
    return signs * gradient


def field_and_curvature(z_value: torch.Tensor, game: Game) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    z = z_value.detach().requires_grad_(True)
    field = saddle_field(z, game).detach()

    def field_map(point: torch.Tensor) -> torch.Tensor:
        return saddle_field(point, game, create_graph=True)

    _, curvature = torch.autograd.functional.jvp(field_map, z, field, create_graph=False, strict=False)
    curvature = curvature.detach()
    f_norm = float(torch.linalg.vector_norm(field))
    g_norm = float(torch.linalg.vector_norm(curvature))
    ratio = g_norm / max(f_norm, 1.0e-15)
    cosine = float(torch.dot(field, curvature) / max(f_norm * g_norm, 1.0e-15))
    return field, curvature, ratio, cosine


def population_policy_advantages(
    z: torch.Tensor,
    game: Game,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact on-policy occupancies and advantages for minimax PPO.

    The calculation uses the same discounted Markov game as the population
    saddle-field methods.  It changes only the policy update rule and therefore
    preserves the common neural-policy backbone used in the benchmark.
    """
    with torch.no_grad():
        transmitter, jammer = policies(z, game)
        reward = torch.einsum("sa,sab,sb->s", transmitter, game.rewards, jammer)
        transition = torch.einsum("sa,sabn,sb->sn", transmitter, game.transitions, jammer)
        identity = torch.eye(len(game.rho), dtype=transition.dtype)
        value = torch.linalg.solve(identity - DISCOUNT * transition, reward)
        joint_q = game.rewards + DISCOUNT * torch.einsum("sabn,n->sab", game.transitions, value)
        state_value = torch.einsum("sa,sab,sb->s", transmitter, joint_q, jammer)
        transmitter_advantage = torch.einsum("sab,sb->sa", joint_q, jammer) - state_value[:, None]
        jammer_advantage = torch.einsum("sa,sab->sb", transmitter, joint_q) - state_value[:, None]
        occupancy = (1.0 - DISCOUNT) * torch.linalg.solve(
            identity - DISCOUNT * transition.T,
            game.rho,
        )

        transmitter_scale = torch.sqrt(
            torch.sum(occupancy[:, None] * transmitter * transmitter_advantage.square())
        ).clamp_min(1.0e-8)
        jammer_scale = torch.sqrt(
            torch.sum(occupancy[:, None] * jammer * jammer_advantage.square())
        ).clamp_min(1.0e-8)
        transmitter_advantage = transmitter_advantage / transmitter_scale
        jammer_advantage = jammer_advantage / jammer_scale
    return transmitter, jammer, occupancy, transmitter_advantage, jammer_advantage


def minimax_ppo_update(
    z: torch.Tensor,
    game: Game,
    learning_rate: float,
) -> torch.Tensor:
    """Apply simultaneous PPO updates to the transmitter and jammer policies."""
    old_transmitter, old_jammer, occupancy, transmitter_advantage, jammer_advantage = (
        population_policy_advantages(z, game)
    )
    old_theta, old_psi = split_blocks(z, game)
    theta = old_theta.detach().clone().requires_grad_(True)
    psi = old_psi.detach().clone().requires_grad_(True)
    theta_optimizer = torch.optim.Adam([theta], lr=learning_rate)
    psi_optimizer = torch.optim.Adam([psi], lr=learning_rate)
    for _ in range(PPO_EPOCHS):
        transmitter = policy_from_block(theta, game)
        transmitter_ratio = transmitter / old_transmitter.clamp_min(1.0e-12)
        transmitter_unclipped = transmitter_ratio * transmitter_advantage
        transmitter_clipped = torch.clamp(
            transmitter_ratio,
            1.0 - PPO_CLIP,
            1.0 + PPO_CLIP,
        ) * transmitter_advantage
        transmitter_surrogate = torch.sum(
            occupancy[:, None]
            * old_transmitter
            * torch.minimum(transmitter_unclipped, transmitter_clipped)
        )
        transmitter_entropy = -torch.sum(
            occupancy[:, None]
            * transmitter
            * torch.log(transmitter.clamp_min(1.0e-15))
        )
        theta_optimizer.zero_grad()
        (-(transmitter_surrogate + ENTROPY_TAU * transmitter_entropy)).backward()
        theta_optimizer.step()

        jammer = policy_from_block(psi, game)
        jammer_ratio = jammer / old_jammer.clamp_min(1.0e-12)
        minimizer_advantage = -jammer_advantage
        jammer_unclipped = jammer_ratio * minimizer_advantage
        jammer_clipped = torch.clamp(
            jammer_ratio,
            1.0 - PPO_CLIP,
            1.0 + PPO_CLIP,
        ) * minimizer_advantage
        jammer_surrogate = torch.sum(
            occupancy[:, None]
            * old_jammer
            * torch.minimum(jammer_unclipped, jammer_clipped)
        )
        jammer_entropy = -torch.sum(
            occupancy[:, None]
            * jammer
            * torch.log(jammer.clamp_min(1.0e-15))
        )
        psi_optimizer.zero_grad()
        (-(jammer_surrogate + ENTROPY_TAU * jammer_entropy)).backward()
        psi_optimizer.step()
    return torch.cat((theta.detach(), psi.detach()))


def policies_numpy(z: torch.Tensor, game: Game) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        transmitter, jammer = policies(z, game)
    return transmitter.cpu().numpy(), jammer.cpu().numpy()


def entropy(probabilities: np.ndarray) -> np.ndarray:
    return -(probabilities * np.log(np.maximum(probabilities, 1.0e-15))).sum(axis=1)


def best_response_value(
    fixed_policy: np.ndarray,
    game: Game,
    maximizing: bool,
    regularized: bool,
    tolerance: float = 1.0e-10,
    max_iterations: int = 500,
) -> Tuple[float, float]:
    rewards = game.rewards.cpu().numpy()
    transitions = game.transitions.cpu().numpy()
    rho = game.rho.cpu().numpy()
    fixed_entropy = entropy(fixed_policy)
    if maximizing:
        action_reward = np.einsum("sb,sab->sa", fixed_policy, rewards)
        action_transition = np.einsum("sb,sabn->san", fixed_policy, transitions)
        if regularized:
            action_reward -= ENTROPY_TAU * fixed_entropy[:, None]
    else:
        action_reward = np.einsum("sa,sab->sb", fixed_policy, rewards)
        action_transition = np.einsum("sa,sabn->sbn", fixed_policy, transitions)
        if regularized:
            action_reward += ENTROPY_TAU * fixed_entropy[:, None]
    value = np.zeros(len(rho), dtype=float)
    identity = np.eye(len(rho))
    residual = math.inf
    previous_actions = None
    for _ in range(max_iterations):
        q_values = action_reward + DISCOUNT * (action_transition @ value)
        if regularized:
            logits = q_values / ENTROPY_TAU if maximizing else -q_values / ENTROPY_TAU
            logits -= logits.max(axis=1, keepdims=True)
            response = np.exp(logits)
            response /= response.sum(axis=1, keepdims=True)
            policy_reward = np.einsum("sa,sa->s", response, action_reward)
            response_entropy = entropy(response)
            policy_reward += ENTROPY_TAU * response_entropy if maximizing else -ENTROPY_TAU * response_entropy
            policy_transition = np.einsum("sa,san->sn", response, action_transition)
            updated = np.linalg.solve(identity - DISCOUNT * policy_transition, policy_reward)
            bellman_q = action_reward + DISCOUNT * (action_transition @ updated)
            bellman = (
                ENTROPY_TAU * logsumexp(bellman_q / ENTROPY_TAU, axis=1)
                if maximizing
                else -ENTROPY_TAU * logsumexp(-bellman_q / ENTROPY_TAU, axis=1)
            )
            stable = False
        else:
            actions = q_values.argmax(axis=1) if maximizing else q_values.argmin(axis=1)
            policy_reward = action_reward[np.arange(len(rho)), actions]
            policy_transition = action_transition[np.arange(len(rho)), actions]
            updated = np.linalg.solve(identity - DISCOUNT * policy_transition, policy_reward)
            bellman_q = action_reward + DISCOUNT * (action_transition @ updated)
            bellman = bellman_q.max(axis=1) if maximizing else bellman_q.min(axis=1)
            stable = previous_actions is not None and np.array_equal(actions, previous_actions)
            previous_actions = actions
        residual = float(np.max(np.abs(bellman - updated)))
        value = updated
        if residual <= tolerance or stable:
            return float((1.0 - DISCOUNT) * (rho @ value)), residual
    raise RuntimeError(f"Best response did not converge; residual={residual:.3e}.")


def metrics(z: torch.Tensor, game: Game) -> Dict[str, float]:
    transmitter, jammer = policies_numpy(z, game)
    hard_max, residual_a = best_response_value(jammer, game, maximizing=True, regularized=False)
    hard_min, residual_b = best_response_value(transmitter, game, maximizing=False, regularized=False)
    field = saddle_field(z.detach().requires_grad_(True), game).detach()
    return {
        "robust_rate": hard_min,
        "hard_exploitability": hard_max - hard_min,
        "field_norm": float(torch.linalg.vector_norm(field)),
        "maximum_br_residual": max(residual_a, residual_b),
    }


def merit_components(z: torch.Tensor, game: Game) -> Tuple[float, float, float]:
    field = saddle_field(z.detach().requires_grad_(True), game).detach()
    transmitter, jammer = policies_numpy(z, game)
    soft_max, _ = best_response_value(jammer, game, maximizing=True, regularized=True)
    soft_min, _ = best_response_value(transmitter, game, maximizing=False, regularized=True)
    regularized_upper = float(torch.max(game.rewards)) + ENTROPY_TAU * math.log(game.spec.channels)
    return (
        0.5 * float(torch.dot(field, field)),
        max(soft_max - soft_min, 0.0),
        max(regularized_upper - soft_min, 0.0),
    )


def merit(z: torch.Tensor, game: Game, scales: Tuple[float, float, float]) -> float:
    energy, gap, deficiency = merit_components(z, game)
    return (
        energy / scales[0]
        + MERIT_GAP_WEIGHT * (gap / scales[1]) ** 2
        + MERIT_DEFICIENCY_WEIGHT * (deficiency / scales[2]) ** 2
    )


def solve_box_qp(gradient: np.ndarray, hessian: np.ndarray, upper: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
    psd = eigenvectors @ np.diag(np.maximum(eigenvalues, 1.0e-6)) @ eigenvectors.T
    candidates = [np.zeros(2), np.array([upper[0], 0.0]), np.array([0.0, upper[1]]), upper.copy()]
    try:
        candidates.append(np.clip(-np.linalg.solve(psd, gradient), 0.0, upper))
    except np.linalg.LinAlgError:
        pass
    for x_value in (0.0, upper[0]):
        y_value = np.clip(-(gradient[1] + psd[0, 1] * x_value) / psd[1, 1], 0.0, upper[1])
        candidates.append(np.array([x_value, y_value]))
    for y_value in (0.0, upper[1]):
        x_value = np.clip(-(gradient[0] + psd[0, 1] * y_value) / psd[0, 0], 0.0, upper[0])
        candidates.append(np.array([x_value, y_value]))
    return min(candidates, key=lambda point: float(gradient @ point + 0.5 * point @ psd @ point))


def local_coefficients(
    z: torch.Tensor,
    field: torch.Tensor,
    curvature: torch.Tensor | None,
    game: Game,
    scales: Tuple[float, float, float],
) -> Tuple[float, float, float, Dict[str, float]]:
    value_0 = merit(z, game, scales)
    f_norm = float(torch.linalg.vector_norm(field))
    beta_probe = min(0.012, 0.02 / max(f_norm, 1.0e-12))
    f_plus = merit(z - beta_probe * field, game, scales)
    f_minus = merit(z + beta_probe * field, game, scales)
    linear_f = (f_plus - f_minus) / (2.0 * beta_probe)
    quadratic_ff = max((f_plus - 2.0 * value_0 + f_minus) / (beta_probe**2), 1.0e-6)
    beta_no_g = float(np.clip(-linear_f / quadratic_ff, 0.0, BETA_CAP))
    if curvature is None:
        predicted = linear_f * beta_no_g + 0.5 * quadratic_ff * beta_no_g**2
        return beta_no_g, 0.0, beta_no_g, {
            "probe_count": 3.0,
            "predicted_model_change": predicted,
        }
    g_norm = float(torch.linalg.vector_norm(curvature))
    if g_norm <= 1.0e-14:
        predicted = linear_f * beta_no_g + 0.5 * quadratic_ff * beta_no_g**2
        return beta_no_g, 0.0, beta_no_g, {
            "probe_count": 3.0,
            "predicted_model_change": predicted,
        }
    gamma_probe = float(np.clip(beta_probe * f_norm / g_norm, 2.0e-4, 0.025))
    gamma_upper = float(np.clip(BETA_CAP * f_norm / g_norm, 5.0e-4, 0.12))
    g_plus = merit(z + gamma_probe * curvature, game, scales)
    g_minus = merit(z - gamma_probe * curvature, game, scales)
    linear_g = (g_plus - g_minus) / (2.0 * gamma_probe)
    quadratic_gg = max((g_plus - 2.0 * value_0 + g_minus) / (gamma_probe**2), 1.0e-6)
    gradient = np.array([linear_f, linear_g])
    hessian = np.diag([quadratic_ff, quadratic_gg])
    point = solve_box_qp(
        gradient,
        hessian,
        np.array([BETA_CAP, gamma_upper]),
    )
    predicted = float(gradient @ point + 0.5 * point @ hessian @ point)
    return float(point[0]), float(point[1]), beta_no_g, {
        "probe_count": 5.0,
        "predicted_model_change": predicted,
    }


def safeguard(
    z: torch.Tensor,
    field: torch.Tensor,
    curvature: torch.Tensor | None,
    beta: float,
    gamma: float,
    game: Game,
    scales: Tuple[float, float, float],
) -> Tuple[torch.Tensor, float, float, int]:
    value_0 = merit(z, game, scales)
    correction = torch.zeros_like(field) if curvature is None else curvature
    factor = 1.0
    for backtrack in range(BACKTRACKS + 1):
        candidate = (z - factor * beta * field + factor * gamma * correction).detach()
        if merit(candidate, game, scales) <= value_0 + 1.0e-10:
            return candidate, factor * beta, factor * gamma, backtrack
        factor *= 0.5
    return z.detach(), 0.0, 0.0, BACKTRACKS + 1


def adaptive_update(
    z: torch.Tensor,
    game: Game,
    scales: Tuple[float, float, float],
    use_curvature: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if use_curvature:
        field, curvature, ratio, cosine = field_and_curvature(z, game)
    else:
        field = saddle_field(z.detach().requires_grad_(True), game).detach()
        curvature = None
        ratio = 0.0
        cosine = 0.0
    value_before = merit(z, game, scales)
    beta, gamma, beta_no_g, model_diagnostics = local_coefficients(z, field, curvature, game, scales)
    candidate, beta_used, gamma_used, backtracks = safeguard(z, field, curvature, beta, gamma, game, scales)
    rejected_curvature = 0.0
    if use_curvature:
        no_g_candidate, _, _, _ = safeguard(z, field, None, beta_no_g, 0.0, game, scales)
        if (
            metrics(candidate, game)["robust_rate"] + 1.0e-10 < metrics(no_g_candidate, game)["robust_rate"]
            or merit(candidate, game, scales) > merit(no_g_candidate, game, scales) + 1.0e-10
        ):
            candidate = no_g_candidate
            beta_used = beta_no_g
            gamma_used = 0.0
            rejected_curvature = float(gamma > 0.0)
    realized_change = merit(candidate, game, scales) - value_before
    return candidate, {
        "beta": beta_used,
        "gamma": gamma_used,
        "g_over_f": ratio,
        "cosine_fg": cosine,
        "field_norm_oracle": float(torch.linalg.vector_norm(field)),
        "curvature_norm_oracle": 0.0 if curvature is None else float(torch.linalg.vector_norm(curvature)),
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


def classical_update(method: str, z: torch.Tensor, game: Game, learning_rate: float) -> torch.Tensor:
    if method == "Minimax-PPO":
        return minimax_ppo_update(z, game, learning_rate)
    field = saddle_field(z.detach().requires_grad_(True), game).detach()
    if method == "GDA":
        return (z - learning_rate * field).detach()
    if method == "EGM":
        predictor = (z - learning_rate * field).detach()
        predictor_field = saddle_field(predictor.requires_grad_(True), game).detach()
        return (z - learning_rate * predictor_field).detach()
    if method == "PPM-3":
        iterate = z.detach()
        for _ in range(3):
            implicit_field = saddle_field(iterate.requires_grad_(True), game).detach()
            iterate = (z - learning_rate * implicit_field).detach()
        return iterate
    raise ValueError(method)


def initialize(seed: int, game: Game) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260807 + seed)
    blocks = []
    for _ in range(2):
        w1 = torch.randn(
            game.spec.hidden_width,
            game.input_dim,
            generator=generator,
        ) / math.sqrt(game.input_dim)
        b1 = torch.zeros(game.spec.hidden_width)
        w2 = torch.randn(
            game.spec.channels,
            game.spec.hidden_width,
            generator=generator,
        ) / math.sqrt(game.spec.hidden_width)
        b2 = 0.005 * torch.randn(game.spec.channels, generator=generator)
        blocks.append(torch.cat((w1.flatten(), b1, w2.flatten(), b2)))
    return torch.cat(blocks)


def initial_scales(z: torch.Tensor, game: Game) -> Tuple[float, float, float]:
    components = merit_components(z, game)
    return max(components[0], 1.0e-12), max(components[1], 1.0e-8), max(components[2], 1.0e-8)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_configuration(
    spec: GameSpec,
    seeds: Iterable[int],
    steps: int,
    methods: Sequence[str],
    checkpoint_every: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    game = build_game(spec)
    result_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    for seed in seeds:
        initial = initialize(seed, game)
        scales = initial_scales(initial, game)
        for method in methods:
            z = initial.clone()
            cumulative_oracle_calls = 0.0
            cumulative_jvp_calls = 0.0
            method_started = time.perf_counter()
            last_diagnostics = {"beta": 0.0, "gamma": 0.0, "g_over_f": 0.0, "cosine_fg": 0.0}
            for step in range(steps + 1):
                if step % checkpoint_every == 0 or step == steps:
                    result_rows.append(
                        {
                            "config": spec.name,
                            "channels": spec.channels,
                            "hidden_width": spec.hidden_width,
                            "parameters_per_player": game.block_size,
                            "seed": seed,
                            "method": method,
                            "step": step,
                            "cumulative_environment_transitions": 0,
                            "cumulative_oracle_calls": cumulative_oracle_calls,
                            "cumulative_jvp_calls": cumulative_jvp_calls,
                            "wall_clock_seconds": time.perf_counter() - method_started,
                            **metrics(z, game),
                        }
                    )
                if step == steps:
                    break
                if method == "QP+G":
                    z, last_diagnostics = adaptive_update(z, game, scales, use_curvature=True)
                elif method == "noG":
                    z, last_diagnostics = adaptive_update(z, game, scales, use_curvature=False)
                else:
                    learning_rate = PPO_LEARNING_RATE if method == "Minimax-PPO" else FIXED_LEARNING_RATE
                    z = classical_update(method, z, game, learning_rate=learning_rate)
                    if method == "GDA":
                        last_diagnostics = {"beta": learning_rate, "gamma": 0.0, "field_oracle_count": 1.0, "jvp_count": 0.0}
                    elif method == "EGM":
                        last_diagnostics = {"beta": learning_rate, "gamma": 0.0, "field_oracle_count": 2.0, "jvp_count": 0.0}
                    elif method == "PPM-3":
                        last_diagnostics = {"beta": learning_rate, "gamma": 0.0, "field_oracle_count": 3.0, "jvp_count": 0.0}
                    else:
                        last_diagnostics = {"beta": learning_rate, "gamma": 0.0, "field_oracle_count": float(2 * PPO_EPOCHS), "jvp_count": 0.0}
                cumulative_oracle_calls += float(last_diagnostics.get("field_oracle_count", 0.0))
                cumulative_jvp_calls += float(last_diagnostics.get("jvp_count", 0.0))
                diagnostic_rows.append(
                    {
                        "config": spec.name,
                        "channels": spec.channels,
                        "hidden_width": spec.hidden_width,
                        "parameters_per_player": game.block_size,
                        "seed": seed,
                        "method": method,
                        "step": step + 1,
                        "cumulative_environment_transitions": 0,
                        "cumulative_oracle_calls": cumulative_oracle_calls,
                        "cumulative_jvp_calls": cumulative_jvp_calls,
                        "wall_clock_seconds": time.perf_counter() - method_started,
                        **last_diagnostics,
                    }
                )
        print(
            f"CONFIG={spec.name} seed={seed} channels={spec.channels} "
            f"hidden={spec.hidden_width} parameters={game.block_size}",
            flush=True,
        )
    return result_rows, diagnostic_rows


def run(output: Path, pilot: bool, smoke: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if pilot:
        configurations = [
            (
                GameSpec("Pilot-C6-H16", channels=6, hidden_width=16),
                (301,),
                3,
                ("QP+G", "noG"),
                1,
            )
        ]
    elif smoke:
        configurations = [
            (
                GameSpec("Smoke-C8-H24", channels=8, hidden_width=24),
                tuple(range(300, 306)),
                10,
                ("QP+G", "noG", "GDA", "EGM", "PPM-3"),
                2,
            )
        ]
    else:
        configurations = [
            (
                GameSpec("Main-C8-H24", channels=8, hidden_width=24),
                tuple(range(300, 306)),
                30,
                ("QP+G", "noG", "GDA", "EGM", "PPM-3"),
                5,
            ),
            *[
                (
                    GameSpec(f"Channels-C{channels}-H16", channels=channels, hidden_width=16),
                    tuple(range(320, 324)),
                    25,
                    ("QP+G", "noG", "PPM-3"),
                    5,
                )
                for channels in (4, 6, 8)
            ],
            *[
                (
                    GameSpec(f"Width-C6-H{hidden}", channels=6, hidden_width=hidden),
                    tuple(range(330, 334)),
                    25,
                    ("QP+G", "noG", "PPM-3"),
                    5,
                )
                for hidden in (8, 16, 32, 64)
            ],
        ]
    rows: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    for arguments in configurations:
        new_rows, new_diagnostics = run_configuration(*arguments)
        rows.extend(new_rows)
        diagnostics.extend(new_diagnostics)
        write_csv(output / "neural_results.csv", rows)
        write_csv(output / "neural_diagnostics.csv", diagnostics)
    summary = {
        "device": "CPU",
        "pilot": pilot,
        "smoke": smoke,
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
    parser.add_argument("--output", type=Path, default=Path("results/icc_neural_scale"))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.pilot and arguments.smoke:
        raise ValueError("Choose at most one of --pilot and --smoke.")
    run(arguments.output, arguments.pilot, arguments.smoke)
