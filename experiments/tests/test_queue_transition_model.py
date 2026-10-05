"""Validate the queue-aware joint-action-dependent Markov transition."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parents[1] / "run"
sys.path.insert(0, str(RUN_DIR))

from run_queue_aware_stress import QueueGameSpec, build_queue_game


def main() -> None:
    game, _ = build_queue_game(QueueGameSpec("test", arrival_rate=0.45))
    transition = game.transitions.detach().cpu().numpy()
    assert np.allclose(transition.sum(axis=-1), 1.0, atol=1.0e-12)
    transmitter_sensitive = any(
        not np.allclose(transition[state, 0, 0], transition[state, -1, 0])
        for state in range(len(game.rho))
    )
    jammer_sensitive = any(
        not np.allclose(transition[state, 0, 0], transition[state, 0, -1])
        for state in range(len(game.rho))
    )
    assert transmitter_sensitive
    assert jammer_sensitive
    print("queue-aware transition validation passed")


if __name__ == "__main__":
    main()
