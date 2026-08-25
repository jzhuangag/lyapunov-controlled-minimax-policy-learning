"""Deterministic validation of the exogenous radio-mode transition model."""

from __future__ import annotations

import numpy as np

from run_neural_scale_game import GameSpec, build_game, validate_transition_model


def main() -> None:
    for channels in (4, 6, 8, 10, 12):
        for rho_h in (0.40, 0.55, 0.70, 0.75, 0.85):
            game = build_game(
                GameSpec(
                    name=f"validation-C{channels}-rho{rho_h:.2f}",
                    channels=channels,
                    hidden_width=8,
                    rho_h=rho_h,
                )
            )
            validate_transition_model(game)
            transitions = game.transitions.numpy()
            assert np.allclose(transitions.sum(axis=-1), 1.0, atol=1.0e-12)
    print("transition-model validation passed")


if __name__ == "__main__":
    main()
