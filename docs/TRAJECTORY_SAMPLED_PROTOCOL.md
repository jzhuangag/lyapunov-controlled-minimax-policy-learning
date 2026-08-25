# Locked trajectory-sampled ICC protocol

## Research question

Does the LCMPL curvature-coordinate gain persist when the saddle-field and
Jacobian--field direction are estimated from finite on-policy trajectories in
the `Hard-C8-H64` wireless game?

## Protocol locked before formal seeds

- Environment: `Hard-C8-H64`, unchanged from the population-oracle study.
- Neural policies: separate transmitter and jammer `16-64-8` tanh--softmax
  networks, initialized exactly as in the population experiment.
- Methods: LCMPL (`QP+G`), LCMPL-F (`noG`), and `PPM-3`.
- Seeds: `400--411`, fixed before the run and identical to the population study.
- Updates: 30 simultaneous joint-policy updates; no warm-up.
- Rollout horizon: 64 transitions.
- Per-update sample budget: 128 trajectories and 8192 transitions for every
  method.
- Total budget: 245760 transitions per method and seed.
- Oracle: per-decision likelihood-ratio finite-horizon policy-gradient field.
- Curvature: autograd JVP of the same empirical field on the same batch.
- Batch sharing: the transmitter, jammer, field, and JVP use the same batch.
  PPM-3 also reuses one behavior batch for its three inner field evaluations,
  with explicit prefix likelihood ratios at the displaced policies.
- Critic/baseline: none.
- Learning-rate/QP settings: unchanged from the population experiment;
  PPM-3 uses 0.03.
- Merit and safeguard: the exact finite-game composite Lyapunov merit and
  hard-best-response rate gate remain unchanged.  They are model-assisted
  training quantities.
- Checkpoint evaluation: exact finite-game hard best responses and population
  saddle-field norm every five updates.
- Primary statistics: paired final LCMPL minus LCMPL-F worst-case-rate gain and
  paired LCMPL-F minus LCMPL exploitability, with Student-t 95% intervals.

The batch size is fixed from the parameter dimension, discount horizon, and
the previously validated higher-order likelihood-ratio estimator.  No ICC
evaluation seed is used to select it.  The experiment is therefore described
as a trajectory-sampled saddle-field oracle test, not as fully model-free
LCMPL.
