# Experiment code layout

- `run/run_queue_nominal_benchmark.py`: locked six-method queue-aware benchmark.
- `run/run_queue_aware_stress.py`: locked light/moderate/heavy load sweep.
- `run/run_neural_scale_game.py`: shared policy, field, curvature, best-response, and LCMPL routines.
- `plot/plot_queue_nominal_benchmark.py`: Fig. 4 and its paired-statistics audit.
- `plot/plot_queue_aware_results.py`: Fig. 5 and its paired-statistics audit.
- `plot/plot_queue_system_model.py` and `plot/plot_queue_algorithm_framework.py`: Figs. 2--3.
- `tests/test_queue_transition_model.py`: verifies stochastic rows and dependence on both players' actions.
- `configs/final_icc2027.json`: locked seeds, budgets, methods, traffic loads, and radio parameters.

The confirmatory scripts create a new output directory and refuse to overwrite an existing run. The committed manifests record all fixed seeds; no row is removed or reordered based on its outcome.
