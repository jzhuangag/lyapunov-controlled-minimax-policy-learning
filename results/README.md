# Results layout

The final publication figures are stored directly in this directory:

- `queue_nominal_benchmark.pdf/png`: Fig. 4, the six-method moderate-load benchmark.
- `queue_load_results.pdf/png`: Fig. 5, the light/moderate/near-capacity load sweep.

Machine-readable records are grouped by the two reported experiments:

- `queue_nominal/`: 12-seed raw checkpoints, controller diagnostics, locked manifest, failures, and paired statistics for Fig. 4.
- `queue_aware/`: 12-seed raw checkpoints, controller diagnostics, locked manifest, failures, and paired statistics for Fig. 5.

The manifests were written before confirmatory execution. Both failure logs are empty, and no seed or operating point was removed based on its outcome.
