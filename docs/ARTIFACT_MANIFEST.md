# Artifact manifest

## Manuscripts

- `ICC2027_submit.tex/pdf`: submission form.
- `ICC2027_camera_ready.tex/pdf`: full author and funding metadata.
- `refs.bib`: the bibliography used by both versions.

`ICC2027_camera_ready.tex` defines the camera-ready flag and inputs `ICC2027_submit.tex`; the scientific body is therefore shared by construction. Author and funding metadata differ by design.

## Figures

- `figures/mathematical_motivation.pdf`: generated mathematical motivation.
- `figures/system_model_queue.pdf`: generated queue-aware Markov-game diagram.
- `figures/algorithm_framework_queue.pdf`: generated LCMPL framework diagram.
- `results/queue_nominal_benchmark.pdf`: generated Fig. 4.
- `results/queue_load_results.pdf`: generated Fig. 5.

## Data-to-figure map

- Fig. 4(a)–(d): `results/queue_nominal/queue_nominal_results.csv`.
- Fig. 5(a)–(d): `results/queue_aware/queue_results.csv`.
- Curvature-activation claims: `results/queue_aware/queue_diagnostics.csv`.
- Locked seeds, configurations, failures, and software versions: the `manifest.json`, `summary.json`, and `failures.jsonl` files beside each result table.

The repository omits caches, temporary builds, partial CSV duplicates, historical drafts, internal review notes, failed pilot runs, and third-party paper PDFs because none is required to reproduce the reported manuscript.
