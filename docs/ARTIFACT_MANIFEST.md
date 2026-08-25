# Artifact manifest

## Manuscripts

- `ICC2027_submit.tex/pdf`: submission form.
- `ICC2027_camera_ready.tex/pdf`: full author and funding metadata.
- `refs.bib`: the bibliography used by both versions.

The scientific bodies of the two TeX files are synchronized. Their author metadata differs by design.

## Figures

- `figures/mathematical_motivation.pdf`: generated mathematical motivation.
- `figures/system_model.pdf`: locked wireless-system artwork.
- `figures/algorithm_framework.pdf`: locked LCMPL framework artwork.
- `results/theory_aligned_v1_20260813/figures/neural_benchmark_theory.pdf`: locked Fig. 4.
- `results/theory_aligned_v1_20260813/figures/wireless_stress_rate.pdf`: locked Fig. 5.

## Data-to-figure map

- Fig. 4(a)–(c), (f): `icc_neural_hard/hard_neural_results.csv`.
- Fig. 4(d): `icc_neural_stress_12seed/neural_stress_diagnostics.csv`.
- Fig. 4(e): `oracle_quality_v2/oracle_quality_raw.csv`.
- Fig. 4(g)–(h): `icc_trajectory_sampled/trajectory_results.csv`.
- Fig. 5(a)–(d): `icc_neural_stress_12seed/neural_stress_results.csv`.

The repository omits caches, temporary builds, partial CSV duplicates, historical drafts, internal review notes, failed pilot runs, and third-party paper PDFs because none is required to reproduce the reported manuscript.
