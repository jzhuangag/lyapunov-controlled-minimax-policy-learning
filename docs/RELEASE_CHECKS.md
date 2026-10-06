# Release checks

Validated on 2026-10-06 with the recorded Windows/Python environment.

- `scripts/verify_release.py`: passed.
- Queue-transition validation: passed row normalization, dependence on both players' actions, and previous-channel identity.
- Confirmatory nominal run: completed all six methods and twelve fixed seeds with zero failures.
- Confirmatory load sweep: completed three loads, five methods, and twelve fixed seeds with zero failures.
- Maximum dynamic-programming best-response residual: below $1.3\times10^{-14}$.
- Locked-data plotting: regenerated the mathematical motivation, system diagram, framework diagram, nominal benchmark, and load-stress figures.
- Release integrity: verified 936 nominal result rows, 4320 nominal diagnostic rows, 2340 load result rows, 10800 load diagnostic rows, dependency pins, and 58 file hashes.
- LaTeX: both manuscript sources compiled to six-page, 10-point, US-Letter PDFs.
- References: final LaTeX logs contained no undefined citations or cross-references.
- Fonts: all manuscript fonts are embedded; no Type 3 fonts were found.
- Visual inspection: all twelve manuscript pages were rendered and checked after the queue-aware revision.

PDF byte hashes can change after recompilation because Matplotlib and pdfTeX embed creation metadata. The committed locked PDFs and all other release files are covered by `SHA256SUMS`.
