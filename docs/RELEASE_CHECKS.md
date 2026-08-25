# Release checks

Validated on 2026-08-25 with the recorded Windows/Python environment.

- `scripts/verify_release.py`: passed.
- Transition-kernel validation: passed, including row normalization, action independence, and next previous-channel identity.
- Population smoke run: completed for two fixed smoke seeds.
- Trajectory-sampled smoke run: completed with one shared on-policy batch per update.
- Same-batch oracle-quality smoke run: completed with zero recorded failures.
- Wireless-stress smoke run: completed all sixteen one-factor cells for seed 500.
- Locked-data plotting: regenerated the mathematical-motivation, neural-benchmark, and wireless-stress figures.
- LaTeX: both manuscript sources compiled to six-page, 10-point, US-Letter PDFs.
- References: final LaTeX logs contained no undefined citations or cross-references.
- Fonts: all manuscript fonts are embedded; no Type 3 fonts were found.
- Visual regression: all twelve rendered manuscript pages were pixel-identical to the locked PDFs at 120 dpi.

PDF byte hashes can change after recompilation because Matplotlib and pdfTeX embed creation metadata. The committed locked PDFs and all other release files are covered by `SHA256SUMS`.
