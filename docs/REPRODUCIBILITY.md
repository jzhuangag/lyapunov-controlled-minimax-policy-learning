# Reproducibility protocol

## Scope

The repository supports three levels of reproduction:

1. **Locked-data reproduction:** regenerate Figs. 1, 4, and 5 and compile the manuscripts from the committed data.
2. **Smoke validation:** verify the transition model and run small population, sampled-oracle, oracle-quality, and wireless-stress jobs.
3. **Full rerun:** regenerate the fixed-seed population, trajectory-sampled, oracle-quality, and wireless-stress results in a new timestamped directory.

Figs. 2–3 are publication artwork and are retained as locked vector assets. They do not encode experimental data.

## Environment

The recorded environment is:

| Component | Version |
|---|---:|
| Python | 3.8.12 |
| NumPy | 1.20.3 |
| SciPy | 1.7.3 |
| pandas | 1.4.1 |
| PyTorch | 1.11.0+cpu |
| Matplotlib | 3.5.1 |

The experiments run on CPU with one PyTorch thread. `environment.yml` recreates the Conda environment. `requirements.txt` records the Python package versions for reference.

## Fixed protocols

### Population-oracle benchmark and scaling

- Nominal configuration: eight channels, hidden width 64, 1608 parameters per player.
- Nominal seeds: 400–411.
- Nominal methods: LCMPL (`QP+G`), LCMPL-F (`noG`), Minimax PPO, GDA, EGM, and PPM-3.
- Joint updates: 30; checkpoint interval: 5.
- Channel-scaling seeds: 420–427.
- The exact finite game supplies the population saddle field and Jacobian–field product.

### Trajectory-sampled benchmark

- Configuration: the same eight-channel, width-64 game.
- Seeds: 400–411.
- Methods: LCMPL, LCMPL-F, GDA, EGM, and PPM-3.
- Joint updates: 30.
- Horizon: 64.
- Trajectories per update: 128.
- Transitions per method and seed: 245,760.
- One shared on-policy batch supplies both players' empirical saddle field and the same-batch automatic-differentiation JVP.
- The code does not claim that the sampled JVP is unbiased.

### Oracle-quality diagnostic

- Training seeds: 400–411.
- Fixed checkpoints: 0, 15, and 30.
- Batch sizes: 32, 64, 128, and 256 trajectories.
- Four fixed replicates per seed/checkpoint/batch-size cell.
- Total raw rows: 576.

### Wireless-stress study

- Six channels and hidden width 32.
- Seeds: 500–511.
- Methods: LCMPL, LCMPL-F, and PPM-3.
- Joint updates: 25.
- Sixteen plotted one-factor cells cover jammer-to-noise ratio, switching cost, adjacent-channel leakage, and exogenous radio-mode persistence.
- Each method is independently retrained at every plotted cell.

The full manifest is stored in `experiments/configs/final_icc2027.json`. Hyperparameter validation used seeds 0–1, disjoint from the locked test seeds. The validation table is retained under `results/exogenous_ph_v1_20260812_rerun2/learning_rate_validation/`.

## Commands

Verify the committed release:

```bash
python scripts/verify_release.py
```

Regenerate publication plots from locked raw data:

```bash
python scripts/reproduce.py figures
```

Compile both manuscripts:

```bash
python scripts/reproduce.py paper
```

Run smoke tests without modifying locked data:

```bash
python scripts/reproduce.py smoke
```

Run the full protocol in `outputs/full-<timestamp>/`:

```bash
python scripts/reproduce.py full
```

The runner records commands and timing in `run_manifest.json`. It fails if its output directory already exists.

## Locked data and expected dimensions

| Artifact | Expected rows |
|---|---:|
| `hard_neural_results.csv` | 1728 |
| `trajectory_results.csv` | 420 |
| `trajectory_diagnostics.csv` | 1800 |
| `oracle_quality_raw.csv` | 576 |
| `neural_stress_results.csv` | 3456 |

`scripts/verify_release.py` also checks the twelve-seed manifests, the 60 final trajectory method–seed cells, the equal 245,760-transition budget, the sixteen stress configurations, and the manuscript's five figure inputs.

## Determinism and PDF hashes

The locked CSV/JSON files and manuscript PDFs have exact hashes in `SHA256SUMS`. A fresh numerical run can differ in wall-clock fields and may exhibit platform-level floating-point variation. Matplotlib and pdfTeX also embed creation metadata, so regenerated PDF bytes need not match even when the rendered scientific content does.
