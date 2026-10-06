# Reproducibility protocol

## Scope

The repository supports three levels of reproduction:

1. **Locked-data reproduction:** regenerate the paper figures and compile both manuscripts from committed data.
2. **Smoke validation:** verify the action-dependent queue transition and run a minimal queue-aware learning job.
3. **Full rerun:** regenerate the fixed-seed nominal benchmark and traffic-load stress in a new timestamped directory.

Figs. 2–3 are generated vector diagrams. Figs. 4–5 are generated from the committed per-seed CSV files.

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
The release verifier checks that both files agree with `results/package_versions.json`, which records the environment used for the locked runs.

## Fixed protocols

### Queue-aware Markov game

- State: radio mode, queue length, and previous transmitter channel.
- Four channels, four radio modes, queue capacity four, hidden width 32, and 452 parameters per player.
- Every slot has an independent Bernoulli arrival; service is Bernoulli with a probability controlled by both channel actions.
- Both channel actions therefore affect the stochastic next-queue distribution.
- The transition test checks stochastic rows and dependence on each player's action.

### Nominal six-method benchmark

- Arrival rate: 0.45 packet/slot.
- Seeds: 800–811.
- Nominal methods: LCMPL (`QP+G`), LCMPL-F (`noG`), Minimax PPO, GDA, EGM, and PPM-3.
- Joint updates: 60; checkpoint interval: 5.
- Exact dynamic programming evaluates worst-case queue utility, exploitability, delivered goodput, backlog, and dropping.

### Traffic-load stress

- Arrival rates: 0.20, 0.45, and 0.70 packet/slot.
- Seeds: 800–811.
- Methods: LCMPL, LCMPL-F, GDA, EGM, and PPM-3.
- Joint updates: 60.
- Each method is independently retrained at every load.

The radio configuration uses jammer-to-noise ratio 30. LCMPL uses step-size box `[0,0.16] × [0,0.12]`, eight backtracking trials, and merit weights `(0.35,1.5)`. On disjoint validation seeds 100–103, GDA, EGM, and PPM-3 select step size 0.16 from `{0.01,0.04,0.08,0.16}`; Minimax PPO uses step size 0.001, clip 0.2, and three epochs. The full manifest is stored in `experiments/configs/final_icc2027.json`; no test seed was removed or reordered based on results.

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
| `queue_nominal_results.csv` | 936 |
| `queue_nominal_diagnostics.csv` | 4320 |
| `queue_results.csv` | 2340 |
| `queue_diagnostics.csv` | 10800 |

`scripts/verify_release.py` also checks both empty failure logs through their summaries, the 12-seed nominal run, all 36 load–seed cells, dependency pins, and the manuscript's five figure inputs.

## Determinism and PDF hashes

The locked CSV/JSON files and manuscript PDFs have hashes in `SHA256SUMS`. Text files use canonical LF line endings so the check is stable on Windows and Linux; binary files use their exact bytes. A fresh numerical run can differ in wall-clock fields and may exhibit platform-level floating-point variation. Matplotlib and pdfTeX also embed creation metadata, so regenerated PDF bytes need not match even when the rendered scientific content does.
