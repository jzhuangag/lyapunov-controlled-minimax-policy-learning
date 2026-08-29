"""Reproduce locked figures, paper builds, smoke tests, or the full experiments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUTPUTS = ROOT / "outputs"
LOCKED = ROOT / "results" / "exogenous_ph_v1_20260813_tuned"
ANALYSIS = ROOT / "results" / "theory_aligned_v1_20260813"
STRESS_SEEDS = tuple(range(500, 512))


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def command_text(command: Sequence[str]) -> str:
    return " ".join('"{}"'.format(part) if " " in str(part) else str(part) for part in command)


def runtime_environment() -> Dict[str, str]:
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["NUMEXPR_NUM_THREADS"] = "1"
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", str(OUTPUTS / ".mplconfig"))
    return environment


def run(command: Sequence[str], records: List[Dict[str, object]], cwd: Path = ROOT) -> None:
    started = time.time()
    print("+ {}".format(command_text(command)), flush=True)
    completed = subprocess.run(list(command), cwd=str(cwd), env=runtime_environment(), check=False)
    records.append(
        {
            "command": [str(part) for part in command],
            "cwd": str(cwd),
            "returncode": completed.returncode,
            "elapsed_seconds": time.time() - started,
        }
    )
    if completed.returncode != 0:
        raise RuntimeError("Command failed with exit code {}: {}".format(completed.returncode, command_text(command)))


def python(script: str, *arguments: object) -> List[str]:
    return [sys.executable, str(EXPERIMENTS / script), *[str(argument) for argument in arguments]]


def new_output(prefix: str, requested: Optional[Path]) -> Path:
    output = requested.resolve() if requested is not None else OUTPUTS / "{}-{}".format(prefix, timestamp())
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing output: {}".format(output))
    output.mkdir(parents=True)
    return output


def write_manifest(output: Path, mode: str, records: List[Dict[str, object]]) -> None:
    manifest = {
        "mode": mode,
        "python": sys.version,
        "platform": sys.platform,
        "root": str(ROOT),
        "commands": records,
        "completed": True,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def reproduce_figures(requested: Optional[Path]) -> Path:
    output = new_output("figures", requested)
    records: List[Dict[str, object]] = []
    run(python("plot_math_motivation.py", "--output", output), records)
    run(
        python(
            "plot_theory_aligned_icc.py",
            "--locked-root", LOCKED,
            "--analysis-root", ANALYSIS,
            "--output", output,
        ),
        records,
    )
    write_manifest(output, "figures", records)
    return output


def compile_papers(requested: Optional[Path]) -> Path:
    output = new_output("paper", requested)
    records: List[Dict[str, object]] = []
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError("latexmk was not found. Install a TeX distribution with IEEEtran and BibTeX.")
    for source in ("ICC2027_submit.tex", "ICC2027_camera_ready.tex"):
        target = output / Path(source).stem
        target.mkdir()
        shutil.copy2(ROOT / source, target / source)
        shutil.copy2(ROOT / "refs.bib", target / "refs.bib")
        shutil.copytree(ROOT / "figures", target / "figures")
        locked_figures = target / "results" / "theory_aligned_v1_20260813" / "figures"
        locked_figures.mkdir(parents=True)
        for figure in ("neural_benchmark_theory.pdf", "wireless_stress_rate.pdf"):
            shutil.copy2(ANALYSIS / "figures" / figure, locked_figures / figure)
        run(
            [
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                source,
            ],
            records,
            cwd=target,
        )
    write_manifest(output, "paper", records)
    return output


def smoke(requested: Optional[Path]) -> Path:
    output = new_output("smoke", requested)
    records: List[Dict[str, object]] = []
    run(python("test_transition_model.py"), records)
    run(python("run_hard_neural_benchmark.py", "--smoke", "--output", output / "population"), records)
    run(python("run_trajectory_sampled_neural.py", "--phase", "smoke", "--output", output / "trajectory"), records)
    run(python("run_oracle_quality_diagnostic.py", "--smoke", "--output", output / "oracle"), records)
    run(python("run_neural_communication_stress.py", "--output", output / "stress", "--seeds", 500), records)
    write_manifest(output, "smoke", records)
    return output


def full(requested: Optional[Path]) -> Path:
    output = new_output("full", requested)
    records: List[Dict[str, object]] = []
    population = output / "population"
    analysis = output / "analysis"
    run(python("test_transition_model.py"), records)
    run(python("run_hard_neural_benchmark.py", "--output", population / "icc_neural_hard"), records)
    run(python("run_trajectory_sampled_neural.py", "--phase", "formal", "--output", population / "icc_trajectory_sampled"), records)
    run(
        python(
            "run_neural_communication_stress.py",
            "--output", analysis / "icc_neural_stress_12seed",
            "--seeds", *STRESS_SEEDS,
        ),
        records,
    )
    run(python("run_oracle_quality_diagnostic.py", "--output", analysis / "oracle_quality_v2"), records)
    figure_output = output / "figures"
    run(python("plot_math_motivation.py", "--output", figure_output), records)
    run(
        python(
            "plot_theory_aligned_icc.py",
            "--locked-root", population,
            "--analysis-root", analysis,
            "--output", figure_output,
        ),
        records,
    )
    write_manifest(output, "full", records)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "figures", "paper", "full"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.mode == "smoke":
        output = smoke(args.output)
    elif args.mode == "figures":
        output = reproduce_figures(args.output)
    elif args.mode == "paper":
        output = compile_papers(args.output)
    else:
        output = full(args.output)
    print(json.dumps({"status": "complete", "mode": args.mode, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
