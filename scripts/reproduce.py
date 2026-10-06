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
RUN = EXPERIMENTS / "run"
PLOT = EXPERIMENTS / "plot"
TESTS = EXPERIMENTS / "tests"
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
QUEUE_SEEDS = tuple(range(800, 812))


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


def python(script: Path, *arguments: object) -> List[str]:
    return [sys.executable, str(script), *[str(argument) for argument in arguments]]


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
    run(python(PLOT / "plot_math_motivation.py", "--output", output), records)
    run(python(PLOT / "plot_queue_system_model.py"), records)
    run(python(PLOT / "plot_queue_algorithm_framework.py"), records)
    run(python(PLOT / "plot_queue_nominal_benchmark.py", "--input", RESULTS / "queue_nominal" / "queue_nominal_results.csv", "--output", output), records)
    run(python(PLOT / "plot_queue_aware_results.py", "--input", RESULTS / "queue_aware" / "queue_results.csv", "--output", output), records)
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
        shutil.copy2(ROOT / "ICC2027_submit.tex", target / "ICC2027_submit.tex")
        shutil.copy2(ROOT / "ICC2027_camera_ready.tex", target / "ICC2027_camera_ready.tex")
        shutil.copy2(ROOT / "refs.bib", target / "refs.bib")
        shutil.copytree(ROOT / "figures", target / "figures")
        locked_figures = target / "results"
        locked_figures.mkdir()
        for figure in ("queue_nominal_benchmark.pdf", "queue_load_results.pdf"):
            shutil.copy2(RESULTS / figure, locked_figures / figure)
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
    run(python(TESTS / "test_queue_transition_model.py"), records)
    run(python(RUN / "run_queue_aware_stress.py", "--smoke", "--output", output / "queue_aware", "--seeds", 800), records)
    write_manifest(output, "smoke", records)
    return output


def full(requested: Optional[Path]) -> Path:
    output = new_output("full", requested)
    records: List[Dict[str, object]] = []
    nominal = output / "queue_nominal"
    load_stress = output / "queue_aware"
    run(python(TESTS / "test_queue_transition_model.py"), records)
    run(python(RUN / "run_queue_nominal_benchmark.py", "--output", nominal, "--seeds", *QUEUE_SEEDS), records)
    run(python(RUN / "run_queue_aware_stress.py", "--output", load_stress, "--seeds", *QUEUE_SEEDS), records)
    run(python(PLOT / "plot_math_motivation.py", "--output", output), records)
    run(python(PLOT / "plot_queue_nominal_benchmark.py", "--input", nominal / "queue_nominal_results.csv", "--output", output), records)
    run(python(PLOT / "plot_queue_aware_results.py", "--input", load_stress / "queue_results.csv", "--output", output), records)
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
