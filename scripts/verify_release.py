"""Verify the locked ICC release without rerunning training."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Set


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".bib", ".cff", ".csv", ".json", ".md", ".py", ".tex", ".txt", ".yml", ".yaml"}
TEXT_NAMES = {".gitattributes", ".gitignore", "requirements.txt"}


def fail(message: str) -> None:
    raise AssertionError(message)


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def column_values(path: Path, column: str) -> Set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[column] for row in csv.DictReader(handle)}


def sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES:
        data = data.replace(b"\r\n", b"\n")
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def check_hashes() -> int:
    manifest = ROOT / "SHA256SUMS"
    if not manifest.exists():
        return 0
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = ROOT / relative
        if not target.is_file():
            fail("Missing hashed file: {}".format(relative))
        if sha256(target) != expected:
            fail("SHA-256 mismatch: {}".format(relative))
        checked += 1
    return checked


def check_dependency_pins() -> int:
    recorded = json.loads(
        (ROOT / "results/package_versions.json").read_text(encoding="utf-8")
    )
    requirements = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            requirements[name] = version
    expected = {
        "numpy": recorded["numpy"],
        "scipy": recorded["scipy"],
        "pandas": recorded["pandas"],
        "torch": recorded["torch"].split("+", 1)[0],
        "matplotlib": recorded["matplotlib"],
    }
    if requirements != expected:
        fail("requirements.txt does not match the locked package versions")
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    conda_pins = {
        "python": recorded["python"].split(" ", 1)[0],
        "numpy": recorded["numpy"],
        "scipy": recorded["scipy"],
        "pandas": recorded["pandas"],
        "pytorch": recorded["torch"].split("+", 1)[0],
        "matplotlib": recorded["matplotlib"],
    }
    for name, version in conda_pins.items():
        if "- {}={}".format(name, version) not in environment:
            fail("environment.yml is missing the locked {}={} pin".format(name, version))
    if "- cpuonly" not in environment:
        fail("environment.yml must retain the CPU-only PyTorch constraint")
    return len(conda_pins)


def main() -> None:
    required = [
        "ICC2027_submit.tex",
        "ICC2027_submit.pdf",
        "ICC2027_camera_ready.tex",
        "ICC2027_camera_ready.pdf",
        "refs.bib",
        "requirements.txt",
        "environment.yml",
        "figures/mathematical_motivation.pdf",
        "figures/system_model_queue.pdf",
        "figures/algorithm_framework_queue.pdf",
        "results/queue_nominal_benchmark.pdf",
        "results/queue_load_results.pdf",
        "results/package_versions.json",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            fail("Missing required artifact: {}".format(relative))

    paths = {
        "queue_nominal": ROOT / "results/queue_nominal/queue_nominal_results.csv",
        "queue_nominal_diagnostics": ROOT / "results/queue_nominal/queue_nominal_diagnostics.csv",
        "queue_load": ROOT / "results/queue_aware/queue_results.csv",
        "queue_load_diagnostics": ROOT / "results/queue_aware/queue_diagnostics.csv",
    }
    expected_rows = {
        "queue_nominal": 504,
        "queue_nominal_diagnostics": 2160,
        "queue_load": 648,
        "queue_load_diagnostics": 2700,
    }
    observed_rows = {name: row_count(path) for name, path in paths.items()}
    if observed_rows != expected_rows:
        fail("Unexpected CSV dimensions: {}".format(observed_rows))

    nominal = json.loads((ROOT / "results/queue_nominal/summary.json").read_text(encoding="utf-8"))
    load = json.loads((ROOT / "results/queue_aware/summary.json").read_text(encoding="utf-8"))
    if nominal.get("failure_count") != 0 or nominal.get("completed_seed_count") != 12:
        fail("Nominal queue-aware benchmark is incomplete")
    if load.get("failure_count") != 0 or load.get("completed_config_seed_pairs") != 36:
        fail("Queue-load stress is incomplete")
    if not nominal.get("protocol_locked_before_confirmatory_execution"):
        fail("Nominal protocol was not locked")
    if not load.get("protocol_locked_before_confirmatory_execution"):
        fail("Load-stress protocol was not locked")

    submit_source = (ROOT / "ICC2027_submit.tex").read_text(encoding="utf-8")
    for tex_name in ("ICC2027_submit.tex", "ICC2027_camera_ready.tex"):
        source = submit_source if tex_name == "ICC2027_submit.tex" else submit_source
        graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source)
        if len(graphics) != 5:
            fail("{} should include exactly five figures".format(tex_name))
        for graphic in graphics:
            if not (ROOT / graphic).is_file():
                fail("{} references missing figure {}".format(tex_name, graphic))

    dependency_pins = check_dependency_pins()
    checked_hashes = check_hashes()
    report = {
        "status": "pass",
        "csv_rows": observed_rows,
        "nominal_seeds": len(nominal["seeds"]),
        "load_stress_seeds": len(load["seeds"]),
        "load_stress_configurations": len(load["configurations"]),
        "verified_dependency_pins": dependency_pins,
        "verified_hashes": checked_hashes,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("release verification failed: {}".format(error), file=sys.stderr)
        raise
