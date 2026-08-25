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


def main() -> None:
    required = [
        "ICC2027_submit.tex",
        "ICC2027_submit.pdf",
        "ICC2027_camera_ready.tex",
        "ICC2027_camera_ready.pdf",
        "refs.bib",
        "figures/mathematical_motivation.pdf",
        "figures/system_model.pdf",
        "figures/algorithm_framework.pdf",
        "results/theory_aligned_v1_20260813/figures/neural_benchmark_theory.pdf",
        "results/theory_aligned_v1_20260813/figures/wireless_stress_rate.pdf",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            fail("Missing required artifact: {}".format(relative))

    paths = {
        "population": ROOT / "results/exogenous_ph_v1_20260813_tuned/icc_neural_hard/hard_neural_results.csv",
        "trajectory": ROOT / "results/exogenous_ph_v1_20260813_tuned/icc_trajectory_sampled/trajectory_results.csv",
        "trajectory_diagnostics": ROOT / "results/exogenous_ph_v1_20260813_tuned/icc_trajectory_sampled/trajectory_diagnostics.csv",
        "oracle": ROOT / "results/theory_aligned_v1_20260813/oracle_quality_v2/oracle_quality_raw.csv",
        "stress": ROOT / "results/theory_aligned_v1_20260813/icc_neural_stress_12seed/neural_stress_results.csv",
    }
    expected_rows = {
        "population": 1728,
        "trajectory": 420,
        "trajectory_diagnostics": 1800,
        "oracle": 576,
        "stress": 3456,
    }
    observed_rows = {name: row_count(path) for name, path in paths.items()}
    if observed_rows != expected_rows:
        fail("Unexpected CSV dimensions: {}".format(observed_rows))

    trajectory_protocol = json.loads(
        (ROOT / "results/exogenous_ph_v1_20260813_tuned/icc_trajectory_sampled/protocol.json").read_text(encoding="utf-8")
    )
    integrity = json.loads(
        (ROOT / "results/exogenous_ph_v1_20260813_tuned/icc_trajectory_sampled/formal_integrity.json").read_text(encoding="utf-8")
    )
    if integrity.get("status") != "pass" or not integrity.get("equal_locked_transition_budget"):
        fail("Trajectory integrity check did not pass")
    if trajectory_protocol.get("total_transitions_per_method_seed") != 245760:
        fail("Unexpected trajectory transition budget")
    if len(trajectory_protocol.get("seeds", [])) != 12:
        fail("Unexpected trajectory seed count")

    stress_summary = json.loads(
        (ROOT / "results/theory_aligned_v1_20260813/icc_neural_stress_12seed/summary.json").read_text(encoding="utf-8")
    )
    if len(stress_summary.get("seeds", [])) != 12:
        fail("Unexpected wireless-stress seed count")
    if len(stress_summary.get("configurations", [])) != 16:
        fail("Unexpected wireless-stress configuration count")
    if set(stress_summary.get("methods", [])) != {"QP+G", "noG", "PPM-3"}:
        fail("Unexpected wireless-stress method set")

    oracle_protocol = json.loads(
        (ROOT / "results/theory_aligned_v1_20260813/oracle_quality_v2/protocol.json").read_text(encoding="utf-8")
    )
    if oracle_protocol.get("failed_runs") != 0 or oracle_protocol.get("raw_rows") != 576:
        fail("Oracle-quality protocol is incomplete")

    for tex_name in ("ICC2027_submit.tex", "ICC2027_camera_ready.tex"):
        source = (ROOT / tex_name).read_text(encoding="utf-8")
        graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source)
        if len(graphics) != 5:
            fail("{} should include exactly five figures".format(tex_name))
        for graphic in graphics:
            if not (ROOT / graphic).is_file():
                fail("{} references missing figure {}".format(tex_name, graphic))

    checked_hashes = check_hashes()
    report = {
        "status": "pass",
        "csv_rows": observed_rows,
        "trajectory_seeds": len(trajectory_protocol["seeds"]),
        "stress_seeds": len(stress_summary["seeds"]),
        "stress_configurations": len(stress_summary["configurations"]),
        "verified_hashes": checked_hashes,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("release verification failed: {}".format(error), file=sys.stderr)
        raise
