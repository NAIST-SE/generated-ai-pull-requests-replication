#!/usr/bin/env python3
"""Verify two independent ten-text smoke repeats for one detector."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .common import B1ValidationError, atomic_write_json, read_csv, sha256_file, utc_now
from .config import (
    DETECTOR_CONFIGS,
    DETECTOR_NAMES,
    EXPECTED_TOTAL,
    RUN_ID,
    SMOKE_ABS_TOLERANCE,
    SMOKE_COUNT,
    SMOKE_REL_TOLERANCE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=DETECTOR_NAMES, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase3/analysis/outputs/tier_b/b1-modern-detectors-v1"),
    )
    return parser.parse_args()


def _load_run(path: Path) -> tuple[dict, list[dict[str, str]]]:
    status_path = path / "status.json"
    results_path = path / "results.csv"
    if not status_path.is_file() or not results_path.is_file():
        raise B1ValidationError(f"Smoke run is missing status/results under {path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    rows = read_csv(results_path)
    if status.get("status") != "complete_and_verified":
        raise B1ValidationError(f"Smoke run is not complete_and_verified: {path}")
    if len(rows) != SMOKE_COUNT or status.get("successful_rows") != SMOKE_COUNT:
        raise B1ValidationError(
            f"Smoke run does not contain {SMOKE_COUNT} successes: {path}"
        )
    if sha256_file(results_path) != status.get("results_sha256"):
        raise B1ValidationError(f"Smoke results hash mismatch: {path}")
    return status, rows


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    detector_root = output_root / "smoke" / args.detector
    first_status, first_rows = _load_run(detector_root / "repeat-1")
    second_status, second_rows = _load_run(detector_root / "repeat-2")
    if (
        first_status["detector_config_sha256"]
        != second_status["detector_config_sha256"]
    ):
        raise B1ValidationError("Smoke repeats used different detector configurations")
    first = {row["description_id"]: row for row in first_rows}
    second = {row["description_id"]: row for row in second_rows}
    if set(first) != set(second) or len(first) != SMOKE_COUNT:
        raise B1ValidationError("Smoke repeat identifiers differ")

    comparisons = []
    all_close = True
    for identifier in first:
        left = first[identifier]
        right = second[identifier]
        left_score = float(left["raw_score"])
        right_score = float(right["raw_score"])
        close = math.isclose(
            left_score,
            right_score,
            abs_tol=SMOKE_ABS_TOLERANCE,
            rel_tol=SMOKE_REL_TOLERANCE,
        )
        tokens_match = (
            left["input_tokens"] == right["input_tokens"]
            and left["used_input_tokens"] == right["used_input_tokens"]
            and left["truncated"] == right["truncated"]
        )
        all_close = all_close and close and tokens_match
        comparisons.append(
            {
                "description_id": identifier,
                "input_sha256": left["input_sha256"],
                "repeat_1_raw_score": left_score,
                "repeat_2_raw_score": right_score,
                "absolute_difference": abs(left_score - right_score),
                "within_tolerance": close,
                "token_diagnostics_identical": tokens_match,
            }
        )

    elapsed_1 = sum(float(row["elapsed_ms"]) for row in first_rows) / 1000.0
    elapsed_2 = sum(float(row["elapsed_ms"]) for row in second_rows) / 1000.0
    mean_seconds_per_text = (elapsed_1 + elapsed_2) / (2 * SMOKE_COUNT)
    report = {
        "run_id": RUN_ID,
        "detector": args.detector,
        "status": "passed" if all_close else "failed",
        "verified_at_utc": utc_now(),
        "detector_config_sha256": first_status["detector_config_sha256"],
        "score_direction": DETECTOR_CONFIGS[args.detector]["score_direction"],
        "threshold_source": DETECTOR_CONFIGS[args.detector]["threshold_source"],
        "repeatability_tolerance": {
            "absolute": SMOKE_ABS_TOLERANCE,
            "relative": SMOKE_REL_TOLERANCE,
        },
        "all_ten_scored": len(first) == SMOKE_COUNT,
        "repeatability_passed": all_close,
        "reference_code_path": (
            "wrapper calls the pinned official Binoculars compute_score implementation directly"
            if args.detector == "binoculars"
            else "wrapper calls the pinned official get_sampling_discrepancy_analytic criterion directly"
        ),
        "manual_reference_parity": "must be confirmed in the execution gate before full scoring",
        "repeat_1_scoring_seconds": elapsed_1,
        "repeat_2_scoring_seconds": elapsed_2,
        "mean_seconds_per_text": mean_seconds_per_text,
        "estimated_full_scoring_hours": mean_seconds_per_text * EXPECTED_TOTAL / 3600.0,
        "comparisons": comparisons,
    }
    report_path = detector_root / "verification.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all_close else 2


if __name__ == "__main__":
    raise SystemExit(main())
