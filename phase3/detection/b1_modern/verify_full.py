#!/usr/bin/env python3
"""Independently verify full B1 attempt coverage and accepted score tables."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .common import (
    B1ValidationError,
    atomic_write_csv,
    atomic_write_json,
    load_manifest,
    read_csv,
    read_jsonl,
    sha256_file,
    utc_now,
)
from .config import (
    ANALYSIS_DETECTOR_NAMES,
    CORPUS_SHA256,
    EXPECTED_TOTAL,
    RESULT_COLUMNS,
    RUN_ID,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase3/analysis/outputs/tier_b/b1-modern-detectors-v1"),
    )
    parser.add_argument(
        "--use-external-binoculars",
        action="store_true",
        help="Use the separately imported and provenance-accepted Apple-MPS output",
    )
    return parser.parse_args()


def verify_detector(
    detector: str,
    output_root: Path,
    manifest_by_id: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    run_dir = output_root / "full" / detector
    status_path = run_dir / "status.json"
    attempts_path = run_dir / "attempts.jsonl"
    results_path = run_dir / "results.csv"
    if not all(path.is_file() for path in (status_path, attempts_path, results_path)):
        raise B1ValidationError(f"Full {detector} run is missing required files")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    attempts = read_jsonl(attempts_path)
    results = read_csv(results_path)
    if status.get("status") not in {"complete_and_verified", "complete_with_failures"}:
        raise B1ValidationError(f"Full {detector} status is incomplete")
    if sha256_file(attempts_path) != status.get("attempts_sha256"):
        raise B1ValidationError(f"Full {detector} attempt hash mismatch")
    if sha256_file(results_path) != status.get("results_sha256"):
        raise B1ValidationError(f"Full {detector} result hash mismatch")
    attempted_ids = {str(record.get("description_id", "")) for record in attempts}
    if attempted_ids != set(manifest_by_id):
        missing = len(set(manifest_by_id).difference(attempted_ids))
        extra = len(attempted_ids.difference(manifest_by_id))
        raise B1ValidationError(
            f"Full {detector} attempt coverage mismatch: missing={missing}, extra={extra}"
        )
    successful_attempts = [
        record for record in attempts if record.get("success") is True
    ]
    success_counts = Counter(record["description_id"] for record in successful_attempts)
    if any(count != 1 for count in success_counts.values()):
        raise B1ValidationError(
            f"Full {detector} contains duplicate successful attempts"
        )
    if len(results) != len(successful_attempts):
        raise B1ValidationError(f"Full {detector} accepted-result count mismatch")
    for row in results:
        identifier = row["description_id"]
        if identifier not in manifest_by_id:
            raise B1ValidationError(f"Unknown accepted description ID: {identifier}")
        if row["input_sha256"] != manifest_by_id[identifier]["text_sha256"]:
            raise B1ValidationError(f"Accepted input hash mismatch: {identifier}")
        if row["corpus_sha256"] != CORPUS_SHA256:
            raise B1ValidationError(f"Accepted corpus hash mismatch: {identifier}")
        if not math.isfinite(float(row["raw_score"])) or not math.isfinite(
            float(row["ai_score"])
        ):
            raise B1ValidationError(f"Non-finite accepted score: {identifier}")
    terminal_failures = set(manifest_by_id).difference(success_counts)
    report = {
        "detector": detector,
        "status": status["status"],
        "attempted_rows": len(attempted_ids),
        "successful_rows": len(results),
        "terminal_failure_rows": len(terminal_failures),
        "attempts_sha256": sha256_file(attempts_path),
        "results_sha256": sha256_file(results_path),
        "detector_config_sha256": status["detector_config_sha256"],
    }
    return report, results


def verify_external_binoculars(
    output_root: Path,
    manifest_by_id: dict[str, dict[str, str]],
    input_manifest_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    verified_dir = output_root / "verified"
    report_path = verified_dir / "external_binoculars_verification.json"
    results_path = verified_dir / "external_binoculars_scores.csv"
    if not report_path.is_file() or not results_path.is_file():
        raise B1ValidationError(
            "The provenance-accepted external Binoculars import is missing"
        )
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    if source_report.get("status") != "accepted_external_detector_result":
        raise B1ValidationError("External Binoculars import is not accepted")
    if source_report.get("input_manifest_sha256") != input_manifest_sha256:
        raise B1ValidationError("External Binoculars input-manifest hash mismatch")
    if sha256_file(results_path) != source_report.get("accepted_scores_sha256"):
        raise B1ValidationError("External Binoculars accepted-score hash mismatch")
    results = read_csv(results_path)
    if not results or tuple(results[0]) != RESULT_COLUMNS:
        raise B1ValidationError("External Binoculars accepted-score schema mismatch")
    if len(results) != len(manifest_by_id):
        raise B1ValidationError("External Binoculars accepted-score row count mismatch")
    result_by_id = {row["description_id"]: row for row in results}
    if len(result_by_id) != len(results) or set(result_by_id) != set(manifest_by_id):
        raise B1ValidationError("External Binoculars accepted-score coverage mismatch")
    for identifier, row in result_by_id.items():
        expected = manifest_by_id[identifier]
        if row["detector_name"] != "Binoculars":
            raise B1ValidationError("Unexpected detector in external accepted scores")
        if row["input_sha256"] != expected["text_sha256"]:
            raise B1ValidationError(
                f"External accepted input hash mismatch: {identifier}"
            )
        if row["corpus_sha256"] != CORPUS_SHA256:
            raise B1ValidationError(
                f"External accepted corpus hash mismatch: {identifier}"
            )
        for field in ("pr_id", "entry_type", "model", "context_code", "run"):
            if row[field] != expected[field]:
                raise B1ValidationError(
                    f"External accepted metadata mismatch: {identifier}/{field}"
                )
        if row["predicted_label"] not in {"AI", "Not AI"}:
            raise B1ValidationError(f"Invalid external accepted label: {identifier}")
        if not math.isfinite(float(row["raw_score"])) or not math.isfinite(
            float(row["ai_score"])
        ):
            raise B1ValidationError(f"Non-finite external accepted score: {identifier}")
    report = {
        "detector": "binoculars",
        "status": "complete_and_verified",
        "source_route": source_report["source_route"],
        "attempted_rows": len(results),
        "successful_rows": len(results),
        "terminal_failure_rows": 0,
        "results_sha256": sha256_file(results_path),
        "detector_config_sha256": results[0]["detector_config_sha256"],
        "source_verification_sha256": sha256_file(report_path),
    }
    return report, results


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    manifest, preparation = load_manifest(output_root)
    manifest_by_id = {row["description_id"]: row for row in manifest}
    if len(manifest_by_id) != EXPECTED_TOTAL:
        raise B1ValidationError("B1 manifest identifiers are not unique")
    reports = []
    accepted: list[dict[str, str]] = []
    successful_sets: dict[str, set[str]] = {}
    for detector in ANALYSIS_DETECTOR_NAMES:
        if detector == "binoculars" and args.use_external_binoculars:
            report, rows = verify_external_binoculars(
                output_root,
                manifest_by_id,
                preparation["input_manifest_sha256"],
            )
        else:
            report, rows = verify_detector(detector, output_root, manifest_by_id)
        reports.append(report)
        accepted.extend(rows)
        successful_sets[detector] = {row["description_id"] for row in rows}
    common_ids = set.intersection(*successful_sets.values())
    accepted_path = output_root / "verified" / "accepted_scores.csv"
    common_path = output_root / "verified" / "strict_common_scores.csv"
    atomic_write_csv(accepted_path, accepted, RESULT_COLUMNS)
    atomic_write_csv(
        common_path,
        [row for row in accepted if row["description_id"] in common_ids],
        RESULT_COLUMNS,
    )
    report = {
        "run_id": RUN_ID,
        "status": "verified",
        "verified_at_utc": utc_now(),
        "corpus_sha256": CORPUS_SHA256,
        "input_manifest_sha256": preparation["input_manifest_sha256"],
        "detectors": reports,
        "strict_common_description_count": len(common_ids),
        "accepted_scores_sha256": sha256_file(accepted_path),
        "strict_common_scores_sha256": sha256_file(common_path),
    }
    atomic_write_json(output_root / "verified" / "verification.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
