#!/usr/bin/env python3
"""Fail-closed import of the verified collaborator Apple-MPS Binoculars run.

This module never reads or writes raw description text. It re-identifies opaque
output IDs through the private frozen map, checks them against the corrected-corpus
input manifest, and emits the internal accepted-score schema only after the reported
package inventory and model provenance are complete.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from .common import (
    B1ValidationError,
    atomic_write_csv,
    atomic_write_json,
    load_manifest,
    sha256_file,
    utc_now,
)
from .config import (
    CORPUS_SHA256,
    DETECTOR_CONFIGS,
    EXPECTED_AI,
    EXPECTED_HUMAN,
    EXPECTED_TOTAL,
    EXTERNAL_BINOCULARS,
    MODEL_SPECS,
    RESULT_COLUMNS,
    RUN_ID,
)


EXTERNAL_RESULT_COLUMNS = (
    "record_id",
    "detector_name",
    "detector_version",
    "observer_model",
    "observer_revision",
    "performer_model",
    "performer_revision",
    "precision",
    "max_length",
    "mode",
    "threshold",
    "threshold_source",
    "input_sha256",
    "data_manifest_sha256",
    "detector_config_sha256",
    "attempt_number",
    "started_at_utc",
    "raw_score",
    "ai_score",
    "predicted_label",
    "input_tokens",
    "used_input_tokens",
    "truncated",
    "success",
    "error_type",
    "error_fingerprint",
    "completed_at_utc",
    "elapsed_ms",
)

REIDENTIFICATION_COLUMNS = (
    "record_id",
    "entry_key",
    "pr_id",
    "entry_type",
    "variation",
    "model",
    "source_experiment_id",
    "text_sha256",
    "source_corpus_row",
    "scoring_order",
)

HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--handoff-manifest", type=Path, required=True)
    parser.add_argument("--reidentification-map", type=Path, required=True)
    parser.add_argument("--package-inventory", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    return parser.parse_args()


def _read_csv_file(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) == ():
            raise B1ValidationError(f"Missing CSV header: {path}")
        return list(reader)


def _read_zip_csv(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    try:
        raw = archive.open(member)
    except KeyError as error:
        raise B1ValidationError(f"Returned archive is missing {member}") from error
    with raw, io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
        reader = csv.DictReader(text)
        if tuple(reader.fieldnames or ()) != EXTERNAL_RESULT_COLUMNS:
            raise B1ValidationError(f"Unexpected schema in {member}")
        return list(reader)


def _read_zip_jsonl(archive: zipfile.ZipFile, member: str) -> list[dict[str, Any]]:
    try:
        raw = archive.open(member)
    except KeyError as error:
        raise B1ValidationError(f"Returned archive is missing {member}") from error
    records: list[dict[str, Any]] = []
    with raw, io.TextIOWrapper(raw, encoding="utf-8") as text:
        for line_number, line in enumerate(text, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise B1ValidationError(
                    f"Invalid JSONL in {member} at line {line_number}"
                ) from error
            if set(record) != set(EXTERNAL_RESULT_COLUMNS):
                raise B1ValidationError(
                    f"Unexpected attempt schema in {member} at line {line_number}"
                )
            records.append(record)
    return records


def _require_sha256(value: Any, field: str) -> str:
    normalized = str(value).strip().lower()
    if not HEX_SHA256.fullmatch(normalized):
        raise B1ValidationError(f"Provenance field {field} is not a SHA-256")
    return normalized


def _load_provenance(path: Path, package_inventory: Path) -> dict[str, Any]:
    if not path.is_file() or not package_inventory.is_file():
        raise B1ValidationError("Package inventory or provenance JSON is missing")
    package_lines = [
        line.strip().lower()
        for line in package_inventory.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for package in ("torch", "transformers"):
        if not any(
            line.startswith((f"{package}==", f"{package} @"))
            for line in package_lines
        ):
            raise B1ValidationError(
                f"Package inventory does not record the required {package} version"
            )
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if provenance.get("status") != "accepted_for_analysis":
        raise B1ValidationError("External Binoculars provenance is not accepted")
    fixed_fields = {
        "archive_sha256": EXTERNAL_BINOCULARS["archive_sha256"],
        "handoff_manifest_sha256": EXTERNAL_BINOCULARS["handoff_manifest_sha256"],
        "smoke_archive_sha256": EXTERNAL_BINOCULARS["smoke_archive_sha256"],
        "fork_repository": EXTERNAL_BINOCULARS["fork_repository"],
        "upstream_commit": EXTERNAL_BINOCULARS["upstream_commit"],
    }
    for field, expected in fixed_fields.items():
        if provenance.get(field) != expected:
            raise B1ValidationError(f"Provenance mismatch for {field}")
    for field in ("fork_commit", "runner_sha256", "detector_sha256"):
        if provenance.get(field) != EXTERNAL_BINOCULARS[field]:
            raise B1ValidationError(f"Provenance mismatch for {field}")
    expected_observer = MODEL_SPECS["falcon-observer"]["revision"]
    expected_performer = MODEL_SPECS["falcon-performer"]["revision"]
    if provenance.get("observer_revision") != expected_observer:
        raise B1ValidationError("Falcon observer snapshot is not the frozen revision")
    if provenance.get("performer_revision") != expected_performer:
        raise B1ValidationError("Falcon performer snapshot is not the frozen revision")
    if _require_sha256(
        provenance.get("package_inventory_sha256"), "package_inventory_sha256"
    ) != sha256_file(package_inventory):
        raise B1ValidationError("Package-inventory hash mismatch")
    if float(provenance.get("smoke_max_abs_diff", float("nan"))) != 0.0:
        raise B1ValidationError("Frozen smoke reproduction was not exact")
    hardware = provenance.get("hardware")
    if not isinstance(hardware, dict):
        raise B1ValidationError("Missing hardware provenance")
    required_hardware = {
        "model": "Mac16,11",
        "chip": "Apple M4 Pro",
        "memory_gb": 64,
        "macos": "26.5.2",
        "build": "25F84",
        "device": "mps",
    }
    for field, expected in required_hardware.items():
        if hardware.get(field) != expected:
            raise B1ValidationError(f"Hardware provenance mismatch for {field}")
    if not str(provenance.get("approved_by", "")).strip() or not str(
        provenance.get("approved_at_utc", "")
    ).strip():
        raise B1ValidationError("Provenance acceptance identity/time is missing")
    return provenance


def _bool(value: Any, *, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise B1ValidationError(f"Invalid boolean in {field}: {value!r}")


def _normalize_label(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized == "ai":
        return "AI"
    if normalized in {"human", "not ai", "not_ai"}:
        return "Not AI"
    raise B1ValidationError(f"Unexpected external label: {value!r}")


def _validate_handoff_manifest(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXTERNAL_BINOCULARS["handoff_manifest_sha256"]:
        raise B1ValidationError("Blinded handoff-manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "run_id": RUN_ID,
        "expected_rows": EXPECTED_TOTAL,
        "corpus_sha256": EXTERNAL_BINOCULARS["handoff_corpus_sha256"],
        "identifiers_are_opaque": True,
        "contains_study_labels": False,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise B1ValidationError(f"Blinded handoff manifest mismatch for {field}")
    return manifest


def _normalize_map_metadata(row: dict[str, str]) -> dict[str, str]:
    human = row["entry_type"].strip().lower() == "human"
    return {
        "description_id": row["entry_key"],
        "pr_id": row["pr_id"],
        "entry_type": row["entry_type"].strip().lower(),
        "model": "" if human else row["model"],
        "context_code": "" if human else row["variation"],
        "run": "human_baseline" if human else row["source_experiment_id"],
        "text_sha256": row["text_sha256"],
        "source_corpus_row": row["source_corpus_row"],
        "scoring_order": row["scoring_order"],
    }


def _validate_reidentification(
    path: Path, internal_manifest: list[dict[str, str]]
) -> dict[str, dict[str, str]]:
    if sha256_file(path) != EXTERNAL_BINOCULARS["reidentification_map_sha256"]:
        raise B1ValidationError("Frozen re-identification-map hash mismatch")
    rows = _read_csv_file(path)
    if not rows or tuple(rows[0]) != REIDENTIFICATION_COLUMNS:
        raise B1ValidationError("Unexpected re-identification-map schema")
    if len(rows) != EXPECTED_TOTAL:
        raise B1ValidationError("Unexpected re-identification-map row count")
    by_record = {row["record_id"]: row for row in rows}
    if len(by_record) != len(rows):
        raise B1ValidationError("Duplicate opaque ID in re-identification map")
    internal_by_id = {row["description_id"]: row for row in internal_manifest}
    if len(internal_by_id) != EXPECTED_TOTAL:
        raise B1ValidationError("Internal manifest IDs are not unique")
    projected_ids: set[str] = set()
    fields = (
        "description_id",
        "pr_id",
        "entry_type",
        "model",
        "context_code",
        "run",
        "text_sha256",
        "source_corpus_row",
        "scoring_order",
    )
    for record_id, raw in by_record.items():
        normalized = _normalize_map_metadata(raw)
        description_id = normalized["description_id"]
        if description_id in projected_ids or description_id not in internal_by_id:
            raise B1ValidationError(
                f"Invalid projected ID for opaque record {record_id}"
            )
        projected_ids.add(description_id)
        internal = internal_by_id[description_id]
        for field in fields:
            if str(normalized[field]) != str(internal[field]):
                raise B1ValidationError(
                    f"Re-identification mismatch for {description_id}/{field}"
                )
    if projected_ids != set(internal_by_id):
        raise B1ValidationError(
            "Re-identification map does not cover the internal manifest"
        )
    return by_record


def _validate_external_rows(
    rows: list[dict[str, str]],
    attempts: list[dict[str, Any]],
    reidentification: dict[str, dict[str, str]],
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_TOTAL or len(attempts) != EXPECTED_TOTAL:
        raise B1ValidationError("External result/attempt row count mismatch")
    result_by_id = {row["record_id"]: row for row in rows}
    attempt_by_id = {str(row["record_id"]): row for row in attempts}
    if len(result_by_id) != len(rows) or len(attempt_by_id) != len(attempts):
        raise B1ValidationError("Duplicate external result or attempt ID")
    if set(result_by_id) != set(reidentification) or set(attempt_by_id) != set(
        reidentification
    ):
        raise B1ValidationError("External rows do not exactly cover the blinded corpus")

    config = DETECTOR_CONFIGS["binoculars"]
    expected_constant = {
        "detector_name": "binoculars",
        "detector_version": "macos-mps",
        "observer_model": MODEL_SPECS[config["observer"]]["model_id"],
        "observer_revision": "cached",
        "performer_model": MODEL_SPECS[config["performer"]]["model_id"],
        "performer_revision": "cached",
        "precision": "bfloat16",
        "max_length": str(config["max_length"]),
        "mode": config["mode"],
        "threshold": str(config["threshold"]),
        "threshold_source": "official-low-fpr",
        "data_manifest_sha256": EXTERNAL_BINOCULARS["handoff_manifest_sha256"],
        "detector_config_sha256": EXTERNAL_BINOCULARS[
            "external_detector_config_sha256"
        ],
        "attempt_number": "1",
    }
    compare_fields = set(EXTERNAL_RESULT_COLUMNS).difference(
        {"error_type", "error_fingerprint"}
    )
    accepted: list[dict[str, Any]] = []
    counts = Counter()
    for record_id in sorted(
        result_by_id,
        key=lambda identifier: int(reidentification[identifier]["scoring_order"]),
    ):
        row = result_by_id[record_id]
        attempt = attempt_by_id[record_id]
        for field in compare_fields:
            if str(attempt[field]) != str(row[field]):
                raise B1ValidationError(
                    f"Attempt/result mismatch for {record_id}/{field}"
                )
        for field, expected in expected_constant.items():
            if str(row[field]) != str(expected):
                raise B1ValidationError(f"External configuration mismatch for {field}")
        if not _bool(row["success"], field="success"):
            raise B1ValidationError(f"External row is not successful: {record_id}")
        if (
            row["error_type"]
            or row["error_fingerprint"]
            or attempt["error_type"] not in {None, ""}
            or attempt["error_fingerprint"] not in {None, ""}
        ):
            raise B1ValidationError(f"Successful row carries an error: {record_id}")
        mapped = reidentification[record_id]
        if row["input_sha256"] != mapped["text_sha256"]:
            raise B1ValidationError(f"External input-hash mismatch: {record_id}")
        raw_score = float(row["raw_score"])
        ai_score = float(row["ai_score"])
        if not math.isfinite(raw_score) or not math.isfinite(ai_score):
            raise B1ValidationError(f"Non-finite external score: {record_id}")
        if not math.isclose(ai_score, -raw_score, rel_tol=0.0, abs_tol=1e-12):
            raise B1ValidationError(f"External score-direction mismatch: {record_id}")
        predicted_label = _normalize_label(row["predicted_label"])
        expected_label = "AI" if raw_score < config["threshold"] else "Not AI"
        if predicted_label != expected_label:
            raise B1ValidationError(f"External threshold-label mismatch: {record_id}")
        input_tokens = int(row["input_tokens"])
        used_input_tokens = int(row["used_input_tokens"])
        truncated = _bool(row["truncated"], field="truncated")
        if input_tokens < 0 or not 0 <= used_input_tokens <= config["max_length"]:
            raise B1ValidationError(f"Invalid token counts: {record_id}")
        if used_input_tokens > input_tokens or truncated != (
            input_tokens > used_input_tokens
        ):
            raise B1ValidationError(f"Inconsistent truncation record: {record_id}")
        counts[mapped["entry_type"].strip().lower()] += 1
        accepted.append(
            {
                "description_id": mapped["entry_key"],
                "pr_id": mapped["pr_id"],
                "entry_type": mapped["entry_type"].strip().lower(),
                "model": "" if mapped["entry_type"] == "human" else mapped["model"],
                "context_code": ""
                if mapped["entry_type"] == "human"
                else mapped["variation"],
                "run": "human_baseline"
                if mapped["entry_type"] == "human"
                else mapped["source_experiment_id"],
                "detector_name": "Binoculars",
                "detector_version": "macos-mps-source-verified",
                "sampling_model": "",
                "sampling_revision": "",
                "observer_model": row["observer_model"],
                "observer_revision": provenance["observer_revision"],
                "scoring_model": row["performer_model"],
                "scoring_revision": provenance["performer_revision"],
                "source_model_required": False,
                "faithful_implementation": True,
                "raw_score": raw_score,
                "calibrated_score": "",
                "ai_score": ai_score,
                "predicted_label": predicted_label,
                "threshold_source": config["threshold_source"],
                "input_tokens": input_tokens,
                "used_input_tokens": used_input_tokens,
                "truncated": truncated,
                "success": True,
                "error_type": "",
                "error_message": "",
                "started_at_utc": row["started_at_utc"],
                "completed_at_utc": row["completed_at_utc"],
                "elapsed_ms": int(row["elapsed_ms"]),
                "input_sha256": row["input_sha256"],
                "corpus_sha256": CORPUS_SHA256,
                "detector_config_sha256": row["detector_config_sha256"],
                "attempt_number": 1,
            }
        )
    if counts["ai"] != EXPECTED_AI or counts["human"] != EXPECTED_HUMAN:
        raise B1ValidationError(f"External class counts are wrong: {dict(counts)}")
    return accepted


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    archive_path = args.archive.resolve()
    if sha256_file(archive_path) != EXTERNAL_BINOCULARS["archive_sha256"]:
        raise B1ValidationError("Returned Binoculars archive hash mismatch")
    provenance = _load_provenance(
        args.provenance.resolve(), args.package_inventory.resolve()
    )
    handoff = _validate_handoff_manifest(args.handoff_manifest.resolve())
    internal_manifest, preparation = load_manifest(output_root)
    reidentification = _validate_reidentification(
        args.reidentification_map.resolve(), internal_manifest
    )
    with zipfile.ZipFile(archive_path) as archive:
        rows = _read_zip_csv(archive, "b1/results.csv")
        attempts = _read_zip_jsonl(archive, "b1/attempts.jsonl")
    accepted = _validate_external_rows(rows, attempts, reidentification, provenance)

    verified_dir = output_root / "verified"
    accepted_path = verified_dir / "external_binoculars_scores.csv"
    report_path = verified_dir / "external_binoculars_verification.json"
    if accepted_path.exists() or report_path.exists():
        raise B1ValidationError("Refusing to overwrite an external Binoculars import")
    atomic_write_csv(accepted_path, accepted, RESULT_COLUMNS)
    report = {
        "run_id": RUN_ID,
        "status": "accepted_external_detector_result",
        "verified_at_utc": utc_now(),
        "source_route": "collaborator-apple-mps",
        "archive_sha256": EXTERNAL_BINOCULARS["archive_sha256"],
        "handoff_manifest_sha256": EXTERNAL_BINOCULARS[
            "handoff_manifest_sha256"
        ],
        "handoff_corpus_sha256": handoff["corpus_sha256"],
        "internal_corpus_sha256": CORPUS_SHA256,
        "input_manifest_sha256": preparation["input_manifest_sha256"],
        "reidentification_map_sha256": EXTERNAL_BINOCULARS[
            "reidentification_map_sha256"
        ],
        "package_inventory_sha256": provenance["package_inventory_sha256"],
        "observer_revision": provenance["observer_revision"],
        "performer_revision": provenance["performer_revision"],
        "fork_commit": provenance["fork_commit"],
        "upstream_commit": provenance["upstream_commit"],
        "runner_sha256": provenance["runner_sha256"],
        "detector_sha256": provenance["detector_sha256"],
        "smoke_archive_sha256": provenance["smoke_archive_sha256"],
        "accepted_rows": len(accepted),
        "accepted_scores_sha256": sha256_file(accepted_path),
        "importer_sha256": sha256_file(Path(__file__)),
        "contains_raw_text": False,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
