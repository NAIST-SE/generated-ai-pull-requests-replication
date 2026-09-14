#!/usr/bin/env python3
"""Run one pinned B1 detector with append-only attempts and strict resumability."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from .adapters import create_adapter
from .common import (
    B1PermanentInputError,
    B1ValidationError,
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
    load_manifest,
    load_verified_corpus,
    read_csv,
    read_jsonl,
    require_approved_host,
    runtime_metadata,
    sha256_file,
    source_bundle_hash,
    stable_json_sha256,
    utc_now,
    verify_official_repositories,
)
from .config import (
    CORPUS_SHA256,
    DETECTOR_CONFIGS,
    DETECTOR_NAMES,
    RESULT_COLUMNS,
    RUN_ID,
    SMOKE_COUNT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=DETECTOR_NAMES, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 2), default=1)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase3/analysis/outputs/tier_b/b1-modern-detectors-v1"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--confirm-full-run", action="store_true")
    return parser.parse_args()


def _attempts_by_id(
    attempts: list[dict[str, Any]], corpus_sha: str, config_sha: str
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    successes: dict[str, dict[str, Any]] = {}
    for record in attempts:
        if record.get("corpus_sha256") != corpus_sha:
            raise B1ValidationError("Attempt log belongs to a different corpus")
        if record.get("detector_config_sha256") != config_sha:
            raise B1ValidationError(
                "Attempt log belongs to a different detector configuration"
            )
        identifier = str(record.get("description_id", ""))
        grouped.setdefault(identifier, []).append(record)
        if record.get("success") is True:
            if identifier in successes:
                raise B1ValidationError(
                    f"Multiple successful attempts for {identifier}"
                )
            successes[identifier] = record
    return grouped, successes


def _result_record(
    corpus_row: dict[str, str],
    manifest_row: dict[str, str],
    detector_result: dict[str, Any],
    *,
    config_sha: str,
    attempt_number: int,
) -> dict[str, Any]:
    return {
        "description_id": manifest_row["description_id"],
        "pr_id": manifest_row["pr_id"],
        "entry_type": manifest_row["entry_type"],
        "model": manifest_row["model"],
        "context_code": manifest_row["context_code"],
        "run": manifest_row["run"],
        **detector_result,
        "success": True,
        "error_type": "",
        "error_message": "",
        "input_sha256": corpus_row["text_sha256"],
        "corpus_sha256": CORPUS_SHA256,
        "detector_config_sha256": config_sha,
        "attempt_number": attempt_number,
    }


def _failure_record(
    manifest_row: dict[str, str],
    error: Exception,
    *,
    detector: str,
    config_sha: str,
    attempt_number: int,
) -> dict[str, Any]:
    config = DETECTOR_CONFIGS[detector]
    now = utc_now()
    return {
        "description_id": manifest_row["description_id"],
        "pr_id": manifest_row["pr_id"],
        "entry_type": manifest_row["entry_type"],
        "model": manifest_row["model"],
        "context_code": manifest_row["context_code"],
        "run": manifest_row["run"],
        "detector_name": config["display_name"],
        "detector_version": "",
        "sampling_model": "",
        "sampling_revision": "",
        "observer_model": "",
        "observer_revision": "",
        "scoring_model": "",
        "scoring_revision": "",
        "source_model_required": config["source_model_required"],
        "faithful_implementation": config["faithful_implementation"],
        "raw_score": "",
        "calibrated_score": "",
        "ai_score": "",
        "predicted_label": "",
        "threshold_source": config["threshold_source"],
        "input_tokens": "",
        "used_input_tokens": "",
        "truncated": "",
        "success": False,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "started_at_utc": now,
        "completed_at_utc": now,
        "elapsed_ms": "",
        "input_sha256": manifest_row["text_sha256"],
        "corpus_sha256": CORPUS_SHA256,
        "detector_config_sha256": config_sha,
        "attempt_number": attempt_number,
        "traceback_sha256": stable_json_sha256(traceback.format_exc()),
    }


def _check_full_gate(
    output_root: Path, detector: str, detector_config_sha: str
) -> None:
    gate_path = output_root / "execution_gates" / f"{detector}.json"
    if not gate_path.is_file():
        raise B1ValidationError(
            f"Full run gate is missing: approve the two smoke repeats first ({gate_path})"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "approved" or gate.get("detector") != detector:
        raise B1ValidationError("Full run gate is not approved for this detector")
    if gate.get("detector_config_sha256") != detector_config_sha:
        raise B1ValidationError(
            "Full run gate was approved for a different detector configuration"
        )


def main() -> int:
    args = parse_args()
    require_approved_host(require_cuda=True)
    if args.max_attempts < 1:
        raise B1ValidationError("--max-attempts must be positive")
    if args.mode == "full" and not args.confirm_full_run:
        raise B1ValidationError(
            "Full scoring requires the explicit --confirm-full-run flag"
        )

    repo_root = args.repo_root.resolve()
    runtime_root = args.runtime_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    manifest_rows, preparation = load_manifest(output_root)
    corpus_rows, _ = load_verified_corpus(repo_root)
    corpus_by_id = {row["entry_key"]: row for row in corpus_rows}
    manifest_by_id = {row["description_id"]: row for row in manifest_rows}
    if set(corpus_by_id) != set(manifest_by_id):
        raise B1ValidationError(
            "Input manifest and corrected corpus identifiers differ"
        )

    if args.mode == "smoke":
        smoke_path = output_root / "smoke_manifest.csv"
        if sha256_file(smoke_path) != preparation["smoke_manifest_sha256"]:
            raise B1ValidationError("Smoke manifest hash mismatch")
        selected_ids = [row["description_id"] for row in read_csv(smoke_path)]
        if len(selected_ids) != SMOKE_COUNT:
            raise B1ValidationError("Smoke manifest does not contain ten cases")
        run_dir = output_root / "smoke" / args.detector / f"repeat-{args.repeat}"
    else:
        selected_ids = [row["description_id"] for row in manifest_rows]
        run_dir = output_root / "full" / args.detector

    official_commits = verify_official_repositories(runtime_root)
    runtime = runtime_metadata(repo_root)
    runtime_fingerprint = {
        "packages": runtime["packages"],
        "torch_cuda_version": runtime.get("torch_cuda_version"),
        "cuda_device_count": runtime.get("cuda_device_count"),
        "cuda_devices": runtime.get("cuda_devices"),
        "requested_device": args.device,
    }
    package_dir = Path(__file__).resolve().parent
    detector_config = {
        "run_id": RUN_ID,
        "detector": args.detector,
        "configuration": DETECTOR_CONFIGS[args.detector],
        "official_repository_commit": official_commits[
            DETECTOR_CONFIGS[args.detector]["official_repository"]
        ],
        "source_bundle_sha256": source_bundle_hash(package_dir),
        "corpus_sha256": CORPUS_SHA256,
        "input_manifest_sha256": preparation["input_manifest_sha256"],
        "runtime_fingerprint": runtime_fingerprint,
    }
    detector_config_sha = stable_json_sha256(detector_config)
    if args.mode == "full":
        _check_full_gate(output_root, args.detector, detector_config_sha)

    run_config = {
        **detector_config,
        "detector_config_sha256": detector_config_sha,
        "mode": args.mode,
        "repeat": args.repeat if args.mode == "smoke" else None,
        "device": args.device,
        "runtime": runtime,
    }
    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        prior = json.loads(run_config_path.read_text(encoding="utf-8"))
        comparable_prior = {key: prior.get(key) for key in detector_config}
        if comparable_prior != detector_config:
            raise B1ValidationError(
                "Existing run directory uses a different frozen configuration"
            )
    else:
        atomic_write_json(run_config_path, run_config)

    attempts_path = run_dir / "attempts.jsonl"
    attempts = read_jsonl(attempts_path)
    grouped, successes = _attempts_by_id(attempts, CORPUS_SHA256, detector_config_sha)
    pending = [identifier for identifier in selected_ids if identifier not in successes]
    eligible = [
        identifier
        for identifier in pending
        if len(grouped.get(identifier, [])) < args.max_attempts
    ]
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "detector": args.detector,
                "mode": args.mode,
                "selected": len(selected_ids),
                "already_successful": len(successes),
                "eligible_pending": len(eligible),
                "detector_config_sha256": detector_config_sha,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    adapter = (
        create_adapter(args.detector, runtime_root, device=args.device)
        if eligible
        else None
    )
    fatal_error: Exception | None = None
    for position, identifier in enumerate(eligible, start=1):
        corpus_row = corpus_by_id[identifier]
        manifest = manifest_by_id[identifier]
        attempt_number = len(grouped.get(identifier, [])) + 1
        try:
            if corpus_row["text_sha256"] != manifest["text_sha256"]:
                raise B1ValidationError("Manifest/corpus text hash mismatch")
            result = adapter.score(corpus_row["text"])
            record = _result_record(
                corpus_row,
                manifest,
                result,
                config_sha=detector_config_sha,
                attempt_number=attempt_number,
            )
        except Exception as error:
            record = _failure_record(
                manifest,
                error,
                detector=args.detector,
                config_sha=detector_config_sha,
                attempt_number=attempt_number,
            )
            if not isinstance(error, B1PermanentInputError):
                fatal_error = error
        append_jsonl(attempts_path, record)
        attempts.append(record)
        grouped.setdefault(identifier, []).append(record)
        if record["success"]:
            successes[identifier] = record
        print(
            f"[{args.detector}] {position}/{len(eligible)} id={identifier} "
            f"success={record['success']}",
            flush=True,
        )
        if fatal_error is not None:
            print(
                f"Fatal detector error recorded; stopping safely: {fatal_error}",
                flush=True,
            )
            break

    ordered_successes = [successes[item] for item in selected_ids if item in successes]
    results_path = run_dir / "results.csv"
    atomic_write_csv(results_path, ordered_successes, RESULT_COLUMNS)
    attempted_ids = {record["description_id"] for record in attempts}
    missing_ids = [
        identifier for identifier in selected_ids if identifier not in successes
    ]
    exhausted_ids = [
        identifier
        for identifier in missing_ids
        if len(grouped.get(identifier, [])) >= args.max_attempts
    ]
    status_name = (
        "complete_and_verified"
        if not missing_ids
        else "complete_with_failures"
        if len(attempted_ids.intersection(selected_ids)) == len(selected_ids)
        and set(missing_ids) == set(exhausted_ids)
        else "incomplete"
    )
    status = {
        "run_id": RUN_ID,
        "detector": args.detector,
        "mode": args.mode,
        "repeat": args.repeat if args.mode == "smoke" else None,
        "status": status_name,
        "updated_at_utc": utc_now(),
        "selected_rows": len(selected_ids),
        "attempted_unique_rows": len(attempted_ids.intersection(selected_ids)),
        "successful_rows": len(ordered_successes),
        "missing_rows": len(missing_ids),
        "exhausted_failure_rows": len(exhausted_ids),
        "detector_config_sha256": detector_config_sha,
        "corpus_sha256": CORPUS_SHA256,
        "attempts_path": str(attempts_path),
        "attempts_sha256": sha256_file(attempts_path) if attempts_path.exists() else "",
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path),
    }
    atomic_write_json(run_dir / "status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    if fatal_error is not None or status_name == "incomplete":
        return 2
    if status_name == "complete_with_failures":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
