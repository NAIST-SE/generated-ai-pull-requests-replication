"""Run one corrected detector with strict resumability and fail-closed completion."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .build_corpus import _atomic_write_csv, _atomic_write_json
from .detectors import (
    PermanentDetectionError,
    RetryableDetectionError,
    RobertaDetector,
    ZeroGPTDetector,
)
from .spec import (
    CONFIRMATORY_DETECTORS,
    DEFAULT_OUTPUT_DIR,
    DETECTOR_CONFIGS,
    EXPECTED_TOTAL_ROWS,
    RUN_ID,
    sha256_file,
    stable_json_sha256,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def verify_cached_roberta_assets(config: Dict[str, Any]) -> Dict[str, str]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as error:
        raise ImportError("huggingface_hub is required for pinned-cache preflight") from error
    verified: Dict[str, str] = {}
    for filename in ("config.json", "tokenizer.json", "pytorch_model.bin"):
        cached = try_to_load_from_cache(
            config["model"], filename, revision=config["revision"]
        )
        if not isinstance(cached, str) or not Path(cached).is_file():
            raise FileNotFoundError(
                f"Pinned cache asset is missing for {config['model']}@{config['revision']}: {filename}"
            )
        verified[filename] = cached
    return verified


def load_verified_corpus(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    corpus_path = run_dir / "corpus.csv"
    manifest_path = run_dir / "corpus_manifest.json"
    if not corpus_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("corpus.csv or corpus_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("run_id") != RUN_ID:
        raise ValueError("Corpus manifest run_id mismatch")
    if manifest.get("status") != "corpus_complete_and_verified":
        raise ValueError("Corpus manifest is not complete_and_verified")
    actual_hash = sha256_file(corpus_path)
    if actual_hash != manifest.get("corpus_sha256"):
        raise ValueError("Corpus CSV hash differs from its manifest")
    frame = pd.read_csv(corpus_path, dtype={"pr_id": str, "entry_key": str})
    if len(frame) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_ROWS} corpus rows, got {len(frame)}")
    if frame["entry_key"].duplicated().any():
        raise ValueError("Corpus contains duplicate entry keys")
    if set(frame["scoring_order"]) != set(range(1, len(frame) + 1)):
        raise ValueError("Corpus scoring_order is not a permutation of 1..N")
    return frame.sort_values("scoring_order", kind="stable"), manifest


def load_attempts(
    path: Path, *, corpus_sha256: str, config_sha256: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    attempts: List[Dict[str, Any]] = []
    successes: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return attempts, successes
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if record.get("corpus_sha256") != corpus_sha256:
                raise ValueError("Attempt log was created for a different corpus")
            if record.get("detector_config_sha256") != config_sha256:
                raise ValueError("Attempt log was created for a different detector configuration")
            attempts.append(record)
            if record.get("success") is True:
                key = record["entry_key"]
                if key in successes:
                    raise ValueError(f"Multiple successful attempts for entry {key}")
                successes[key] = record
    return attempts, successes


def append_attempt(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def base_attempt_record(
    entry: Dict[str, Any],
    *,
    detector: str,
    corpus_sha256: str,
    config_sha256: str,
    attempt_number: int,
) -> Dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "detector": detector,
        "entry_key": entry["entry_key"],
        "corpus_row": int(entry["corpus_row"]),
        "text_sha256": entry["text_sha256"],
        "qa_question_sha256": entry["qa_question_sha256"],
        "corpus_sha256": corpus_sha256,
        "detector_config_sha256": config_sha256,
        "attempt_number": attempt_number,
        "scored_at_utc": utc_now(),
    }


def next_attempt_number(attempts: Sequence[Dict[str, Any]], entry_key: str) -> int:
    return 1 + sum(1 for record in attempts if record.get("entry_key") == entry_key)


def failure_record(
    base: Dict[str, Any], error: Exception, *, retryable: bool
) -> Dict[str, Any]:
    return {
        **base,
        "success": False,
        "retryable": retryable,
        "error_type": type(error).__name__,
        "error": str(error),
        "ai_probability": None,
        "is_ai_generated": None,
        "raw_response": "",
    }


def run_zerogpt(
    detector: ZeroGPTDetector,
    pending: Sequence[Dict[str, Any]],
    *,
    attempts: List[Dict[str, Any]],
    attempts_path: Path,
    corpus_sha256: str,
    config_sha256: str,
    max_attempts: int,
    delay_seconds: float,
) -> None:
    for position, entry in enumerate(pending, start=1):
        existing_attempts = sum(
            1 for record in attempts if record.get("entry_key") == entry["entry_key"]
        )
        for attempt_offset in range(max_attempts):
            local_attempt = existing_attempts + attempt_offset + 1
            base = base_attempt_record(
                entry,
                detector=detector.name,
                corpus_sha256=corpus_sha256,
                config_sha256=config_sha256,
                attempt_number=local_attempt,
            )
            try:
                result = detector.score(entry)
                record = {
                    **base,
                    **result,
                    "success": True,
                    "retryable": False,
                    "error_type": None,
                    "error": None,
                }
            except RetryableDetectionError as error:
                record = failure_record(base, error, retryable=True)
            except PermanentDetectionError as error:
                record = failure_record(base, error, retryable=False)
            append_attempt(attempts_path, record)
            attempts.append(record)
            if record["success"]:
                break
            if not record["retryable"]:
                break
            if attempt_offset + 1 < max_attempts:
                time.sleep(min(60.0, delay_seconds * (2**attempt_offset)))
        print(
            f"[{detector.name}] {position}/{len(pending)} entry={entry['entry_key']}",
            flush=True,
        )
        if delay_seconds > 0:
            time.sleep(delay_seconds)


def _record_local_successes(
    detector: RobertaDetector,
    entries: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    *,
    attempts: List[Dict[str, Any]],
    attempts_path: Path,
    corpus_sha256: str,
    config_sha256: str,
) -> None:
    for entry, result in zip(entries, results):
        base = base_attempt_record(
            entry,
            detector=detector.name,
            corpus_sha256=corpus_sha256,
            config_sha256=config_sha256,
            attempt_number=next_attempt_number(attempts, entry["entry_key"]),
        )
        record = {
            **base,
            **result,
            "success": True,
            "retryable": False,
            "error_type": None,
            "error": None,
        }
        append_attempt(attempts_path, record)
        attempts.append(record)


def run_roberta(
    detector: RobertaDetector,
    pending: Sequence[Dict[str, Any]],
    *,
    attempts: List[Dict[str, Any]],
    attempts_path: Path,
    corpus_sha256: str,
    config_sha256: str,
    batch_size: int,
) -> None:
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        try:
            results = detector.score_batch(batch)
            if len(results) != len(batch):
                raise RuntimeError("RoBERTa result count differs from batch size")
            _record_local_successes(
                detector,
                batch,
                results,
                attempts=attempts,
                attempts_path=attempts_path,
                corpus_sha256=corpus_sha256,
                config_sha256=config_sha256,
            )
        except Exception as batch_error:
            # Isolate a bad row; never silently lose an entire batch.
            for entry in batch:
                try:
                    result = detector.score_batch([entry])
                    _record_local_successes(
                        detector,
                        [entry],
                        result,
                        attempts=attempts,
                        attempts_path=attempts_path,
                        corpus_sha256=corpus_sha256,
                        config_sha256=config_sha256,
                    )
                except Exception as error:
                    base = base_attempt_record(
                        entry,
                        detector=detector.name,
                        corpus_sha256=corpus_sha256,
                        config_sha256=config_sha256,
                        attempt_number=next_attempt_number(attempts, entry["entry_key"]),
                    )
                    record = failure_record(base, error, retryable=False)
                    record["batch_error"] = str(batch_error)
                    append_attempt(attempts_path, record)
                    attempts.append(record)
        print(
            f"[{detector.name}] {min(start + len(batch), len(pending))}/{len(pending)}",
            flush=True,
        )


def successful_attempts(attempts: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    successful: Dict[str, Dict[str, Any]] = {}
    for record in attempts:
        if record.get("success") is True:
            key = record["entry_key"]
            if key in successful:
                raise ValueError(f"Multiple successful results for {key}")
            successful[key] = record
    return successful


def finalise_detector(
    detector: str,
    frame: pd.DataFrame,
    attempts: Sequence[Dict[str, Any]],
    run_dir: Path,
    *,
    corpus_sha256: str,
    config: Dict[str, Any],
    runtime: Dict[str, Any],
    repo_root: Path,
) -> bool:
    successful = successful_attempts(attempts)
    ordered_keys = frame.sort_values("corpus_row")["entry_key"].tolist()
    missing = [key for key in ordered_keys if key not in successful]
    results_path = run_dir / "detectors" / detector / "results.csv"
    status_path = run_dir / "detectors" / detector / "status.json"
    result_frame = pd.DataFrame([successful[key] for key in ordered_keys if key in successful])
    _atomic_write_csv(result_frame, results_path)
    status = {
        "run_id": RUN_ID,
        "detector": detector,
        "status": "complete_and_verified" if not missing else "incomplete",
        "updated_at_utc": utc_now(),
        "corpus_sha256": corpus_sha256,
        "detector_config": config,
        "detector_config_sha256": stable_json_sha256(config),
        "runtime": runtime,
        "scoring_git_commit": git_commit(repo_root),
        "expected_rows": len(frame),
        "successful_rows": len(successful),
        "missing_rows": len(missing),
        "results_path": str(results_path.relative_to(repo_root)),
        "results_sha256": sha256_file(results_path),
    }
    _atomic_write_json(status, status_path)
    return not missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True, choices=sorted(DETECTOR_CONFIGS))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokens-file", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--api-delay-seconds", type=float, default=1.0)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    frame, corpus_manifest = load_verified_corpus(run_dir)
    corpus_sha256 = corpus_manifest["corpus_sha256"]
    detector_config = DETECTOR_CONFIGS[args.detector]
    runtime = {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "requests": package_version("requests"),
        "device": args.device if detector_config["kind"] == "roberta" else "remote API",
        "batch_size": args.batch_size if detector_config["kind"] == "roberta" else 1,
        "local_files_only": not args.allow_model_download,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", "") == "1",
    }
    if detector_config["kind"] == "roberta":
        try:
            import torch as runtime_torch
        except ImportError as error:
            raise ImportError("torch is required for RoBERTa preflight/scoring") from error
        runtime["cuda_available"] = runtime_torch.cuda.is_available()
        runtime["cuda_device_name"] = (
            runtime_torch.cuda.get_device_name(0)
            if runtime_torch.cuda.is_available()
            else ""
        )
        if args.device == "cuda" and not runtime["cuda_available"]:
            raise RuntimeError("CUDA was requested but is unavailable in the selected environment")
    effective_config = {"detector": detector_config, "runtime": runtime}
    config_sha256 = stable_json_sha256(effective_config)
    detector_dir = run_dir / "detectors" / args.detector
    attempts_path = detector_dir / "attempts.jsonl"
    attempts, successes = load_attempts(
        attempts_path,
        corpus_sha256=corpus_sha256,
        config_sha256=config_sha256,
    )
    pending_frame = frame[~frame["entry_key"].isin(successes)].copy()
    if args.limit > 0:
        pending_frame = pending_frame.head(args.limit)
    pending = pending_frame.to_dict("records")

    if detector_config["kind"] == "zerogpt":
        if args.tokens_file is None:
            raise ValueError("--tokens-file is required for ZeroGPT")
        detector_object: Any = ZeroGPTDetector(args.tokens_file)
        print(f"ZeroGPT preflight: {detector_object.token_count} non-empty tokens available")
    else:
        detector_object = None
        if args.preflight_only:
            cached_assets = verify_cached_roberta_assets(detector_config)
            print(
                "RoBERTa preflight: corpus, pinned configuration, tokenizer, and weights "
                f"verified ({len(cached_assets)} cache assets); model loading skipped"
            )

    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "detector": args.detector,
                "corpus_rows": len(frame),
                "already_successful": len(successes),
                "pending_selected": len(pending),
                "corpus_sha256": corpus_sha256,
                "effective_config_sha256": config_sha256,
            },
            sort_keys=True,
        )
    )
    if args.preflight_only:
        return 0

    if detector_config["kind"] == "zerogpt":
        run_zerogpt(
            detector_object,
            pending,
            attempts=attempts,
            attempts_path=attempts_path,
            corpus_sha256=corpus_sha256,
            config_sha256=config_sha256,
            max_attempts=args.max_attempts,
            delay_seconds=args.api_delay_seconds,
        )
    else:
        detector_object = RobertaDetector(
            args.detector,
            device=args.device,
            local_files_only=not args.allow_model_download,
        )
        run_roberta(
            detector_object,
            pending,
            attempts=attempts,
            attempts_path=attempts_path,
            corpus_sha256=corpus_sha256,
            config_sha256=config_sha256,
            batch_size=args.batch_size,
        )

    complete = finalise_detector(
        args.detector,
        frame,
        attempts,
        run_dir,
        corpus_sha256=corpus_sha256,
        config=effective_config,
        runtime=runtime,
        repo_root=repo_root,
    )
    if args.limit > 0 and not complete:
        print("Intentional partial run completed; use the same configuration to resume")
        return 3
    if not complete:
        print("Detector run remains incomplete; inspect status.json and resume", file=sys.stderr)
        return 2
    print(f"{args.detector} complete and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
