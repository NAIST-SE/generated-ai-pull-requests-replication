"""Merge and verify the three corrected detector outputs into a legacy-compatible CSV."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .build_corpus import _atomic_write_csv, _atomic_write_json
from .run_detector import load_verified_corpus
from .spec import (
    CONFIRMATORY_DETECTORS,
    DEFAULT_OUTPUT_DIR,
    DETECTOR_CONFIGS,
    EXPECTED_AI_ROWS,
    EXPECTED_HUMAN_ROWS,
    EXPECTED_TOTAL_ROWS,
    RUN_ID,
    sha256_file,
    sha256_text,
)


def parse_strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"Not a strict boolean value: {value!r}")


def load_verified_detector(run_dir: Path, detector: str, corpus_sha256: str) -> pd.DataFrame:
    detector_dir = run_dir / "detectors" / detector
    status_path = detector_dir / "status.json"
    results_path = detector_dir / "results.csv"
    if not status_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(f"Missing completed result for {detector}")
    status = json.loads(status_path.read_text())
    if status.get("status") != "complete_and_verified":
        raise ValueError(f"{detector} status is not complete_and_verified")
    if status.get("corpus_sha256") != corpus_sha256:
        raise ValueError(f"{detector} was scored on a different corpus")
    if status.get("results_sha256") != sha256_file(results_path):
        raise ValueError(f"{detector} result hash differs from status.json")
    configured = status.get("detector_config", {}).get("detector")
    if configured != DETECTOR_CONFIGS[detector]:
        raise ValueError(f"{detector} configuration differs from the frozen specification")
    frame = pd.read_csv(results_path, dtype={"entry_key": str})
    if len(frame) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"{detector} has {len(frame)} rows, expected {EXPECTED_TOTAL_ROWS}")
    if frame["entry_key"].duplicated().any():
        raise ValueError(f"{detector} has duplicate successful entry keys")
    success = frame["success"].map(parse_strict_bool)
    if not success.all():
        raise ValueError(f"{detector} result contains failed rows")
    probabilities = pd.to_numeric(frame["ai_probability"], errors="raise")
    if not probabilities.between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"{detector} probability outside [0, 1]")
    if detector == "zerogpt" and frame["input_truncated"].map(parse_strict_bool).any():
        raise ValueError("ZeroGPT inputs must not be truncated by the pipeline")
    return frame


def finalize(run_dir: Path) -> Dict[str, Any]:
    corpus, corpus_manifest = load_verified_corpus(run_dir)
    corpus = corpus.sort_values("corpus_row", kind="stable").reset_index(drop=True)
    corpus_sha256 = corpus_manifest["corpus_sha256"]
    merged = corpus.copy()
    detector_hashes: Dict[str, str] = {}

    for detector in CONFIRMATORY_DETECTORS:
        results = load_verified_detector(run_dir, detector, corpus_sha256)
        result_lookup = results.set_index("entry_key")
        if set(result_lookup.index) != set(merged["entry_key"]):
            raise ValueError(f"{detector} entry-key set differs from corpus")
        aligned = result_lookup.loc[merged["entry_key"]]
        if aligned["text_sha256"].tolist() != merged["text_sha256"].tolist():
            raise ValueError(f"{detector} text hashes do not align with corpus")
        prefix = detector
        merged[f"{prefix}_ai_probability"] = aligned["ai_probability"].astype(float).to_numpy()
        merged[f"{prefix}_is_ai"] = aligned["is_ai_generated"].map(parse_strict_bool).to_numpy()
        merged[f"{prefix}_response"] = aligned["raw_response"].fillna("").to_numpy()
        merged[f"{prefix}_success"] = aligned["success"].map(parse_strict_bool).to_numpy()
        merged[f"{prefix}_error"] = aligned["error"].fillna("").to_numpy()
        merged[f"{prefix}_model"] = aligned["detector_model"].to_numpy()
        merged[f"{prefix}_revision"] = aligned["detector_revision"].to_numpy()
        merged[f"{prefix}_input_mode"] = aligned["input_mode"].to_numpy()
        merged[f"{prefix}_input_token_count"] = aligned["input_token_count"].to_numpy()
        merged[f"{prefix}_used_token_count"] = aligned["used_token_count"].to_numpy()
        merged[f"{prefix}_input_truncated"] = aligned["input_truncated"].map(parse_strict_bool).to_numpy()
        merged[f"{prefix}_scored_at_utc"] = aligned["scored_at_utc"].to_numpy()
        detector_hashes[detector] = sha256_file(
            run_dir / "detectors" / detector / "results.csv"
        )

    if len(merged) != EXPECTED_TOTAL_ROWS or merged["entry_key"].duplicated().any():
        raise ValueError("Merged detector output violates row/key invariants")
    if int(merged["entry_type"].eq("human").sum()) != EXPECTED_HUMAN_ROWS:
        raise ValueError("Merged human count mismatch")
    if int(merged["entry_type"].eq("ai").sum()) != EXPECTED_AI_ROWS:
        raise ValueError("Merged AI count mismatch")
    if merged["text_sha256"].ne(merged["text"].map(sha256_text)).any():
        raise ValueError("Merged output text hashes do not match text")

    output_path = run_dir / "detection_results.csv"
    completion_path = run_dir / "completion.json"
    if output_path.exists() or completion_path.exists():
        raise FileExistsError("Refusing to overwrite an existing finalized detector run")
    _atomic_write_csv(merged, output_path)
    completion = {
        "run_id": RUN_ID,
        "status": "complete_and_verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": corpus_sha256,
        "detection_results_path": str(output_path),
        "detection_results_sha256": sha256_file(output_path),
        "detector_result_sha256": detector_hashes,
        "counts": {
            "total": len(merged),
            "human": EXPECTED_HUMAN_ROWS,
            "ai": EXPECTED_AI_ROWS,
            "under_250_chars_scored": int(merged["text"].astype(str).str.len().lt(250).sum()),
        },
        "integrity_gates": {
            "corpus_hash_matches": True,
            "all_three_detectors_complete": True,
            "all_probabilities_finite_and_in_range": True,
            "all_entries_scored_including_short_texts": True,
            "no_duplicate_entry_keys": True,
            "zero_pipeline_text_truncation_for_zerogpt": True,
            "pinned_roberta_revisions": True,
            "qa_detector_uses_pr_title_description_pair": True,
        },
    }
    _atomic_write_json(completion, completion_path)
    return completion


def verify_finalized(run_dir: Path) -> Dict[str, Any]:
    completion_path = run_dir / "completion.json"
    output_path = run_dir / "detection_results.csv"
    if not completion_path.is_file() or not output_path.is_file():
        raise FileNotFoundError("Final completion.json or detection_results.csv is missing")
    completion = json.loads(completion_path.read_text())
    if completion.get("status") != "complete_and_verified":
        raise ValueError("Final run is not complete_and_verified")
    if completion.get("detection_results_sha256") != sha256_file(output_path):
        raise ValueError("Final detection-results hash mismatch")
    corpus, corpus_manifest = load_verified_corpus(run_dir)
    if completion.get("corpus_sha256") != corpus_manifest.get("corpus_sha256"):
        raise ValueError("Final completion references a different corpus")
    frame = pd.read_csv(output_path, dtype={"entry_key": str})
    if len(frame) != EXPECTED_TOTAL_ROWS or frame["entry_key"].duplicated().any():
        raise ValueError("Final output row/key invariant failed")
    for detector in CONFIRMATORY_DETECTORS:
        success = frame[f"{detector}_success"].map(parse_strict_bool)
        if not success.all():
            raise ValueError(f"Final output contains failed {detector} rows")
        probabilities = pd.to_numeric(frame[f"{detector}_ai_probability"], errors="raise")
        if not probabilities.between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"Final output contains invalid {detector} probabilities")
    return completion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["finalize", "verify"])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir
    result = finalize(run_dir) if args.action == "finalize" else verify_finalized(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
