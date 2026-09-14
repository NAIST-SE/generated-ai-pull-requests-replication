#!/usr/bin/env python3
"""Recompute one smoke case through the pinned official detector entry point."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .adapters import create_adapter
from .common import (
    B1ValidationError,
    atomic_write_json,
    gaussian_class_probability,
    load_manifest,
    load_verified_corpus,
    read_csv,
    require_approved_host,
    sha256_file,
    utc_now,
)
from .config import (
    DETECTOR_CONFIGS,
    DETECTOR_NAMES,
    RUN_ID,
    SMOKE_ABS_TOLERANCE,
    SMOKE_REL_TOLERANCE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=DETECTOR_NAMES, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase3/analysis/outputs/tier_b/b1-modern-detectors-v1"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _direct_score(adapter, detector: str, text: str) -> tuple[float, float | None]:
    """Use the official scoring entry point directly, outside adapter.score()."""
    if detector == "binoculars":
        return float(adapter.detector.compute_score(text)), None

    scoring_tokens = adapter._tokenize(  # noqa: SLF001
        adapter.scoring_tokenizer, text, truncate=True
    ).to(adapter.device)
    sampling_tokens = adapter._tokenize(  # noqa: SLF001
        adapter.sampling_tokenizer, text, truncate=True
    ).to(adapter.device)
    if not adapter.torch.equal(scoring_tokens.input_ids, sampling_tokens.input_ids):
        raise B1ValidationError("Reference tokenizers produced different input IDs")
    labels = scoring_tokens.input_ids[:, 1:]
    if labels.shape[1] < 1:
        raise B1ValidationError("Reference input has fewer than two tokens")
    with adapter.torch.inference_mode():
        logits_score = adapter.scoring_model(**scoring_tokens).logits[:, :-1]
        logits_reference = adapter.sampling_model(**sampling_tokens).logits[:, :-1]
        raw = float(adapter.criterion(logits_reference, logits_score, labels))
    probability = gaussian_class_probability(raw, adapter.config["calibration"])
    return raw, probability


def main() -> int:
    args = parse_args()
    require_approved_host(require_cuda=True)
    repo_root = args.repo_root.resolve()
    runtime_root = args.runtime_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    manifest_rows, _ = load_manifest(output_root)
    manifest_by_id = {row["description_id"]: row for row in manifest_rows}
    corpus_rows, _ = load_verified_corpus(repo_root)
    corpus_by_id = {row["entry_key"]: row for row in corpus_rows}

    detector_root = output_root / "smoke" / args.detector
    verification_path = detector_root / "verification.json"
    results_path = detector_root / "repeat-1" / "results.csv"
    if not verification_path.is_file() or not results_path.is_file():
        raise B1ValidationError(
            "Run and verify both smoke repeats before parity checking"
        )
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "passed":
        raise B1ValidationError("Smoke repeatability verification has not passed")
    results = read_csv(results_path)
    if not results:
        raise B1ValidationError("Smoke repeat 1 has no results")
    recorded = results[0]
    identifier = recorded["description_id"]
    if identifier not in manifest_by_id or identifier not in corpus_by_id:
        raise B1ValidationError("Reference case is absent from the frozen manifests")
    text = corpus_by_id[identifier]["text"]
    if recorded["input_sha256"] != manifest_by_id[identifier]["text_sha256"]:
        raise B1ValidationError("Recorded reference-case input hash mismatch")

    adapter = create_adapter(args.detector, runtime_root, device=args.device)
    direct_raw, direct_probability = _direct_score(adapter, args.detector, text)
    recorded_raw = float(recorded["raw_score"])
    raw_close = math.isclose(
        direct_raw,
        recorded_raw,
        abs_tol=SMOKE_ABS_TOLERANCE,
        rel_tol=SMOKE_REL_TOLERANCE,
    )
    probability_close = True
    recorded_probability: float | None = None
    if args.detector == "fast-detectgpt":
        recorded_probability = float(recorded["calibrated_score"])
        probability_close = math.isclose(
            direct_probability,
            recorded_probability,
            abs_tol=SMOKE_ABS_TOLERANCE,
            rel_tol=SMOKE_REL_TOLERANCE,
        )
    passed = raw_close and probability_close
    report = {
        "run_id": RUN_ID,
        "detector": args.detector,
        "status": "passed" if passed else "failed",
        "checked_at_utc": utc_now(),
        "description_id": identifier,
        "input_sha256": recorded["input_sha256"],
        "detector_config_sha256": verification["detector_config_sha256"],
        "official_entry_point": verification["reference_code_path"],
        "recorded_raw_score": recorded_raw,
        "direct_official_raw_score": direct_raw,
        "raw_absolute_difference": abs(recorded_raw - direct_raw),
        "raw_within_tolerance": raw_close,
        "recorded_calibrated_score": recorded_probability,
        "direct_calibrated_score": direct_probability,
        "calibrated_within_tolerance": probability_close,
        "tolerance": {
            "absolute": SMOKE_ABS_TOLERANCE,
            "relative": SMOKE_REL_TOLERANCE,
        },
        "threshold_source": DETECTOR_CONFIGS[args.detector]["threshold_source"],
        "smoke_results_sha256": sha256_file(results_path),
        "contains_raw_text": False,
    }
    report_path = detector_root / "reference_parity.json"
    if report_path.exists():
        raise B1ValidationError(f"Refusing to overwrite parity report: {report_path}")
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
