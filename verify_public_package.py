#!/usr/bin/env python3
"""Fail-closed integrity checks for the curated public replication package.

This script verifies copied artifacts, counts, completion markers, and publication
boundaries. It deliberately does not recompute any study metric or statistical test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent

HASHES = {
    "datasets/temperature_1/run_3_corrected_detection_v1/corpus.csv":
        "07136384785d6d02d17762535ab51a2007a7a47b8c41b77fbb311ce9c8780306",
    "datasets/temperature_1/run_3_corrected_detection_v1/detection_results.csv":
        "b995b02848b4e7487682379afa22c6b7d6a8a06a404f35dac3ebca60daf9aa91",
    "datasets/temperature_1/run_3_corrected_detection_v1/corpus_manifest.json":
        "546ab556c1b7e5030f72d8e86a29d009e31e069fbb98dad613ddd209ebe9c9a9",
    "datasets/temperature_1/run_3_corrected_detection_v1/completion.json":
        "a2b6d050d526ed941cc27023e767de0789c8faa117339ad7454d54c65a23b001",
    "phase3/analysis/outputs/corrected_detection_v1/verification_report.json":
        "6b8bb6cbc056de2f7408352d80730b311bf0eab4fd4e74fa1ce2575e8a4ac074",
    "phase3/analysis/outputs/corrected_detection_v1/figures/rq1_probability_boxplots_paired.pdf":
        "07f0b8a0da6a31f881e8bf7eee210cd30cf21552d2840c49d727581326351af7",
    "phase3/analysis/outputs/corrected_rq23_v1/completion.json":
        "bdd1866d1310a424c50b28a299435bda68bb3dc8f4eeb017f062ada42e9a08b5",
    "supplementary/figures/textual_features_boxplots.pdf":
        "ac84ef02dcb96cae44da7ee14ed5d3d4fc6ef1fe68dd708c2ccf21fa05b2b715",
    "phase3/analysis/outputs/tier_b/b4-issue-source-audit-v1/completion.json":
        "5f02893cc9b767358aa7da40bbe13735600c7fadf2dadd0d7ebb67418eb3cf89",
    "phase3/analysis/outputs/tier_b/generation-session-diagnostic-v1/verification.json":
        "229352273082f1531820185c2914c3049098f9f37f033a08940dabe67849a05a",
    "phase3/analysis/outputs/tier_b/generation-session-outcome-sensitivity-v1/completion.json":
        "804d6bbb8effe44e5a9805ab5310576518da1573a2c7c825b5be868ddad9b64f",
    "phase3/analysis/outputs/tier_b/generation-session-six-context-bridge-v1/completion.json":
        "29cd3172dae3214297b2c624f99f1426794fb1c09ab539caa50047078c5ef89b",
    "phase3/analysis/outputs/tier_b/b4-linked-issue-reuse-v2/completion.json":
        "44ed053466fa9fa9f527cc00df3703023bcd34802caee1f08c1b00f1d44ac4e8",
    "phase3/analysis/outputs/tier_b/b4-linked-issue-reuse-v2/verification_report.json":
        "d579b3d5fb20442c538408f3a61cab31264a882d3a57e2d24cb29425d2829111",
}

MODELS = (
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "gpt-5-mini-2025-08-07",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict:
    with (ROOT / relative).open(encoding="utf-8") as stream:
        return json.load(stream)


def load_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def csv_header(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            return set(next(reader))
        except StopIteration:
            return set()


def main() -> int:
    failures: list[str] = []
    checks: dict[str, object] = {}

    for relative, expected in HASHES.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing hash-anchored artifact: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(
                f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}"
            )
    checks["hash_anchors"] = len(HASHES)

    dataset_link = ROOT / "phase3" / "datasets"
    if not dataset_link.is_symlink():
        failures.append("phase3/datasets is not the required symlink")
    elif dataset_link.resolve() != (ROOT / "datasets").resolve():
        failures.append("phase3/datasets does not resolve to the public datasets directory")
    else:
        checks["dataset_symlink"] = "phase3/datasets -> ../datasets"

    corpus = load_csv(
        "datasets/temperature_1/run_3_corrected_detection_v1/corpus.csv"
    )
    detection = load_csv(
        "datasets/temperature_1/run_3_corrected_detection_v1/detection_results.csv"
    )
    if len(corpus) != 5630:
        failures.append(f"corrected corpus has {len(corpus)} rows, expected 5630")
    if len(detection) != 5630:
        failures.append(f"detection results have {len(detection)} rows, expected 5630")

    entry_counts = Counter(row["entry_type"] for row in corpus)
    expected_entry_counts = Counter({"ai": 5322, "human": 308})
    if entry_counts != expected_entry_counts:
        failures.append(
            f"entry-type counts differ: expected {dict(expected_entry_counts)}, "
            f"got {dict(entry_counts)}"
        )
    checks["corpus_rows"] = len(corpus)
    checks["entry_type_counts"] = dict(entry_counts)

    success_columns = ("zerogpt_success", "roberta_success", "roberta-qa_success")
    false_values = {"", "0", "false", "no", "n"}
    for column in success_columns:
        failed_rows = sum(
            row.get(column, "").strip().lower() in false_values for row in detection
        )
        if failed_rows:
            failures.append(f"{column} is false or empty on {failed_rows} rows")
    checks["detector_success_rows"] = {
        column: len(detection) for column in success_columns
    }

    observed = Counter(
        (row["variation"], row["model"])
        for row in corpus
        if row["entry_type"] == "ai"
    )
    expected: Counter[tuple[str, str]] = Counter()
    for model in MODELS:
        for variation in ("P-1", "P-2", "P-3"):
            expected[(variation, model)] = 308
        expected[("P-4", model)] = 308 if model.startswith("meta-llama/") else 302
        expected[("P-5", model)] = 273
        expected[("P-6", model)] = 273
    if observed != expected:
        missing_or_changed = {
            f"{variation}|{model}": {
                "expected": expected[(variation, model)],
                "observed": observed[(variation, model)],
            }
            for variation, model in sorted(set(expected) | set(observed))
            if expected[(variation, model)] != observed[(variation, model)]
        }
        failures.append(f"AI model/context counts differ: {missing_or_changed}")
    checks["ai_model_context_cells"] = len(observed)

    status_specs = (
        (
            "datasets/temperature_1/run_3_corrected_detection_v1/completion.json",
            "status",
            "complete_and_verified",
        ),
        (
            "phase3/analysis/outputs/corrected_rq23_v1/completion.json",
            "status",
            "complete_and_verified",
        ),
        (
            "phase3/analysis/outputs/tier_b/b4-issue-source-audit-v1/completion.json",
            "status",
            "complete",
        ),
        (
            "phase3/analysis/outputs/tier_b/generation-session-diagnostic-v1/verification.json",
            "status",
            "partial_generation_with_verified_failures",
        ),
        (
            "phase3/analysis/outputs/tier_b/generation-session-outcome-sensitivity-v1/completion.json",
            "status",
            "complete_and_verified",
        ),
        (
            "phase3/analysis/outputs/tier_b/generation-session-six-context-bridge-v1/completion.json",
            "status",
            "complete_and_verified",
        ),
        (
            "phase3/analysis/outputs/tier_b/generation-session-six-context-bridge-v1/completion.json",
            "decision_code",
            "does_not_trigger_complete_rerun",
        ),
        (
            "phase3/analysis/outputs/tier_b/b4-linked-issue-reuse-v2/completion.json",
            "status",
            "complete",
        ),
        (
            "phase3/analysis/outputs/tier_b/b4-linked-issue-reuse-v2/verification_report.json",
            "status",
            "verified",
        ),
    )
    for relative, key, expected_value in status_specs:
        actual = load_json(relative).get(key)
        if actual != expected_value:
            failures.append(
                f"{relative} has {key}={actual!r}; expected {expected_value!r}"
            )

    diagnostic = load_json(
        "phase3/analysis/outputs/tier_b/generation-session-diagnostic-v1/verification.json"
    )
    for key, expected_value in (("successful_rows", 389), ("failed_rows", 11)):
        if diagnostic.get(key) != expected_value:
            failures.append(
                f"generation diagnostic has {key}={diagnostic.get(key)!r}; "
                f"expected {expected_value}"
            )
    checks["completion_markers"] = len(status_specs)

    b4_dir = ROOT / "phase3/analysis/outputs/tier_b/b4-linked-issue-reuse-v2"
    b4_completion = load_json(
        "phase3/analysis/outputs/tier_b/b4-linked-issue-reuse-v2/completion.json"
    )
    expected_b4_counts = {
        "description_rows": 5175,
        "prompt_exact_issue_documents": 273,
        "lexical_score_rows": 10350,
        "component_score_rows": 20700,
        "bm25_score_rows": 5175,
        "bertscore_rows": 5175,
    }
    for key, expected_value in expected_b4_counts.items():
        actual = b4_completion.get("counts", {}).get(key)
        if actual != expected_value:
            failures.append(
                f"B4 completion has counts.{key}={actual!r}; expected {expected_value}"
            )
    b4_hashes = b4_completion.get("output_sha256", {})
    if len(b4_hashes) != 20:
        failures.append(f"B4 completion anchors {len(b4_hashes)} outputs, expected 20")
    for name, expected in b4_hashes.items():
        path = b4_dir / name
        if not path.is_file():
            failures.append(f"missing B4 output: {name}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(
                f"SHA-256 mismatch for B4 output {name}: expected {expected}, got {actual}"
            )
    if not all(b4_completion.get("integrity_gates", {}).values()):
        failures.append("one or more recorded B4 integrity gates are false")
    b4_verification = load_json(
        "phase3/analysis/outputs/tier_b/b4-linked-issue-reuse-v2/verification_report.json"
    )
    if b4_verification.get("recorded_output_hashes_verified") != 20:
        failures.append("B4 independent verifier did not record all 20 output hashes")
    if b4_verification.get("completion_sha256") != HASHES[
        "phase3/analysis/outputs/tier_b/b4-linked-issue-reuse-v2/completion.json"
    ]:
        failures.append("B4 verification report does not anchor the published completion file")
    checks["b4_output_hashes"] = len(b4_hashes)
    checks["b4_counts"] = expected_b4_counts

    authorized_text_artifacts = {
        "description_manifest.csv": {"qa_question"},
        "prompt_exact_issue_sources.csv": {"issue_text", "issue_title", "issue_body"},
        "high_overlap_inspection_sample.csv": {
            "description_text", "issue_text", "span_text"
        },
        "longest_shared_spans.csv": {"span_text"},
        "unrelated_issue_matches.csv": {"unrelated_issue_text"},
    }
    for name, expected_columns in authorized_text_artifacts.items():
        path = b4_dir / name
        if not path.is_file():
            failures.append(f"missing authorized B4 text-bearing artifact: {name}")
            continue
        missing_columns = expected_columns - csv_header(path)
        if missing_columns:
            failures.append(
                f"B4 text-bearing artifact {name} lacks columns {sorted(missing_columns)}"
            )
    checks["authorized_b4_text_bearing_artifacts"] = len(authorized_text_artifacts)

    forbidden_paths = []
    for path in (ROOT / "phase3").rglob("*"):
        if not path.is_file():
            continue
        lower_parts = [part.lower() for part in path.relative_to(ROOT).parts]
        if (
            path.suffix.lower() in {".log", ".exit", ".pyc"}
            or "__pycache__" in lower_parts
            or ".pytest_cache" in lower_parts
            or path.name == ".DS_Store"
            or any("token" in part or "secret" in part for part in lower_parts)
        ):
            forbidden_paths.append(str(path.relative_to(ROOT)))
    if forbidden_paths:
        failures.append(f"forbidden operational paths are present: {forbidden_paths}")

    must_be_absent = (
        "phase3/analysis/outputs/tier_b/b4-linked-issue-reuse-v1",
        "phase3/analysis/outputs/tier_b/generation-session-diagnostic-v1/gpt_current.csv",
        "phase3/analysis/outputs/tier_b/generation-session-diagnostic-v1/qwen_current.csv",
        "phase3/analysis/outputs/tier_b/generation-session-diagnostic-v1/smoke",
    )
    present_exclusions = [relative for relative in must_be_absent if (ROOT / relative).exists()]
    if present_exclusions:
        failures.append(f"deliberately excluded artifacts are present: {present_exclusions}")
    checks["excluded_artifacts_absent"] = len(must_be_absent)

    raw_text_columns = {
        "text",
        "prompt",
        "full_prompt",
        "generated_text",
        "human_text",
        "title",
        "description",
        "body",
    }
    diagnostic_directories = (
        "phase3/analysis/outputs/tier_b/generation-session-diagnostic-v1",
        "phase3/analysis/outputs/tier_b/generation-session-outcome-sensitivity-v1",
        "phase3/analysis/outputs/tier_b/generation-session-six-context-bridge-v1",
    )
    unsafe_csvs: dict[str, list[str]] = {}
    scanned_csvs = 0
    for relative in diagnostic_directories:
        for path in sorted((ROOT / relative).glob("*.csv")):
            scanned_csvs += 1
            overlap = sorted(csv_header(path) & raw_text_columns)
            if overlap:
                unsafe_csvs[str(path.relative_to(ROOT))] = overlap
    if unsafe_csvs:
        failures.append(f"raw-text diagnostic columns are present: {unsafe_csvs}")
    checks["raw_text_free_csvs"] = scanned_csvs

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    supplementary = (ROOT / "supplementary/supplementary.tex").read_text(
        encoding="utf-8"
    )
    required_readme = (
        "5,322 successful AI descriptions",
        "retained for historical provenance only",
        "b4-linked-issue-reuse-v1",
        "b4-linked-issue-reuse-v2",
        "1.20%--1.96%",
    )
    for phrase in required_readme:
        if phrase not in readme:
            failures.append(f"README is missing required boundary text: {phrase!r}")
    if "Without File Contents" not in supplementary:
        failures.append("supplementary TeX is missing the Without File Contents context")
    for stale in ("peak median F1 of 0.136", "Linked Issue consistently yielded the highest"):
        if stale in supplementary:
            failures.append(f"supplementary TeX retains stale claim: {stale!r}")
    historical_markers = (
        "datasets/temperature_1/run_3_cleaned/README.md",
        "analysis/outputs/deep_audit/README.md",
    )
    for relative in historical_markers:
        marker = ROOT / relative
        if not marker.is_file() or "Historical pre-B4" not in marker.read_text(
            encoding="utf-8"
        ):
            failures.append(f"missing historical boundary marker: {relative}")
    checks["documentation_boundary_checks"] = (
        len(required_readme) + len(historical_markers) + 3
    )

    result = {
        "status": "verified" if not failures else "failed",
        "package_root": str(ROOT),
        "checks": checks,
        "failures": failures,
        "note": "Integrity verification only; no study analyses were recomputed.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
