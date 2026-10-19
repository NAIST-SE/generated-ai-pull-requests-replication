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

from verify_rq3_publication import verify as verify_rq3_publication


ROOT = Path(__file__).resolve().parent

HASHES = {
    "phase3/analysis/outputs/descriptive_reporting_v1/manifest.json":
        "34a126c45a6f1a7358139aa887f8824246511f99f93202df4665538841fbe500",
    "rq3_publication_manifest.json":
        "80b254d149767efcc1506df384a5787db58864879ef4fc31a8a2c4df912a7c8e",
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
        "62c5d158cefb4fbf434cbdeddc2f6188040ef409be90797c03467112b1dcbd6e",
    "phase3/analysis/outputs/corrected_rq23_v1/notebooks/outputs/figures_pdf/textual_features_boxplots.pdf":
        "ac84ef02dcb96cae44da7ee14ed5d3d4fc6ef1fe68dd708c2ccf21fa05b2b715",
    "phase3/analysis/outputs/context_display_v1/presentation_manifest.json":
        "6bf14cb1a469e683b03bd6c3265d80bdd78de82000cca5a161235221b610fb11",
    "phase3/analysis/outputs/bm25_portrait_v2/presentation_manifest.json":
        "770ce9cb54a4365174cd842dfebb8b7dd225f74ac6a31dcb308315815654640a",
    "supplementary/figures/bm25_retrieval_portrait.pdf":
        "154a8a334326af270fe8ca39747231e16135f4b5f7125403ce6bd2c022ae7e64",
    "supplementary/generation_prompt.tex":
        "0ead0b56d665674b279a45aacbd49e1f13e960a0145edd474015a2e6c2cd6d94",
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
    "phase3/analysis/outputs/tier_b/b1-modern-detectors-v1/analysis_verification.json":
        "82568546f598fc35f59150ef952b2b6009720b416b449aef18b677a9e6bf67b3",
    "phase3/analysis/outputs/tier_b/b1-modern-detectors-v1/provenance/public-provenance.json":
        "b16fa090b36168f78554250b5c821ea2796c40f7b19d5d4c9d6ef60b59bef80c",
    "phase3/analysis/outputs/tier_b/b1-modern-detectors-v1/provenance/binoculars-package-inventory.txt":
        "573bc368b8510100481cad4a015872488fd5d6fe5f7c6cfcec01ff53b3a9f524",
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
    failures.extend(verify_rq3_publication())
    checks["rq3_publication"] = "61 source/numeric artifacts; 10 declared cache exclusions"

    # Additive descriptive reporting: verify copies and receipts, not statistics.
    reporting = ROOT / "phase3/analysis/outputs/descriptive_reporting_v1"
    report = load_json("phase3/analysis/outputs/descriptive_reporting_v1/manifest.json")
    for name, expected in report["output_sha256"].items():
        if sha256(reporting / name) != expected:
            failures.append(f"Descriptive reporting output mismatch: {name}")
    for relative, expected in report["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            failures.append(f"Descriptive reporting input mismatch: {relative}")
    if sha256(ROOT / "phase3/analysis/descriptive_reporting_v1/analyse_reporting.py") != report["script_sha256"]:
        failures.append("Descriptive reporting producer differs from executed source")
    independent = load_json("phase3/analysis/outputs/descriptive_reporting_v1/independent_verification.json")
    if independent["status"] != "PASS" or independent["manifest_sha256"] != sha256(reporting / "manifest.json"):
        failures.append("Descriptive reporting independent receipt does not bind to this run")
    if sha256(ROOT / "phase3/analysis/descriptive_reporting_v1/verify_reporting.py") != independent["verifier_sha256"]:
        failures.append("Descriptive reporting verifier differs from executed source")
    for name, expected_count in [("matched_textual_descriptives.csv", 108),
                                 ("detector_recall_intervals.csv", 88),
                                 ("human_false_positive_intervals.csv", 4)]:
        if len(load_csv("phase3/analysis/outputs/descriptive_reporting_v1/" + name)) != expected_count:
            failures.append(f"Descriptive reporting row count mismatch: {name}")
    if sha256(reporting / "detector_uncertainty_tables.tex") != sha256(ROOT / "supplementary/detector_uncertainty_tables.tex"):
        failures.append("Supplement detector intervals differ from verified output")
    if r"\input{detector_uncertainty_tables}" not in (ROOT / "supplementary/supplementary.tex").read_text():
        failures.append("Supplement does not include verified detector interval tables")
    checks["descriptive_reporting"] = "108 matched summaries; 88 recall and 4 human-FPR intervals; source hashes and independent receipt verified"

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

    # Reader-facing labels changed; frozen numerical/source artifacts did not.
    presentation_dir = ROOT / "phase3/analysis/outputs/context_display_v1"
    presentation = load_json("phase3/analysis/outputs/context_display_v1/presentation_manifest.json")
    for relative, expected in presentation["inputs"].items():
        if sha256(ROOT / relative) != expected:
            failures.append(f"presentation input hash mismatch: {relative}")
    for relative, expected in presentation["outputs"].items():
        if sha256(presentation_dir / relative) != expected:
            failures.append(f"presentation output hash mismatch: {relative}")
    renderer = ROOT / "phase3/analysis/presentation_labels/render_context_labels.py"
    if sha256(renderer) != presentation["script_sha256"]:
        failures.append("presentation renderer does not match the executed source")
    geometry_checks = presentation["plot_geometry_checks"]
    if len(geometry_checks) != 15 or any(
        check["geometry_before"] != check["geometry_after"] for check in geometry_checks
    ):
        failures.append("presentation plot-geometry checks are incomplete or failed")
    if sha256(presentation_dir / "textual_features_boxplots.pdf") != sha256(
        ROOT / "supplementary/figures/textual_features_boxplots.pdf"
    ):
        failures.append("supplement figure does not match the current presentation bundle")
    checks["context_display_refresh"] = {
        "inputs": len(presentation["inputs"]), "outputs": len(presentation["outputs"]),
        "geometry_checks": len(geometry_checks),
    }

    # Figure 7 uses a separate portrait asset; keep the earlier bundle immutable.
    portrait_dir = ROOT / "phase3/analysis/outputs/bm25_portrait_v2"
    portrait = load_json("phase3/analysis/outputs/bm25_portrait_v2/presentation_manifest.json")
    for relative, expected in portrait["inputs"].items():
        if sha256(ROOT / relative) != expected:
            failures.append(f"BM25 portrait input mismatch: {relative}")
    for relative, expected in portrait["outputs"].items():
        if sha256(portrait_dir / relative) != expected:
            failures.append(f"BM25 portrait output mismatch: {relative}")
    if sha256(ROOT / "phase3/analysis/presentation_labels/render_bm25_portrait.py") != portrait["script_sha256"]:
        failures.append("BM25 portrait renderer differs from executed source")
    if portrait["reference_geometry_sha256"] != portrait["portrait_geometry_sha256"]:
        failures.append("BM25 portrait changed plotted data geometry")
    if portrait["rows"] != 5322 or len(portrait["cell_counts"]) != 18:
        failures.append("BM25 portrait population is incomplete")
    if sha256(portrait_dir / "bm25_retrieval_portrait.pdf") != sha256(
        ROOT / "supplementary/figures/bm25_retrieval_portrait.pdf"
    ):
        failures.append("supplement BM25 figure differs from verified portrait")
    supplement_source = (ROOT / "supplementary/supplementary.tex").read_text()
    if r"\includegraphics[width=0.65\textwidth]{figures/bm25_retrieval_portrait.pdf}" not in supplement_source:
        failures.append("supplement does not use the current BM25 portrait asset")
    for include in ("generation_prompt",):
        if rf"\input{{{include}}}" not in supplement_source:
            failures.append(f"supplement does not include relocated material: {include}")
    checks["bm25_portrait_refresh"] = {
        "rows": portrait["rows"], "cells": len(portrait["cell_counts"]),
        "geometry_unchanged": True,
    }

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
        (
            "phase3/analysis/outputs/tier_b/b1-modern-detectors-v1/provenance/public-provenance.json",
            "status",
            "complete_and_independently_verified",
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

    b1_relative = "phase3/analysis/outputs/tier_b/b1-modern-detectors-v1"
    b1_dir = ROOT / b1_relative
    b1_metadata = load_json(f"{b1_relative}/metadata.json")
    b1_verification = load_json(f"{b1_relative}/analysis_verification.json")
    b1_provenance = load_json(f"{b1_relative}/provenance/public-provenance.json")

    if b1_verification.get("status") != "PASS":
        failures.append("B1 independent verification status is not PASS")
    for key, expected_value in (
        ("detector_count", 4),
        ("score_rows", 22520),
        ("strict_common_description_count", 5630),
        ("bootstrap_replicates", 2000),
    ):
        actual = b1_verification.get(key)
        if actual != expected_value:
            failures.append(
                f"B1 verification has {key}={actual!r}; expected {expected_value!r}"
            )

    b1_public_hashes = b1_metadata.get("output_hashes", {})
    if len(b1_public_hashes) != 9:
        failures.append(
            f"B1 metadata exposes {len(b1_public_hashes)} output hashes, expected 9"
        )
    for name, expected in b1_public_hashes.items():
        path = b1_dir / name
        if not path.is_file():
            failures.append(f"missing public B1 output: {name}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(
                f"SHA-256 mismatch for B1 output {name}: expected {expected}, got {actual}"
            )

    b1_scores = load_csv(f"{b1_relative}/all_detector_scores.csv")
    if len(b1_scores) != 22520:
        failures.append(f"B1 row-level score table has {len(b1_scores)} rows, expected 22520")
    b1_score_detectors = Counter(row.get("detector_key") for row in b1_scores)
    expected_score_detectors = {
        "binoculars": 5630,
        "roberta": 5630,
        "roberta-qa": 5630,
        "zerogpt": 5630,
    }
    if b1_score_detectors != expected_score_detectors:
        failures.append(
            f"B1 row-level detector counts are {dict(b1_score_detectors)!r}; "
            f"expected {expected_score_detectors!r}"
        )
    if len({row.get("description_id") for row in b1_scores}) != 5630:
        failures.append("B1 row-level score table does not contain 5,630 unique descriptions")
    if any(row.get("success") != "True" for row in b1_scores):
        failures.append("B1 row-level score table contains an unsuccessful detector row")

    b1_performance = load_csv(f"{b1_relative}/performance.csv")
    b1_pooled = [
        row for row in b1_performance
        if row.get("subset") == "successful_only"
        and row.get("detector") == "binoculars"
        and row.get("view_level") == "pooled"
    ]
    if len(b1_pooled) != 1:
        failures.append(f"B1 performance has {len(b1_pooled)} pooled Binoculars rows, expected 1")
    else:
        pooled = b1_pooled[0]
        expected_values = {
            "n_human": "308",
            "n_ai": "5322",
            "tp": "1546",
            "fp": "4",
        }
        for key, expected_value in expected_values.items():
            if pooled.get(key) != expected_value:
                failures.append(
                    f"B1 pooled performance has {key}={pooled.get(key)!r}; "
                    f"expected {expected_value!r}"
                )

    if b1_provenance.get("scope", {}).get("analysis_detector") != "binoculars":
        failures.append("B1 public provenance does not identify Binoculars as the analysis detector")
    if b1_provenance.get("scope", {}).get("fast_detectgpt") != (
        "deferred_unless_requested_in_tse_revision"
    ):
        failures.append("B1 public provenance does not record Fast-DetectGPT as deferred")
    if b1_provenance.get("publication_boundary", {}).get(
        "record_level_combined_scores_included"
    ) is not True:
        failures.append("B1 public provenance does not record the row-level score publication")
    checks["b1_output_hashes"] = len(b1_public_hashes)
    checks["b1_verification"] = {
        "detectors": b1_verification.get("detector_count"),
        "verified_score_rows": b1_verification.get("score_rows"),
        "public_row_level_scores": True,
    }

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
        "phase3/analysis/outputs/tier_b/b1-modern-detectors-v1/external-input",
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
        "phase3/analysis/outputs/tier_b/b1-modern-detectors-v1",
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
        "post-hoc B1 analysis",
        "Fast-DetectGPT was not run",
        "record-level scores",
        "must not be interpreted as",
        "evidence of real-world PR authorship",
    )
    for phrase in required_readme:
        if phrase not in readme:
            failures.append(f"README is missing required boundary text: {phrase!r}")
    if r"\newcommand{\WithoutFileContents}{Multi-source}" not in supplementary:
        failures.append("supplementary TeX does not map P-6's macro to Multi-source")
    if r"\WithoutFileContents~has the highest MRR" not in supplementary:
        failures.append("supplementary BM25 leading configuration is not P-6 / Multi-source")
    if "Post-hoc Binoculars Robustness Check" not in supplementary:
        failures.append("supplementary TeX is missing the post-hoc Binoculars results")
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
