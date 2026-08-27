#!/usr/bin/env python3
"""Independently verify deep-audit output contracts on nokogiri.

This program does not recompute study metrics. It verifies hashes, cleaned-data keys and
cell sizes, result-family cardinalities, numeric domains, and the presence of required
candidate artifacts before anything is copied into the manuscript repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from server_guard import require_nokogiri


MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "gpt-5-mini-2025-08-07",
    "meta-llama/llama-3.1-70b-instruct",
]
CONTEXTS = ["P-1", "P-2", "P-3", "P-4", "P-5", "P-6"]
DETECTORS = [
    "ZeroGPT",
    "ChatGPT Detector RoBERTa",
    "ChatGPT-QA Detector RoBERTa",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def in_unit_interval(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return bool(((values >= 0) & (values <= 1)).all())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=Path("phase3/datasets/temperature_1/run_3_cleaned"),
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=Path("phase3/analysis")
    )
    return parser.parse_args()


def main() -> None:
    require_nokogiri()
    args = parse_args()
    clean_dir = args.clean_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    deep_outputs = analysis_dir / "outputs" / "deep_audit"
    figures = analysis_dir / "outputs" / "figures_pdf"

    integrity = json.loads((clean_dir / "integrity_summary.json").read_text())
    for key, expected in integrity["expected"].items():
        require(integrity["observed"].get(key) == expected, f"Integrity mismatch: {key}")
    for filename, expected_hash in integrity["output_sha256"].items():
        require(sha256(clean_dir / filename) == expected_hash, f"Hash mismatch: {filename}")

    clean_csv = clean_dir / "detection_results.csv"
    clean_hash = sha256(clean_csv)
    frame = pd.read_csv(clean_csv)
    ai = frame[frame["entry_type"].astype(str).str.lower().eq("ai")].copy()
    human = frame[frame["entry_type"].astype(str).str.lower().eq("human")].copy()
    ai["pr_id"] = ai["pr_id"].astype(str).str.strip()
    require(len(ai) == 5526, f"Expected 5526 AI rows, observed {len(ai)}")
    require(len(human) == 308, f"Expected 308 human rows, observed {len(human)}")
    require(
        not ai.duplicated(["pr_id", "model", "variation"]).any(),
        "Duplicate AI PR/model/context keys",
    )
    require(ai["text"].fillna("").astype(str).str.strip().ne("").all(), "Empty AI text")
    require(human["text"].fillna("").astype(str).str.strip().ne("").all(), "Empty human text")

    cell_counts = (
        ai.groupby(["model", "variation"]).size().unstack(fill_value=0)
        .reindex(index=MODELS, columns=CONTEXTS)
    )
    for model in MODELS:
        for context in CONTEXTS:
            expected = 302 if context == "P-4" else 308
            require(
                int(cell_counts.loc[model, context]) == expected,
                f"Unexpected AI cell n for {model}, {context}",
            )

    for metadata_path in sorted(deep_outputs.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        require(
            metadata.get("input_sha256") == clean_hash,
            f"Input hash mismatch in {metadata_path}",
        )

    prelim_q = pd.read_csv(deep_outputs / "prelim" / "prelim_cochrans_q.csv")
    prelim_pairs = pd.read_csv(deep_outputs / "prelim" / "prelim_mcnemar_pairwise.csv")
    require(len(prelim_q) == 6, "Expected six Preliminary omnibus rows")
    require(len(prelim_pairs) == 18, "Expected 18 Preliminary pairwise rows")
    require(
        prelim_pairs.groupby("view").size().eq(9).all(),
        "Each Preliminary view must contain one nine-test BH family",
    )
    require(in_unit_interval(prelim_q["p"]), "Invalid Preliminary omnibus p-value")
    require(in_unit_interval(prelim_pairs["p"]), "Invalid McNemar p-value")
    require(in_unit_interval(prelim_pairs["adj_p"]), "Invalid adjusted McNemar p-value")

    descriptive_dir = deep_outputs / "detection_descriptive"
    detector_metrics = pd.read_csv(
        descriptive_dir / "detector_metrics_by_view_model_context.csv"
    )
    file_diff_metrics = pd.read_csv(descriptive_dir / "prelim_file_diff_metrics.csv")
    rq1_extrema = pd.read_csv(descriptive_dir / "rq1_f1_extrema.csv")
    require(len(detector_metrics) == 108, "Expected 108 descriptive detector rows")
    require(
        detector_metrics.groupby("view").size().to_dict()
        == {"operational": 54, "successful_only": 54},
        "Expected 54 rows in each descriptive detector view",
    )
    require(len(file_diff_metrics) == 9, "Expected nine File-Diff metric rows")
    require(len(rq1_extrema) == 9, "Expected nine RQ1 F1-extrema rows")
    require(
        not detector_metrics.duplicated(
            ["view", "detector", "model", "variation"]
        ).any(),
        "Duplicate descriptive detector cell",
    )
    for metric in ["accuracy", "precision", "recall", "f1", "specificity"]:
        require(in_unit_interval(detector_metrics[metric]), f"Invalid {metric} value")
    require(
        detector_metrics[["tp", "fp", "tn", "fn"]].sum(axis=1)
        .eq(detector_metrics["n_human"] + detector_metrics["n_ai"])
        .all(),
        "Confusion counts do not sum to the evaluated samples",
    )
    operational_metrics = detector_metrics[
        detector_metrics["view"].eq("operational")
    ]
    require(
        operational_metrics["n_human"].eq(308).all(),
        "Operational descriptive cells must use 308 human negatives",
    )
    expected_ai_n = operational_metrics["variation"].map(
        {context: 302 if context == "P-4" else 308 for context in CONTEXTS}
    )
    require(
        operational_metrics["n_ai"].eq(expected_ai_n).all(),
        "Unexpected operational AI sample size",
    )
    require(
        rq1_extrema["best_f1"].ge(rq1_extrema["worst_f1"]).all(),
        "RQ1 best F1 is below worst F1",
    )

    rq1_friedman = pd.read_csv(deep_outputs / "rq1" / "rq1_friedman.csv")
    rq1_pairs = pd.read_csv(deep_outputs / "rq1" / "rq1_model_context_pairwise.csv")
    rq1_robustness = pd.read_csv(
        deep_outputs / "rq1" / "rq1_operational_vs_successful_only.csv"
    )
    require(len(rq1_friedman) == 6, "Expected six RQ1 Friedman rows")
    require(len(rq1_pairs) == 270, "Expected 270 RQ1 paired contrasts")
    family_sizes = rq1_pairs.groupby(["view", "detector", "model"]).size()
    require(family_sizes.eq(15).all(), "Every RQ1 BH family must contain 15 tests")
    require(in_unit_interval(rq1_friedman["p"]), "Invalid RQ1 Friedman p-value")
    require(in_unit_interval(rq1_pairs["p"]), "Invalid RQ1 pairwise p-value")
    require(in_unit_interval(rq1_pairs["adj_p"]), "Invalid adjusted RQ1 p-value")
    require(len(rq1_robustness) == 135, "Expected 135 RQ1 view-comparison rows")
    require(
        int(rq1_robustness["direction_reversal"].sum()) == 10,
        "Expected ten RQ1 direction reversals in successful-only sensitivity",
    )
    require(
        int(rq1_robustness["significance_change"].sum()) == 9,
        "Expected nine RQ1 significance-status changes in successful-only sensitivity",
    )
    require(
        rq1_robustness["n_successful_only"].between(11, 308).all(),
        "Unexpected successful-only paired sample size",
    )

    bm25_dir = deep_outputs / "bm25_retrieval"
    bm25_scores = pd.read_csv(bm25_dir / "bm25_retrieval_scores.csv")
    bm25_summary = pd.read_csv(bm25_dir / "bm25_retrieval_summary.csv")
    bm25_friedman = pd.read_csv(bm25_dir / "bm25_retrieval_friedman.csv")
    bm25_pairs = pd.read_csv(bm25_dir / "bm25_retrieval_linked_issue_contrasts.csv")
    require(len(bm25_scores) == 5526, "BM25 must score every cleaned AI description")
    require(len(bm25_summary) == 18, "Expected 18 BM25 model/context summaries")
    require(len(bm25_friedman) == 3, "Expected three BM25 Friedman rows")
    require(len(bm25_pairs) == 15, "Expected one 15-test BM25 contrast family")
    require(bm25_pairs.groupby("model").size().eq(5).all(), "Expected five BM25 contrasts per model")
    require(bm25_scores["relevant_rank"].between(1, 308).all(), "BM25 rank outside corpus")
    require(in_unit_interval(bm25_scores["reciprocal_rank"]), "Invalid reciprocal rank")
    require(
        (bm25_scores["recall_at_1"] <= bm25_scores["recall_at_5"]).all()
        and (bm25_scores["recall_at_5"] <= bm25_scores["recall_at_10"]).all(),
        "Non-monotonic BM25 recall indicators",
    )
    require(in_unit_interval(bm25_friedman["p"]), "Invalid BM25 Friedman p-value")
    require(in_unit_interval(bm25_pairs["adj_p"]), "Invalid adjusted BM25 p-value")

    paired_expected_rows = {
        "rq2_word_sentence_paired_stats.csv": 36,
        "rq2_reading_ease_paired_stats.csv": 18,
        "rq2_ttr_paired_stats.csv": 18,
        "rq2_mtld_paired_stats.csv": 36,
        "rq2_lexical_density_paired_stats.csv": 18,
        "rq3_codebert_paired_stats.csv": 18,
        "rq3_graphcodebert_paired_stats.csv": 18,
    }
    for filename, expected_rows in paired_expected_rows.items():
        result = pd.read_csv(analysis_dir / "outputs" / filename)
        require(len(result) == expected_rows, f"Unexpected row count in {filename}")
        require(
            not result.duplicated(["Metric", "Model", "Context"]).any(),
            f"Duplicate paired result cell in {filename}",
        )
        require(in_unit_interval(result["p"]), f"Invalid raw p-value in {filename}")
        require(in_unit_interval(result["adj_p"]), f"Invalid adjusted p-value in {filename}")
        require(pd.to_numeric(result["n"], errors="coerce").gt(0).all(), f"Invalid n in {filename}")

    required_figures = [
        "rq1_probability_boxplots_paired.pdf",
        "bm25_retrieval.pdf",
        "textual_features_boxplots.pdf",
    ]
    for filename in required_figures:
        require((figures / filename).stat().st_size > 0, f"Missing/empty figure: {filename}")

    report = {
        "status": "PASS",
        "cleaned_input_sha256": clean_hash,
        "cleaned_rows": {"ai": int(len(ai)), "human": int(len(human))},
        "ai_rows_by_model_context": {
            model: {context: int(cell_counts.loc[model, context]) for context in CONTEXTS}
            for model in MODELS
        },
        "contracts": {
            "prelim_omnibus_rows": int(len(prelim_q)),
            "prelim_pairwise_rows": int(len(prelim_pairs)),
            "descriptive_detector_rows": int(len(detector_metrics)),
            "prelim_file_diff_metric_rows": int(len(file_diff_metrics)),
            "rq1_f1_extrema_rows": int(len(rq1_extrema)),
            "rq1_friedman_rows": int(len(rq1_friedman)),
            "rq1_pairwise_rows": int(len(rq1_pairs)),
            "rq1_view_comparison_rows": int(len(rq1_robustness)),
            "rq1_view_direction_reversals": int(
                rq1_robustness["direction_reversal"].sum()
            ),
            "rq1_view_significance_changes": int(
                rq1_robustness["significance_change"].sum()
            ),
            "bm25_score_rows": int(len(bm25_scores)),
            "bm25_summary_rows": int(len(bm25_summary)),
            "bm25_pairwise_rows": int(len(bm25_pairs)),
        },
        "required_figures": {
            filename: sha256(figures / filename) for filename in required_figures
        },
    }
    output_path = deep_outputs / "verification_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
