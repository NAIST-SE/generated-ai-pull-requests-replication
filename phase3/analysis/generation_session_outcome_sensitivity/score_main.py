#!/usr/bin/env python3
"""Compute non-neural RQ2 outcomes and validate historical values."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from scipy.stats import rankdata
import textstat

from common import (
    CONTEXTS,
    PROTOCOL,
    load_inputs,
    output_dir,
    project_root,
    public_pair_manifest,
    require_nokogiri,
    sha256_file,
    source_paths,
    utc_now,
    validate_column,
    write_json,
)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
TOLERANCE = 1e-10


def tokenize_bm25(text: object) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def base_metrics(text: str) -> dict[str, float]:
    tokens = word_tokenize(text.lower())
    return {
        "word_count": float(len(text.split())),
        "sentence_count": float(len(sent_tokenize(text))),
        "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
        "ttr": float(len(set(tokens)) / len(tokens)) if tokens else float("nan"),
    }


def main() -> None:
    require_nokogiri()
    root = project_root(Path(__file__))
    out = output_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    pairs, humans, source_hashes = load_inputs(root)

    manifest = public_pair_manifest(pairs)
    manifest_path = out / "pair_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    human_cache = {
        row.pr_id: base_metrics(str(row.text)) for row in humans.itertuples(index=False)
    }
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    human_sorted = humans.sort_values("pr_id").reset_index(drop=True)
    corpus_ids = human_sorted["pr_id"].astype(str).tolist()
    corpus_tokens = human_sorted["text"].map(tokenize_bm25).tolist()
    if any(not tokens for tokens in corpus_tokens):
        raise RuntimeError("Empty token sequence in fixed human BM25 corpus")
    relevant_index = {pr_id: index for index, pr_id in enumerate(corpus_ids)}
    bm25 = BM25Okapi(corpus_tokens, k1=1.5, b=0.75)

    rows: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        human_values = human_cache[pair.pr_id]
        for version in ("historical", "current"):
            candidate = getattr(pair, f"{version}_text")
            values = base_metrics(candidate)
            rouge = scorer.score(pair.human_text, candidate)
            query_tokens = tokenize_bm25(candidate)
            if not query_tokens:
                raise RuntimeError(f"Empty BM25 query: {pair.model} {pair.pr_id} {pair.variation}")
            scores = np.asarray(bm25.get_scores(query_tokens), dtype=float)
            target = relevant_index[pair.pr_id]
            rank = float(rankdata(-scores, method="average")[target])
            rows.append(
                {
                    "model_label": pair.model_label,
                    "model": pair.model,
                    "pr_id": pair.pr_id,
                    "variation": pair.variation,
                    "context": pair.context,
                    "version": version,
                    **values,
                    **{f"human_{name}": value for name, value in human_values.items()},
                    "rouge1_precision": float(rouge["rouge1"].precision),
                    "rouge1_recall": float(rouge["rouge1"].recall),
                    "rouge1_fmeasure": float(rouge["rouge1"].fmeasure),
                    "rouge2_precision": float(rouge["rouge2"].precision),
                    "rouge2_recall": float(rouge["rouge2"].recall),
                    "rouge2_fmeasure": float(rouge["rouge2"].fmeasure),
                    "rougeL_precision": float(rouge["rougeL"].precision),
                    "rougeL_recall": float(rouge["rougeL"].recall),
                    "rougeL_fmeasure": float(rouge["rougeL"].fmeasure),
                    "bm25_query_tokens": len(query_tokens),
                    "bm25_relevant_score": float(scores[target]),
                    "bm25_relevant_rank": rank,
                    "bm25_reciprocal_rank": 1.0 / rank,
                    "bm25_recall_at_1": int(rank <= 1),
                    "bm25_recall_at_5": int(rank <= 5),
                    "bm25_recall_at_10": int(rank <= 10),
                }
            )
    metrics = pd.DataFrame(rows).sort_values(
        ["model", "variation", "pr_id", "version"]
    )
    if len(metrics) != 778:
        raise RuntimeError(f"Expected 778 versioned metric rows, observed {len(metrics)}")

    rq2 = source_paths(root)["rq2_outputs"]
    historical = metrics[metrics["version"].eq("historical")].copy()
    historical = historical.rename(columns={"pr_id": "normalized_pr_id"})
    keys = ["normalized_pr_id", "model", "variation"]
    validations: list[dict[str, Any]] = []
    authoritative_specs: list[tuple[str, list[tuple[str, str]]]] = [
        ("row_level/word_sentence_metrics.csv", [("word_count", "word_count"), ("sentence_count", "sentence_count")]),
        ("row_level/fkgl_metrics.csv", [("flesch_kincaid_grade", "flesch_kincaid_grade")]),
        ("row_level/ttr_metrics.csv", [("ttr", "ttr")]),
        (
            "row_level/rouge_metrics.csv",
            [(name, name) for name in [
                "rouge1_precision", "rouge1_recall", "rouge1_fmeasure",
                "rouge2_precision", "rouge2_recall", "rouge2_fmeasure",
                "rougeL_precision", "rougeL_recall", "rougeL_fmeasure",
            ]],
        ),
    ]
    for relative, columns in authoritative_specs:
        expected = pd.read_csv(rq2 / relative, dtype={"normalized_pr_id": str})
        expected["normalized_pr_id"] = expected["normalized_pr_id"].astype(str).str.replace(
            r"\.0$", "", regex=True
        )
        expected = expected[
            expected["entry_type"].str.lower().eq("ai")
            & expected["model"].isin(historical["model"])
            & expected["variation"].isin(CONTEXTS)
            & expected["normalized_pr_id"].isin(historical["normalized_pr_id"])
        ]
        current_keys = historical[keys].drop_duplicates()
        expected = expected.merge(current_keys, on=keys, how="inner", validate="one_to_one")
        for observed_col, expected_col in columns:
            validations.append(
                validate_column(
                    historical,
                    expected,
                    observed_col,
                    expected_col,
                    keys,
                    f"historical_{observed_col}",
                )
            )

    bm25_expected = pd.read_csv(
        rq2 / "bm25_retrieval/bm25_retrieval_scores.csv",
        dtype={"normalized_pr_id": str},
    )
    bm25_expected["normalized_pr_id"] = bm25_expected["normalized_pr_id"].astype(str).str.replace(
        r"\.0$", "", regex=True
    )
    bm25_expected = bm25_expected.merge(
        historical[keys].drop_duplicates(), on=keys, how="inner", validate="one_to_one"
    )
    for observed_col, expected_col in [
        ("bm25_query_tokens", "query_tokens"),
        ("bm25_relevant_score", "relevant_bm25_score"),
        ("bm25_relevant_rank", "relevant_rank"),
        ("bm25_reciprocal_rank", "reciprocal_rank"),
        ("bm25_recall_at_1", "recall_at_1"),
        ("bm25_recall_at_5", "recall_at_5"),
        ("bm25_recall_at_10", "recall_at_10"),
    ]:
        validations.append(
            validate_column(
                historical,
                bm25_expected,
                observed_col,
                expected_col,
                keys,
                f"historical_{observed_col}",
            )
        )

    metrics_path = out / "main_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": utc_now(),
        "stage": "main_metrics",
        "rows": len(metrics),
        "unique_pairs": len(pairs),
        "fixed_bm25_corpus_documents": len(humans),
        "bm25_token_pattern": TOKEN_PATTERN.pattern,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "bm25_tie_policy": "average rank",
        "rouge_use_stemmer": True,
        "source_sha256": source_hashes,
        "pair_manifest_sha256": sha256_file(manifest_path),
        "output_sha256": sha256_file(metrics_path),
        "historical_validation_tolerance": TOLERANCE,
        "historical_validations": validations,
        "raw_text_in_outputs": False,
    }
    write_json(out / "main_metrics_metadata.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
