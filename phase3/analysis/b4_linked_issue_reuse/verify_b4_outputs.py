#!/usr/bin/env python3
"""Independent structural, provenance, metric, and statistics verification for B4 v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
from deep_audit.server_guard import require_nokogiri  # noqa: E402


PROTOCOL_VERSION = "b4-linked-issue-reuse-v2"
EXPECTED_ROWS = {
    "prompt_exact_issue_sources.csv": 273,
    "unrelated_issue_matches.csv": 273,
    "description_manifest.csv": 5175,
    "lexical_scores.csv": 10350,
    "metrics_long.csv": 93150,
    "component_scores.csv": 20700,
    "longest_shared_spans.csv": 10350,
    "unrelated_issue_scores.csv": 5175,
    "bm25_retrieval_scores.csv": 5175,
    "bertscore_scores.csv": 5175,
    "lexical_summary.csv": 114,
    "bm25_summary.csv": 76,
    "bertscore_summary.csv": 76,
    "high_overlap_inspection_sample.csv": 190,
    "stats_primary_p5_5gram.csv": 12,
    "stats_secondary_p6_5gram.csv": 12,
    "stats_p5_vs_p6_5gram.csv": 3,
    "stats_secondary_bm25_p5.csv": 12,
    "stats_secondary_bertscore_p5.csv": 12,
}
MODELS = {
    "Qwen/Qwen2.5-72B-Instruct",
    "gpt-5-mini-2025-08-07",
    "meta-llama/llama-3.1-70b-instruct",
}
UNEXPOSED = {"P-1", "P-2", "P-3", "P-4"}
ISSUE_MANIFEST_SHA256 = "bdeb1d70b03c46d9f54e8775bb816d71183d486a5e55d9f76f992ab7938b49dc"
CORRECTED_CORPUS_SHA256 = "07136384785d6d02d17762535ab51a2007a7a47b8c41b77fbb311ce9c8780306"
WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^\)]+\)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase3/analysis/outputs/tier_b/b4-linked-issue-reuse-v2"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "phase3/analysis/outputs/tier_b/"
            "b4-linked-issue-reuse-v2-verification.json"
        ),
    )
    return parser.parse_args()


def resolve_under(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_columns(frame: pd.DataFrame, columns: set[str], filename: str) -> None:
    missing = columns.difference(frame.columns)
    require(not missing, f"{filename} is missing columns: {sorted(missing)}")


def normalize(text: str, representation: str) -> str:
    if representation == "raw_case_sensitive":
        return unicodedata.normalize("NFC", text)
    require(representation == "conservative_normalized", "Unexpected representation")
    value = unicodedata.normalize("NFKC", text)
    value = MARKDOWN_LINK_PATTERN.sub(r"\1", value)
    return " ".join(value.split()).lower()


def tokens(text: str, representation: str = "conservative_normalized") -> list[str]:
    return WORD_PATTERN.findall(normalize(text, representation))


def ngram_containment(source: list[str], generated: list[str], n: int) -> float:
    total = len(generated) - n + 1
    if total <= 0 or len(source) < n:
        return 0.0
    source_windows = {
        tuple(source[index:index + n]) for index in range(len(source) - n + 1)
    }
    matched = sum(
        tuple(generated[index:index + n]) in source_windows for index in range(total)
    )
    return matched / total


def matched_pairs_rank_biserial(exposed: np.ndarray, control: np.ndarray) -> float:
    difference = exposed - control
    difference = difference[~np.isnan(difference)]
    difference = difference[difference != 0]
    if not len(difference):
        return float("nan")
    ranks = rankdata(np.abs(difference))
    positive = ranks[difference > 0].sum()
    negative = ranks[difference < 0].sum()
    return float((positive - negative) / (len(difference) * (len(difference) + 1) / 2))


def expected_n(model: str, control: str) -> int:
    if control == "P-4" and model in {
        "Qwen/Qwen2.5-72B-Instruct",
        "gpt-5-mini-2025-08-07",
    }:
        return 267
    return 273


def verify_stats(
    path: Path,
    frame: pd.DataFrame,
    value_col: str,
    expected_exposed: str,
    expected_controls: set[str],
) -> dict:
    recorded = pd.read_csv(path)
    require_columns(
        recorded,
        {
            "model",
            "exposed_context",
            "control_context",
            "n",
            "median_exposed",
            "median_control",
            "rank_biserial",
            "wilcoxon_stat",
            "p",
            "adj_p",
            "sig",
        },
        path.name,
    )
    require(set(recorded["model"]) == MODELS, f"{path.name}: unexpected model set")
    require(
        set(recorded["exposed_context"]) == {expected_exposed},
        f"{path.name}: unexpected exposed context",
    )
    require(
        set(recorded["control_context"]) == expected_controls,
        f"{path.name}: unexpected controls",
    )
    reconstructed = []
    ai = frame[frame["source_type"].eq("AI")]
    for model in sorted(MODELS):
        model_data = ai[ai["model"].eq(model)]
        exposed = model_data[model_data["context_code"].eq(expected_exposed)].set_index(
            "pr_id"
        )[value_col]
        for control_code in sorted(expected_controls):
            control = model_data[model_data["context_code"].eq(control_code)].set_index(
                "pr_id"
            )[value_col]
            aligned = pd.concat(
                [exposed.rename("exposed"), control.rename("control")],
                axis=1,
                join="inner",
            ).dropna()
            exposed_values = aligned["exposed"].to_numpy(float)
            control_values = aligned["control"].to_numpy(float)
            difference = exposed_values - control_values
            if np.all(difference == 0):
                statistic, p_value = float("nan"), 1.0
            else:
                statistic, p_value = wilcoxon(exposed_values, control_values)
            reconstructed.append(
                {
                    "model": model,
                    "exposed_context": expected_exposed,
                    "control_context": control_code,
                    "n": len(aligned),
                    "median_exposed": float(np.median(exposed_values)),
                    "median_control": float(np.median(control_values)),
                    "rank_biserial": matched_pairs_rank_biserial(
                        exposed_values, control_values
                    ),
                    "wilcoxon_stat": float(statistic),
                    "p": float(p_value),
                }
            )
    rebuilt = pd.DataFrame(reconstructed)
    rebuilt["adj_p"] = multipletests(rebuilt["p"], method="fdr_bh")[1]
    rebuilt["sig"] = rebuilt["adj_p"] < 0.05
    keys = ["model", "exposed_context", "control_context"]
    comparison = recorded.merge(
        rebuilt, on=keys, suffixes=("_recorded", "_rebuilt"), validate="one_to_one"
    )
    require(len(comparison) == len(recorded), f"{path.name}: contrast key mismatch")
    for column in [
        "n",
        "median_exposed",
        "median_control",
        "rank_biserial",
        "wilcoxon_stat",
        "p",
        "adj_p",
    ]:
        left = pd.to_numeric(comparison[f"{column}_recorded"], errors="coerce")
        right = pd.to_numeric(comparison[f"{column}_rebuilt"], errors="coerce")
        require(
            np.allclose(left, right, rtol=1e-9, atol=1e-12, equal_nan=True),
            f"{path.name}: reconstructed {column} mismatch",
        )
    recorded_sig = comparison["sig_recorded"].map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() == "true"
    )
    require(
        recorded_sig.eq(comparison["sig_rebuilt"]).all(),
        f"{path.name}: reconstructed significance mismatch",
    )
    for row in recorded.itertuples(index=False):
        require(
            int(row.n) == expected_n(row.model, row.control_context),
            f"{path.name}: unexpected paired n for {row.model}/{row.control_context}",
        )
    return {
        "rows": len(recorded),
        "min_n": int(recorded["n"].min()),
        "max_n": int(recorded["n"].max()),
        "reconstructed": True,
    }


def main() -> None:
    require_nokogiri()
    args = parse_args()
    input_root = args.repo_root.resolve()
    code_root = Path(__file__).resolve().parents[3]
    output_dir = resolve_under(code_root, args.output_dir)
    report_path = resolve_under(code_root, args.report)
    require(output_dir.is_dir(), f"Missing B4 output: {output_dir}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite verification report: {report_path}")

    completion_path = output_dir / "completion.json"
    metadata_path = output_dir / "metadata.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(completion.get("protocol_version") == PROTOCOL_VERSION, "Protocol mismatch")
    require(completion.get("status") == "complete", "B4 output is not complete")
    require(metadata.get("protocol_version") == PROTOCOL_VERSION, "Metadata mismatch")
    require(
        metadata.get("generation_protocol_version") == "b4-corrected-generation-v3.2",
        "Generation protocol mismatch",
    )

    recorded_hashes = completion.get("output_sha256", {})
    require(recorded_hashes, "Completion has no recorded output hashes")
    for filename, expected_hash in recorded_hashes.items():
        require(
            sha256_file(output_dir / filename) == expected_hash,
            f"Output hash mismatch: {filename}",
        )

    observed_rows = {}
    frames = {}
    for filename, expected_rows in EXPECTED_ROWS.items():
        path = output_dir / filename
        require(path.is_file(), f"Missing output: {filename}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frames[filename] = frame
        observed_rows[filename] = len(frame)
        require(
            len(frame) == expected_rows,
            f"{filename}: expected {expected_rows} rows, observed {len(frame)}",
        )

    issue_manifest_path = (
        input_root
        / "phase3/analysis/outputs/tier_b/b4-issue-source-audit-v1/"
        "regeneration_sources.jsonl"
    )
    corpus_path = (
        input_root
        / "phase3/datasets/temperature_1/run_3_corrected_detection_v1/corpus.csv"
    )
    require(sha256_file(issue_manifest_path) == ISSUE_MANIFEST_SHA256, "Issue input hash mismatch")
    require(sha256_file(corpus_path) == CORRECTED_CORPUS_SHA256, "Corpus input hash mismatch")

    source_records = [
        json.loads(line)
        for line in issue_manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected_source_hashes = {
        str(record["pr_id"]): record["prompt_issue_block_sha256"]
        for record in source_records
    }
    sources = frames["prompt_exact_issue_sources.csv"]
    require_columns(
        sources,
        {"pr_id", "issue_text", "issue_sha256", "prompt_rows_verified"},
        "prompt_exact_issue_sources.csv",
    )
    require(sources["pr_id"].nunique() == 273, "Source table does not have 273 unique PRs")
    require(sources["issue_sha256"].nunique() == 265, "Expected 265 unique issue blocks")
    require(
        int((sources["issue_sha256"].value_counts() > 1).sum()) == 8,
        "Expected eight duplicate rendered issue-block groups",
    )
    require(sources["prompt_rows_verified"].astype(int).eq(6).all(), "Prompt coverage mismatch")
    require(
        sources["issue_text"].map(sha256_text).eq(sources["issue_sha256"]).all(),
        "Source text hash mismatch",
    )
    require(
        sources.set_index("pr_id")["issue_sha256"].to_dict() == expected_source_hashes,
        "Output sources differ from frozen issue manifest",
    )

    manifest = frames["description_manifest.csv"]
    require_columns(
        manifest,
        {
            "description_id",
            "description_sha256",
            "pr_id",
            "entry_type",
            "variation",
            "model",
            "exposure",
        },
        "description_manifest.csv",
    )
    require(not manifest["description_id"].duplicated().any(), "Duplicate description IDs")
    ai = manifest[manifest["entry_type"].str.lower().eq("ai")]
    human = manifest[manifest["entry_type"].str.lower().eq("human")]
    require(len(ai) == 4902 and len(human) == 273, "AI/human description counts mismatch")
    require(set(ai["model"]) == MODELS, "Description model set mismatch")
    require(
        not ai.duplicated(["pr_id", "model", "variation"]).any(),
        "Duplicate AI model/PR/context keys",
    )

    corpus = pd.read_csv(corpus_path, dtype=str, keep_default_na=False)
    eligible_corpus = corpus[corpus["pr_id"].isin(set(sources["pr_id"]))].copy()
    eligible_corpus.loc[
        eligible_corpus["entry_type"].str.lower().eq("human"), "model"
    ] = "Human"
    eligible_corpus.loc[
        eligible_corpus["entry_type"].str.lower().eq("human"), "variation"
    ] = "HUMAN"
    corpus_by_key = eligible_corpus.set_index(["pr_id", "model", "variation"])
    manifest_by_key = manifest.set_index(["pr_id", "model", "variation"])
    require(set(corpus_by_key.index) == set(manifest_by_key.index), "Corpus/manifest key mismatch")
    aligned_corpus = corpus_by_key.loc[manifest_by_key.index]
    require(
        aligned_corpus["text_sha256"].eq(manifest_by_key["description_sha256"]).all(),
        "Corpus/manifest description hash mismatch",
    )

    lexical = pd.read_csv(output_dir / "lexical_scores.csv")
    require(
        not lexical.duplicated(["description_id", "representation"]).any(),
        "Duplicate lexical score keys",
    )
    for column in [
        "ngram_3_containment",
        "ngram_5_containment",
        "ngram_8_containment",
        "matched_5gram_token_coverage",
        "longest_span_generated_fraction",
        "rouge_l_precision",
        "rouge_l_recall",
        "rouge_l_f1",
    ]:
        values = pd.to_numeric(lexical[column], errors="coerce")
        require(values.notna().all() and values.between(0, 1).all(), f"Invalid {column}")

    source_text = sources.set_index("pr_id")["issue_text"]
    text_by_description = {
        manifest_by_key.at[key, "description_id"]: row["text"]
        for key, row in corpus_by_key.iterrows()
    }
    primary = lexical[lexical["representation"].eq("conservative_normalized")].copy()
    independently_computed = []
    for row in primary.itertuples(index=False):
        independently_computed.append(
            ngram_containment(
                tokens(source_text.loc[str(row.pr_id)]),
                tokens(text_by_description[row.description_id]),
                5,
            )
        )
    require(
        np.allclose(
            primary["ngram_5_containment"].to_numpy(float),
            np.asarray(independently_computed),
            rtol=0,
            atol=1e-15,
        ),
        "Independent normalized 5-gram reconstruction mismatch",
    )

    unrelated = frames["unrelated_issue_matches.csv"]
    require(
        unrelated["pr_id"].ne(unrelated["unrelated_pr_id"]).all(),
        "Unrelated control reused the target PR",
    )
    source_hash_by_pr = sources.set_index("pr_id")["issue_sha256"]
    require(
        unrelated.apply(
            lambda row: row["unrelated_issue_sha256"] != source_hash_by_pr.loc[row["pr_id"]],
            axis=1,
        ).all(),
        "Unrelated control reused identical issue text",
    )
    source_lengths = {
        row.pr_id: len(tokens(row.issue_text)) for row in sources.itertuples(index=False)
    }
    for row in unrelated.itertuples(index=False):
        candidates = sources[
            sources["pr_id"].ne(row.pr_id)
            & sources["issue_sha256"].ne(source_hash_by_pr.loc[row.pr_id])
        ].copy()
        candidates["distance"] = candidates["pr_id"].map(
            lambda pr_id: abs(source_lengths[pr_id] - source_lengths[row.pr_id])
        )
        chosen = candidates.sort_values(["distance", "issue_sha256", "pr_id"]).iloc[0]
        require(chosen["pr_id"] == row.unrelated_pr_id, "Unrelated selection is not deterministic")
        require(
            chosen["issue_sha256"] == row.unrelated_issue_sha256,
            "Unrelated issue hash mismatch",
        )

    bm25 = pd.read_csv(output_dir / "bm25_retrieval_scores.csv")
    ordered_sources = sources.sort_values("pr_id").reset_index(drop=True)
    bm25_engine = BM25Okapi(
        ordered_sources["issue_text"].map(tokens).tolist(), k1=1.5, b=0.75
    )
    target_index = {
        pr_id: index for index, pr_id in enumerate(ordered_sources["pr_id"].tolist())
    }
    rebuilt_ranks = []
    rebuilt_scores = []
    for row in bm25.itertuples(index=False):
        scores = np.asarray(bm25_engine.get_scores(tokens(text_by_description[row.description_id])))
        rebuilt_scores.append(float(scores[target_index[str(row.pr_id)]]))
        rebuilt_ranks.append(float(rankdata(-scores, method="average")[target_index[str(row.pr_id)]]))
    require(
        np.allclose(bm25["true_issue_score"].to_numpy(float), rebuilt_scores, rtol=1e-10, atol=1e-12),
        "Independent BM25 score reconstruction mismatch",
    )
    require(
        np.allclose(bm25["true_issue_rank"].to_numpy(float), rebuilt_ranks, rtol=0, atol=0),
        "Independent BM25 rank reconstruction mismatch",
    )
    require(
        np.allclose(bm25["reciprocal_rank"].to_numpy(float), 1 / np.asarray(rebuilt_ranks)),
        "Independent BM25 reciprocal-rank mismatch",
    )

    bertscore = pd.read_csv(output_dir / "bertscore_scores.csv")
    for column in ["bertscore_precision", "bertscore_recall", "bertscore_f1"]:
        values = pd.to_numeric(bertscore[column], errors="coerce")
        require(values.notna().all() and np.isfinite(values).all(), f"Invalid {column}")
    require(
        set(bertscore["description_id"]) == set(manifest["description_id"]),
        "BERTScore/description key mismatch",
    )

    stat_diagnostics = {
        "primary_p5_5gram": verify_stats(
            output_dir / "stats_primary_p5_5gram.csv",
            primary,
            "ngram_5_containment",
            "P-5",
            UNEXPOSED,
        ),
        "secondary_p6_5gram": verify_stats(
            output_dir / "stats_secondary_p6_5gram.csv",
            primary,
            "ngram_5_containment",
            "P-6",
            UNEXPOSED,
        ),
        "p5_vs_p6_5gram": verify_stats(
            output_dir / "stats_p5_vs_p6_5gram.csv",
            primary,
            "ngram_5_containment",
            "P-5",
            {"P-6"},
        ),
        "secondary_bm25_p5": verify_stats(
            output_dir / "stats_secondary_bm25_p5.csv",
            bm25,
            "reciprocal_rank",
            "P-5",
            UNEXPOSED,
        ),
        "secondary_bertscore_p5": verify_stats(
            output_dir / "stats_secondary_bertscore_p5.csv",
            bertscore,
            "bertscore_f1",
            "P-5",
            UNEXPOSED,
        ),
    }

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "completion_sha256": sha256_file(completion_path),
        "metadata_sha256": sha256_file(metadata_path),
        "recorded_output_hashes_verified": len(recorded_hashes),
        "row_counts": observed_rows,
        "statistical_tables": stat_diagnostics,
        "checks": {
            "frozen_input_hashes": "pass",
            "issue_manifest_agreement": "pass",
            "duplicate_issue_block_audit": "pass",
            "description_corpus_agreement": "pass",
            "primary_5gram_independent_reconstruction": "pass",
            "unrelated_control_distinct_hash_and_deterministic": "pass",
            "bm25_independent_reconstruction": "pass",
            "bertscore_finite_and_complete": "pass",
            "predeclared_statistics_independent_reconstruction": "pass",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
