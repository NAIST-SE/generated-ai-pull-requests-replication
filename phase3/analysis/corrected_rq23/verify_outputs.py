#!/usr/bin/env python3
"""Preflight and independent statistical verification for corrected RQ2/RQ3 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


CORPUS_SHA256 = "07136384785d6d02d17762535ab51a2007a7a47b8c41b77fbb311ce9c8780306"
DIFFS_SHA256 = "45c71d0170fc9343d999998ad6515e9b5b8298f63b66215f4110b3911f681d0c"
EXPECTED_HUMAN = 308
EXPECTED_AI = 5322
EXPECTED_TOTAL = 5630

MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "gpt-5-mini-2025-08-07",
    "meta-llama/llama-3.1-70b-instruct",
]
VARIATIONS = ["P-1", "P-2", "P-3", "P-4", "P-5", "P-6"]
CONTEXTS = {
    "P-1": "Repo Path",
    "P-2": "PR Title",
    "P-3": "File Diff",
    "P-4": "File Contents",
    "P-5": "Linked Issue",
    "P-6": "Without File Contents",
}
EXPECTED_COUNTS = {
    (model, variation): (
        308
        if variation in {"P-1", "P-2", "P-3"}
        or (variation == "P-4" and model == "meta-llama/llama-3.1-70b-instruct")
        else 302
        if variation == "P-4"
        else 273
    )
    for model in MODELS
    for variation in VARIATIONS
}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class MetricSpec:
    row_file: str
    stats_file: str
    columns: tuple[tuple[str, str], ...]


METRICS = (
    MetricSpec(
        "word_sentence_metrics.csv",
        "rq2_word_sentence_paired_stats.csv",
        (("word_count", "Word Count"), ("sentence_count", "Sentence Count")),
    ),
    MetricSpec(
        "fkgl_metrics.csv",
        "rq2_reading_ease_paired_stats.csv",
        (("flesch_kincaid_grade", "Flesch Kincaid Grade"),),
    ),
    MetricSpec(
        "ttr_metrics.csv",
        "rq2_ttr_paired_stats.csv",
        (("ttr", "Type-Token Ratio"),),
    ),
    MetricSpec(
        "mtld_metrics.csv",
        "rq2_mtld_paired_stats.csv",
        (("mtld_original", "MTLD Original"), ("mtld_bidir", "MTLD Bidirectional")),
    ),
    MetricSpec(
        "lexical_density_metrics.csv",
        "rq2_lexical_density_paired_stats.csv",
        (("lexical_density", "Lexical Density"),),
    ),
    MetricSpec(
        "codebert_metrics.csv",
        "rq3_codebert_paired_stats.csv",
        (("codebert_similarity", "CodeBERT Similarity"),),
    ),
    MetricSpec(
        "graphcodebert_metrics.csv",
        "rq3_graphcodebert_paired_stats.csv",
        (("graphcodebert_similarity", "GraphCodeBERT Similarity"),),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_nokogiri() -> None:
    if socket.gethostname().split(".")[0] != "nokogiri":
        raise RuntimeError("Corrected RQ2/RQ3 verification must run on nokogiri")


def normalize_model(value: object) -> str:
    text = str(value).strip()
    aliases = {
        "meta-llama_llama-3.1-70b-instruct": "meta-llama/llama-3.1-70b-instruct",
        "human": "Human",
    }
    return aliases.get(text, text)


def load_corpus(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["pr_id", "entry_type", "model", "variation", "text", "entry_key", "text_sha256"],
    )
    frame["normalized_pr_id"] = frame["pr_id"].astype(str).str.strip()
    frame["entry_type"] = frame["entry_type"].astype(str).str.lower()
    frame["model"] = frame["model"].map(normalize_model)
    frame["variation"] = frame["variation"].astype(str).str.strip()
    return frame


def validate_inputs(corpus_path: Path, diffs_path: Path) -> dict:
    if sha256(corpus_path) != CORPUS_SHA256:
        raise AssertionError("Corrected corpus SHA-256 mismatch")
    if sha256(diffs_path) != DIFFS_SHA256:
        raise AssertionError("PR diff source SHA-256 mismatch")

    frame = load_corpus(corpus_path)
    if len(frame) != EXPECTED_TOTAL:
        raise AssertionError(f"Expected {EXPECTED_TOTAL} corpus rows, observed {len(frame)}")
    if frame["entry_key"].duplicated().any():
        raise AssertionError("Corpus entry keys are not unique")
    if frame["text"].fillna("").astype(str).str.strip().eq("").any():
        raise AssertionError("Corpus contains an empty description")
    observed_text_hash = frame["text"].astype(str).map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    if not observed_text_hash.equals(frame["text_sha256"].astype(str)):
        raise AssertionError("Corpus text hashes do not match the stored descriptions")

    human = frame[frame["entry_type"].eq("human")]
    ai = frame[frame["entry_type"].eq("ai")]
    if len(human) != EXPECTED_HUMAN or len(ai) != EXPECTED_AI:
        raise AssertionError(f"Unexpected human/AI counts: {len(human)}/{len(ai)}")
    if human["normalized_pr_id"].duplicated().any():
        raise AssertionError("Human baselines are not unique by PR")
    if ai.duplicated(["normalized_pr_id", "model", "variation"]).any():
        raise AssertionError("AI corpus has duplicate PR/model/context cells")

    counts = ai.groupby(["model", "variation"]).size().to_dict()
    if counts != EXPECTED_COUNTS:
        raise AssertionError(f"AI cell counts do not match the frozen design: {counts}")

    diffs = pd.read_csv(diffs_path, usecols=["id", "patch"])
    diffs["normalized_pr_id"] = diffs["id"].astype(str).str.strip()
    diff_text = (
        diffs.groupby("normalized_pr_id")["patch"]
        .apply(lambda values: "\n".join(str(value) for value in values if pd.notna(value)))
    )
    human_ids = set(human["normalized_pr_id"])
    if set(diff_text.index) != human_ids:
        missing = sorted(human_ids.difference(diff_text.index))
        extra = sorted(set(diff_text.index).difference(human_ids))
        raise AssertionError(f"Diff PR set mismatch; missing={missing[:5]}, extra={extra[:5]}")
    if diff_text.astype(str).str.strip().eq("").any():
        raise AssertionError("At least one sampled PR has an empty aggregate diff")

    return {
        "corpus_sha256": CORPUS_SHA256,
        "diffs_sha256": DIFFS_SHA256,
        "rows": {"total": len(frame), "human": len(human), "ai": len(ai)},
        "cell_counts": {f"{model}|{variation}": count for (model, variation), count in counts.items()},
        "diff_prs": int(len(diff_text)),
    }


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    left = np.asarray(list(a), dtype=float)
    right = np.asarray(list(b), dtype=float)
    left = left[~np.isnan(left)]
    right = right[~np.isnan(right)]
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    sorted_right = np.sort(right)
    less = np.searchsorted(sorted_right, left, side="left")
    greater = len(right) - np.searchsorted(sorted_right, left, side="right")
    return float((less.sum() - greater.sum()) / (len(left) * len(right)))


def magnitude(delta: float) -> str:
    value = abs(delta)
    if value < 0.147:
        return "Negligible"
    if value < 0.33:
        return "Small"
    if value < 0.474:
        return "Medium"
    return "Large"


def rank_biserial(human: np.ndarray, ai: np.ndarray) -> float:
    differences = np.asarray(human, dtype=float) - np.asarray(ai, dtype=float)
    differences = differences[~np.isnan(differences)]
    differences = differences[differences != 0]
    if len(differences) == 0:
        return float("nan")
    ranks = rankdata(np.abs(differences))
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / (positive + negative))


def paired_result(human: np.ndarray, ai: np.ndarray) -> dict:
    human = np.asarray(human, dtype=float)
    ai = np.asarray(ai, dtype=float)
    valid = ~(np.isnan(human) | np.isnan(ai))
    human = human[valid]
    ai = ai[valid]
    if len(human) == 0:
        raise AssertionError("A paired comparison has no valid observations")
    if np.all(human - ai == 0):
        statistic, p_value = float("nan"), 1.0
    else:
        result = wilcoxon(human, ai)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    delta = cliffs_delta(human, ai)
    return {
        "n": int(len(human)),
        "median_human": float(np.median(human)),
        "median_ai": float(np.median(ai)),
        "cliffs_delta": delta,
        "magnitude": magnitude(delta),
        "rank_biserial": rank_biserial(human, ai),
        "wilcoxon_stat": statistic,
        "p": p_value,
    }


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def sig_mark(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def normalize_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["normalized_pr_id"] = frame["normalized_pr_id"].astype(str).str.strip()
    frame["entry_type"] = frame["entry_type"].astype(str).str.lower()
    frame["model"] = frame["model"].map(normalize_model)
    frame["variation"] = frame["variation"].astype(str).str.strip()
    return frame


def validate_row_keys(frame: pd.DataFrame, corpus: pd.DataFrame, ai_only: bool = False) -> None:
    expected = corpus[corpus["entry_type"].eq("ai")] if ai_only else corpus
    expected_keys = set(
        zip(expected["normalized_pr_id"], expected["entry_type"], expected["model"], expected["variation"])
    )
    observed_keys = set(
        zip(frame["normalized_pr_id"], frame["entry_type"], frame["model"], frame["variation"])
    )
    if len(frame) != len(expected) or observed_keys != expected_keys:
        raise AssertionError(
            f"Row-level metric keys differ from corpus (expected={len(expected)}, observed={len(frame)})"
        )
    if frame.duplicated(["normalized_pr_id", "entry_type", "model", "variation"]).any():
        raise AssertionError("Row-level metric output has duplicate keys")


def align_metric(frame: pd.DataFrame, model: str, variation: str, column: str) -> tuple[np.ndarray, np.ndarray]:
    human = (
        frame[frame["entry_type"].eq("human")]
        .groupby("normalized_pr_id")[column]
        .mean()
    )
    ai = (
        frame[
            frame["entry_type"].eq("ai")
            & frame["model"].eq(model)
            & frame["variation"].eq(variation)
        ]
        .groupby("normalized_pr_id")[column]
        .mean()
    )
    merged = pd.concat([human.rename("human"), ai.rename("ai")], axis=1, join="inner").dropna()
    return merged["human"].to_numpy(), merged["ai"].to_numpy()


def context_stats(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for variation in VARIATIONS:
            human, ai = align_metric(frame, model, variation, column)
            rows.append(
                {
                    "Metric": label,
                    "Model": model,
                    "Context": CONTEXTS[variation],
                    **paired_result(human, ai),
                }
            )
    result = pd.DataFrame(rows)
    result["adj_p"] = bh_adjust(result["p"])
    result["Sig"] = result["adj_p"].map(sig_mark)
    return result


def pooled_model_stats(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    human = (
        frame[frame["entry_type"].eq("human")]
        .groupby("normalized_pr_id")[column]
        .mean()
    )
    rows = []
    for model in MODELS:
        ai = (
            frame[frame["entry_type"].eq("ai") & frame["model"].eq(model)]
            .groupby("normalized_pr_id")[column]
            .mean()
        )
        merged = pd.concat([human.rename("human"), ai.rename("ai")], axis=1, join="inner").dropna()
        rows.append({"Metric": label, "Model": model, **paired_result(merged["human"], merged["ai"])})
    result = pd.DataFrame(rows)
    result["adj_p"] = bh_adjust(result["p"])
    result["Sig"] = result["adj_p"].map(sig_mark)
    return result


def compare_context_output(canonical: pd.DataFrame, observed_path: Path) -> None:
    observed = pd.read_csv(observed_path)
    keys = ["Metric", "Model", "Context"]
    metric_labels = set(canonical["Metric"].astype(str))
    observed = observed[observed["Metric"].astype(str).isin(metric_labels)].copy()
    merged = canonical.merge(observed, on=keys, suffixes=("_v", "_o"), validate="one_to_one")
    if len(merged) != len(canonical) or len(observed) != len(canonical):
        raise AssertionError(f"Context-stat row count mismatch in {observed_path}")
    numeric_pairs = {
        "n": ("n", 1e-12),
        "cliffs_delta": ("cliffs_delta", 1e-12),
        # Decimal CSV round-tripping can split exact ties in rank-based metrics.
        "rank_biserial": ("rank_biserial", 5e-5),
        "p": ("p", 1e-10),
        "adj_p": ("adj_p", 1e-10),
    }
    for canonical_col, (observed_col, tolerance) in numeric_pairs.items():
        left = merged[f"{canonical_col}_v"].to_numpy(dtype=float)
        right = merged[f"{observed_col}_o"].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=1e-10, atol=tolerance, equal_nan=True):
            difference = np.nanmax(np.abs(left - right))
            raise AssertionError(f"{canonical_col} mismatch in {observed_path}; max diff={difference}")
    for column in ("magnitude", "Sig"):
        if not merged[f"{column}_v"].astype(str).equals(merged[f"{column}_o"].astype(str)):
            raise AssertionError(f"{column} mismatch in {observed_path}")


def descriptive_stats(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    rows = []
    human = frame[frame["entry_type"].eq("human")][column].dropna()
    rows.append(
        {
            "Metric": label,
            "Model": "Human",
            "Variation": "original",
            "Context": "Human",
            "n": len(human),
            "mean": human.mean(),
            "median": human.median(),
            "q1": human.quantile(0.25),
            "q3": human.quantile(0.75),
        }
    )
    for model in MODELS:
        for variation in VARIATIONS:
            values = frame[
                frame["entry_type"].eq("ai")
                & frame["model"].eq(model)
                & frame["variation"].eq(variation)
            ][column].dropna()
            rows.append(
                {
                    "Metric": label,
                    "Model": model,
                    "Variation": variation,
                    "Context": CONTEXTS[variation],
                    "n": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                }
            )
    return pd.DataFrame(rows)


def compare_generated_summary(canonical: pd.DataFrame, observed: pd.DataFrame, keys: list[str]) -> None:
    merged = canonical.merge(observed, on=keys, suffixes=("_v", "_o"), validate="one_to_one")
    if len(merged) != len(canonical) or len(observed) != len(canonical):
        raise AssertionError("Generated-to-reference summary row count mismatch")
    for column in sorted(set(canonical.columns).intersection(observed.columns).difference(keys)):
        if pd.api.types.is_numeric_dtype(canonical[column]):
            if not np.allclose(
                merged[f"{column}_v"].to_numpy(float),
                merged[f"{column}_o"].to_numpy(float),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                raise AssertionError(f"Generated-to-reference summary mismatch: {column}")


def tokenize(text: object) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def verify_bm25(corpus: pd.DataFrame, artifacts: Path, verification_dir: Path) -> dict:
    bm25_dir = artifacts / "bm25_retrieval"
    observed_scores = pd.read_csv(bm25_dir / "bm25_retrieval_scores.csv")
    observed_scores["normalized_pr_id"] = observed_scores["normalized_pr_id"].astype(str)
    observed_scores["model"] = observed_scores["model"].map(normalize_model)
    observed_scores["variation"] = observed_scores["variation"].astype(str)

    human = corpus[corpus["entry_type"].eq("human")].sort_values("normalized_pr_id")
    ai = corpus[corpus["entry_type"].eq("ai")].copy()
    corpus_ids = human["normalized_pr_id"].tolist()
    index_by_id = {pr_id: index for index, pr_id in enumerate(corpus_ids)}
    engine = BM25Okapi(human["text"].map(tokenize).tolist(), k1=1.5, b=0.75)

    rows = []
    for row in ai.itertuples(index=False):
        scores = np.asarray(engine.get_scores(tokenize(row.text)), dtype=float)
        target = index_by_id[row.normalized_pr_id]
        rank = float(rankdata(-scores, method="average")[target])
        rows.append(
            {
                "normalized_pr_id": row.normalized_pr_id,
                "model": row.model,
                "variation": row.variation,
                "relevant_bm25_score": float(scores[target]),
                "relevant_rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "recall_at_1": int(rank <= 1),
                "recall_at_5": int(rank <= 5),
                "recall_at_10": int(rank <= 10),
            }
        )
    canonical = pd.DataFrame(rows)
    keys = ["normalized_pr_id", "model", "variation"]
    merged = canonical.merge(observed_scores, on=keys, suffixes=("_v", "_o"), validate="one_to_one")
    if len(merged) != EXPECTED_AI or len(observed_scores) != EXPECTED_AI:
        raise AssertionError("BM25 row-level output does not cover all AI descriptions")
    for column in (
        "relevant_bm25_score", "relevant_rank", "reciprocal_rank",
        "recall_at_1", "recall_at_5", "recall_at_10",
    ):
        if not np.allclose(
            merged[f"{column}_v"].to_numpy(float),
            merged[f"{column}_o"].to_numpy(float),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise AssertionError(f"BM25 row-level mismatch: {column}")

    summary = (
        canonical.assign(context=canonical["variation"].map(CONTEXTS))
        .groupby(["model", "variation", "context"], as_index=False)
        .agg(
            n=("normalized_pr_id", "size"),
            median_rank=("relevant_rank", "median"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            median_reciprocal_rank=("reciprocal_rank", "median"),
            recall_at_1=("recall_at_1", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            recall_at_10=("recall_at_10", "mean"),
        )
        .sort_values(["model", "variation"])
    )
    observed_summary = pd.read_csv(bm25_dir / "bm25_retrieval_summary.csv")
    compare_generated_summary(summary, observed_summary, ["model", "variation", "context"])

    friedman_rows = []
    contrast_rows = []
    for model in MODELS:
        subset = canonical[canonical["model"].eq(model)]
        wide = subset.pivot(
            index="normalized_pr_id", columns="variation", values="reciprocal_rank"
        ).reindex(columns=VARIATIONS).dropna()
        test = friedmanchisquare(*(wide[variation].to_numpy() for variation in VARIATIONS))
        friedman_rows.append(
            {
                "model": model,
                "n_complete_prs": len(wide),
                "statistic": float(test.statistic),
                "df": 5,
                "p": float(test.pvalue),
            }
        )
        primary = subset[subset["variation"].eq("P-5")].set_index("normalized_pr_id")["reciprocal_rank"]
        for variation in VARIATIONS:
            if variation == "P-5":
                continue
            comparison = subset[subset["variation"].eq(variation)].set_index("normalized_pr_id")["reciprocal_rank"]
            paired = pd.concat([primary.rename("primary"), comparison.rename("comparison")], axis=1).dropna()
            contrast_rows.append(
                {
                    "model": model,
                    "context_a": "Linked Issue",
                    "context_b": CONTEXTS[variation],
                    "sign": "positive values favour Linked Issue",
                    **paired_result(paired["primary"].to_numpy(), paired["comparison"].to_numpy()),
                }
            )
    friedman = pd.DataFrame(friedman_rows)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["adj_p"] = bh_adjust(contrasts["p"])
    contrasts["sig"] = contrasts["adj_p"] < 0.05

    observed_friedman = pd.read_csv(bm25_dir / "bm25_retrieval_friedman.csv")
    compare_generated_summary(friedman, observed_friedman, ["model"])
    observed_contrasts = pd.read_csv(bm25_dir / "bm25_retrieval_linked_issue_contrasts.csv")
    compare_generated_summary(contrasts, observed_contrasts, ["model", "context_a", "context_b", "sign"])

    canonical.to_csv(verification_dir / "bm25_scores_reconstructed.csv", index=False)
    summary.to_csv(verification_dir / "bm25_summary_reconstructed.csv", index=False)
    friedman.to_csv(verification_dir / "bm25_friedman_reconstructed.csv", index=False)
    contrasts.to_csv(verification_dir / "bm25_contrasts_reconstructed.csv", index=False)
    return {"rows": len(canonical), "friedman_tests": len(friedman), "contrasts": len(contrasts)}


def check_notebooks(output_root: Path) -> dict:
    manifest = json.loads((output_root / "notebooks" / "preparation_manifest.json").read_text())
    executed = 0
    for record in manifest["notebooks"]:
        path = output_root / "notebooks" / record["name"]
        notebook = json.loads(path.read_text(encoding="utf-8"))
        errors = [
            output
            for cell in notebook.get("cells", [])
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        if errors:
            raise AssertionError(f"Executed notebook contains an error output: {record['name']}")
        code_cells = [
            cell
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code" and "".join(cell.get("source", [])).strip()
        ]
        if any(cell.get("execution_count") is None for cell in code_cells):
            raise AssertionError(f"Notebook has unexecuted code cells: {record['name']}")
        executed += 1
    return {"executed_without_error": executed}


def file_hashes(root: Path) -> dict[str, str]:
    values = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "completion.json":
            values[str(path.relative_to(root))] = sha256(path)
    return values


def verify(corpus_path: Path, diffs_path: Path, output_root: Path, artifacts: Path) -> dict:
    input_report = validate_inputs(corpus_path, diffs_path)
    corpus = load_corpus(corpus_path)
    verification_dir = output_root / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)

    context_frames = []
    pooled_frames = []
    descriptive_frames = []
    checked_metric_columns = 0
    for spec in METRICS:
        row_path = artifacts / "row_level" / spec.row_file
        rows = normalize_metric_rows(pd.read_csv(row_path))
        validate_row_keys(rows, corpus)
        for column, label in spec.columns:
            if column not in rows:
                raise AssertionError(f"Missing metric column {column} in {row_path}")
            canonical_context = context_stats(rows, column, label)
            compare_context_output(canonical_context, artifacts / spec.stats_file)
            context_frames.append(canonical_context)
            pooled_frames.append(pooled_model_stats(rows, column, label))
            descriptive_frames.append(descriptive_stats(rows, column, label))
            checked_metric_columns += 1

    all_context = pd.concat(context_frames, ignore_index=True)
    all_pooled = pd.concat(pooled_frames, ignore_index=True)
    all_descriptive = pd.concat(descriptive_frames, ignore_index=True)
    all_context.to_csv(verification_dir / "canonical_context_paired_stats.csv", index=False)
    all_pooled.to_csv(verification_dir / "canonical_pooled_model_stats.csv", index=False)
    all_descriptive.to_csv(verification_dir / "canonical_descriptive_stats.csv", index=False)

    rouge = normalize_metric_rows(pd.read_csv(artifacts / "row_level" / "rouge_metrics.csv"))
    bertscore = normalize_metric_rows(pd.read_csv(artifacts / "row_level" / "bertscore_metrics.csv"))
    validate_row_keys(rouge, corpus, ai_only=True)
    validate_row_keys(bertscore, corpus, ai_only=True)
    if not rouge[["rouge1_fmeasure", "rouge2_fmeasure", "rougeL_fmeasure"]].apply(
        lambda series: series.between(0, 1).all()
    ).all():
        raise AssertionError("ROUGE scores are outside [0, 1]")

    rouge_summary = (
        rouge.groupby(["model", "variation", "Context_Name"], as_index=False)
        .agg(
            n=("normalized_pr_id", "size"),
            rouge1_mean=("rouge1_fmeasure", "mean"),
            rouge1_median=("rouge1_fmeasure", "median"),
            rouge2_mean=("rouge2_fmeasure", "mean"),
            rouge2_median=("rouge2_fmeasure", "median"),
            rougeL_mean=("rougeL_fmeasure", "mean"),
            rougeL_median=("rougeL_fmeasure", "median"),
        )
        .rename(columns={"Context_Name": "context"})
        .sort_values(["model", "variation"])
    )
    bertscore_summary = (
        bertscore.groupby(["model", "variation", "Context_Name"], as_index=False)
        .agg(n=("normalized_pr_id", "size"), mean=("bertscore_f1", "mean"), median=("bertscore_f1", "median"))
        .rename(columns={"Context_Name": "context"})
        .sort_values(["model", "variation"])
    )
    compare_generated_summary(
        rouge_summary,
        pd.read_csv(artifacts / "rq2_rouge_generated_to_reference_summary.csv"),
        ["model", "variation", "context"],
    )
    compare_generated_summary(
        bertscore_summary,
        pd.read_csv(artifacts / "rq2_bertscore_generated_to_reference_summary.csv"),
        ["model", "variation", "context"],
    )
    rouge_summary.to_csv(verification_dir / "canonical_rouge_summary.csv", index=False)
    bertscore_summary.to_csv(verification_dir / "canonical_bertscore_summary.csv", index=False)

    model_metadata = {
        name: json.loads((artifacts / "model_metadata" / f"{name}.json").read_text())
        for name in ("bertscore", "codebert", "graphcodebert")
    }
    expected_revisions = {
        "bertscore": "5b07a9086c1dbb79981ff7b05b4d1ad83b3af51c",
        "codebert": "3b0952feddeffad0063f274080e3c23d75e7eb39",
        "graphcodebert": "2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d",
    }
    for name, revision in expected_revisions.items():
        if model_metadata[name]["model_revision"] != revision:
            raise AssertionError(f"Unexpected {name} revision")

    bm25_report = verify_bm25(corpus, artifacts, verification_dir)
    notebook_report = check_notebooks(output_root)

    expected_figures = [
        "verbosity_word_count.pdf", "fkgl.pdf", "ttr.pdf",
        "MTLD_Bidirectional_Distribution_Human_vs_Contexts.pdf",
        "lexical_density.pdf", "rouge.pdf", "bertscore.pdf", "bm25_retrieval.pdf",
        "codebert.pdf", "graphcodebert.pdf", "textual_features_boxplots.pdf",
    ]
    figures = artifacts / "figures_pdf"
    for name in expected_figures:
        path = figures / name
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty figure: {path}")
    assembled_pages = len(PdfReader(str(figures / "textual_features_boxplots.pdf")).pages)
    if assembled_pages != 10:
        raise AssertionError(f"Expected 10 assembled figure pages, observed {assembled_pages}")

    report = {
        "status": "PASS",
        "protocol": "corrected-rq23-v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": input_report,
        "metric_columns_reconstructed": checked_metric_columns,
        "context_tests_reconstructed": int(len(all_context)),
        "pooled_model_tests_reconstructed": int(len(all_pooled)),
        "descriptive_cells_reconstructed": int(len(all_descriptive)),
        "generated_similarity_cells": {
            "rouge": len(rouge_summary), "bertscore": len(bertscore_summary),
        },
        "bm25": bm25_report,
        "notebooks": notebook_report,
        "assembled_figure_pages": assembled_pages,
        "model_metadata": model_metadata,
    }
    (verification_dir / "verification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    completion = {
        "status": "complete_and_verified",
        "protocol": "corrected-rq23-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_report": str((verification_dir / "verification_report.json").relative_to(output_root)),
        "output_hashes": file_hashes(output_root),
    }
    (output_root / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "verify"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--corpus", type=Path, required=True)
        sub.add_argument("--diffs", type=Path, required=True)
        if command == "verify":
            sub.add_argument("--output-root", type=Path, required=True)
            sub.add_argument("--artifacts", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    require_nokogiri()
    args = parse_args()
    if args.command == "preflight":
        print(json.dumps(validate_inputs(args.corpus.resolve(), args.diffs.resolve()), indent=2, sort_keys=True))
    else:
        verify(
            args.corpus.resolve(),
            args.diffs.resolve(),
            args.output_root.resolve(),
            args.artifacts.resolve(),
        )


if __name__ == "__main__":
    main()
