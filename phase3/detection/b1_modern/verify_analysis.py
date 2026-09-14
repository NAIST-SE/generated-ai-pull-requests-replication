#!/usr/bin/env python3
"""Independently reconstruct the frozen B1 scientific summaries.

This verifier intentionally does not import ``analyse.py`` or ``paired_stats.py``.
It reads the immutable combined-detector score table and recomputes the reported
performance, paired differences, agreement counts, and modern-detector context tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import (
    B1ValidationError,
    atomic_write_json,
    load_manifest,
    require_approved_host,
    sha256_file,
    utc_now,
)
from .config import (
    ANALYSIS_DETECTOR_NAMES,
    CONTEXT_CODES,
    CONTEXT_LABELS,
    ORIGINAL_DETECTION_RESULTS_RELATIVE_PATH,
    ORIGINAL_DETECTION_RESULTS_SHA256,
    ORIGINAL_DETECTORS,
    RUN_ID,
    RUN_SEED,
)


FLOAT_TOLERANCE = 1e-10
EXPECTED_FILES = {
    "all_detector_scores.csv",
    "performance.csv",
    "failure_and_truncation_rates.csv",
    "paired_modern_vs_existing_recall.csv",
    "paired_modern_vs_existing_specificity.csv",
    "pairwise_detector_agreement.csv",
    "rq1_friedman.csv",
    "rq1_pooled_context_pairwise.csv",
    "rq1_generator_context_descriptive.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase3/analysis/outputs/tier_b/b1-modern-detectors-v1"),
    )
    return parser.parse_args()


def _close(actual: Any, expected: Any, label: str) -> None:
    actual_float = float("nan") if str(actual).strip() == "" else float(actual)
    expected_float = float("nan") if str(expected).strip() == "" else float(expected)
    if math.isnan(actual_float) and math.isnan(expected_float):
        return
    if not np.isclose(
        actual_float,
        expected_float,
        rtol=FLOAT_TOLERANCE,
        atol=FLOAT_TOLERANCE,
    ):
        raise B1ValidationError(
            f"{label}: expected {expected_float!r}, observed {actual_float!r}"
        )


def _threshold(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame["entry_type"].eq("ai").to_numpy(dtype=bool)
    prediction = frame["predicted_label"].eq("AI").to_numpy(dtype=bool)
    tp = int((truth & prediction).sum())
    fn = int((truth & ~prediction).sum())
    tn = int((~truth & ~prediction).sum())
    fp = int((~truth & prediction).sum())
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "accuracy": (tp + tn) / len(frame),
        "balanced_accuracy": (recall + specificity) / 2.0,
    }


def _verify_score_sources(
    repo_root: Path,
    output_root: Path,
    scores: pd.DataFrame,
) -> None:
    verification_path = output_root / "verified" / "verification.json"
    accepted_path = output_root / "verified" / "accepted_scores.csv"
    if not verification_path.is_file() or not accepted_path.is_file():
        raise B1ValidationError("Verified modern-detector source table is missing")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (
        verification.get("status") != "verified"
        or sha256_file(accepted_path) != verification.get("accepted_scores_sha256")
    ):
        raise B1ValidationError("Modern-detector accepted-score provenance is invalid")
    accepted = pd.read_csv(
        accepted_path,
        dtype={"description_id": str, "pr_id": str},
        keep_default_na=False,
    )
    detector_map = {
        "Binoculars": "binoculars",
        "Fast-DetectGPT": "fast-detectgpt",
    }
    accepted["detector_key"] = accepted["detector_name"].map(detector_map)
    if accepted["detector_key"].isna().any():
        raise B1ValidationError("Unknown detector in modern accepted-score table")
    modern = scores[scores["detector_key"].isin(ANALYSIS_DETECTOR_NAMES)]
    columns = [
        "description_id",
        "detector_key",
        "pr_id",
        "entry_type",
        "model",
        "context_code",
        "run",
        "predicted_label",
    ]
    merged = modern.merge(
        accepted,
        on=["description_id", "detector_key"],
        how="outer",
        suffixes=("_analysis", "_source"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise B1ValidationError("Modern source/analysis score coverage differs")
    for field in columns[2:]:
        if not merged[f"{field}_analysis"].equals(merged[f"{field}_source"]):
            raise B1ValidationError(f"Modern source/analysis mismatch: {field}")
    if not np.allclose(
        pd.to_numeric(merged["ai_score_analysis"]),
        pd.to_numeric(merged["ai_score_source"]),
        rtol=0.0,
        atol=0.0,
    ):
        raise B1ValidationError("Modern source/analysis score values differ")

    original_path = repo_root / ORIGINAL_DETECTION_RESULTS_RELATIVE_PATH
    if (
        not original_path.is_file()
        or sha256_file(original_path) != ORIGINAL_DETECTION_RESULTS_SHA256
    ):
        raise B1ValidationError("Frozen original-detector source hash mismatch")
    original = pd.read_csv(
        original_path, dtype={"entry_key": str, "pr_id": str}, keep_default_na=False
    )
    for detector_key, spec in ORIGINAL_DETECTORS.items():
        prefix = spec["prefix"]
        source = original[
            ["entry_key", f"{prefix}_ai_probability", f"{prefix}_is_ai"]
        ].copy()
        source.columns = ["description_id", "source_score", "source_label"]
        source["source_label"] = np.where(
            source["source_label"].astype(str).str.lower().isin({"true", "1", "yes"}),
            "AI",
            "Not AI",
        )
        analysis = scores[scores["detector_key"].eq(detector_key)][
            ["description_id", "ai_score", "predicted_label"]
        ]
        joined = analysis.merge(
            source,
            on="description_id",
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        if not joined["_merge"].eq("both").all():
            raise B1ValidationError(
                f"Original source/analysis score coverage differs: {detector_key}"
            )
        if not np.allclose(
            pd.to_numeric(joined["ai_score"]),
            pd.to_numeric(joined["source_score"]),
            rtol=0.0,
            atol=0.0,
        ) or not joined["predicted_label"].equals(joined["source_label"]):
            raise B1ValidationError(
                f"Original source/analysis score values differ: {detector_key}"
            )


def _cluster_intervals(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    include_ranking: bool,
) -> dict[str, float]:
    pr_ids, pr_codes = np.unique(frame["pr_id"].astype(str), return_inverse=True)
    truth = frame["entry_type"].eq("ai").to_numpy(dtype=bool)
    prediction = frame["predicted_label"].eq("AI").to_numpy(dtype=bool)
    scores = frame["ai_score"].to_numpy(dtype=float)
    metric_names = [
        "recall",
        "specificity",
        "precision",
        "f1",
        "accuracy",
        "balanced_accuracy",
    ]
    if include_ranking:
        metric_names.extend(("auroc", "auprc"))
    values: dict[str, list[float]] = {metric: [] for metric in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        cluster_weights = rng.multinomial(
            len(pr_ids), np.full(len(pr_ids), 1.0 / len(pr_ids))
        )
        weights = cluster_weights[pr_codes]
        positives = int(weights[truth].sum())
        negatives = int(weights[~truth].sum())
        if not positives or not negatives:
            continue
        tp = int(weights[truth & prediction].sum())
        fn = positives - tp
        fp = int(weights[~truth & prediction].sum())
        tn = negatives - fp
        recall = tp / positives
        specificity = tn / negatives
        precision = tp / (tp + fp) if tp + fp else 0.0
        current = {
            "recall": recall,
            "specificity": specificity,
            "precision": precision,
            "f1": (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "accuracy": (tp + tn) / (positives + negatives),
            "balanced_accuracy": (recall + specificity) / 2.0,
        }
        if include_ranking:
            current["auroc"] = float(
                roc_auc_score(truth.astype(int), scores, sample_weight=weights)
            )
            current["auprc"] = float(
                average_precision_score(
                    truth.astype(int), scores, sample_weight=weights
                )
            )
        if all(math.isfinite(value) for value in current.values()):
            for metric, value in current.items():
                values[metric].append(value)
    valid = len(values["recall"])
    if valid < replicates * 0.95:
        raise B1ValidationError("Independent bootstrap has too few valid replicates")
    output: dict[str, float] = {"bootstrap_valid_replicates": valid}
    for metric, estimates in values.items():
        output[f"{metric}_ci_low"] = float(np.quantile(estimates, 0.025))
        output[f"{metric}_ci_high"] = float(np.quantile(estimates, 0.975))
    return output


def _performance_view(
    scores: pd.DataFrame,
    detector: str,
    row: pd.Series,
) -> pd.DataFrame:
    detector_frame = scores[scores["detector_key"].eq(detector)]
    human = detector_frame[detector_frame["entry_type"].eq("human")]
    ai = detector_frame[detector_frame["entry_type"].eq("ai")]
    level = row["view_level"]
    if level in {"generator", "generator_context"}:
        ai = ai[ai["model"].eq(row["generator"])]
    if level == "generator_context":
        ai = ai[ai["context_code"].eq(row["context_code"])]
    return pd.concat([human, ai], ignore_index=True)


def _verify_performance(
    scores: pd.DataFrame,
    common_ids: set[str],
    reported: pd.DataFrame,
    replicates: int,
) -> None:
    detector_count = len(ANALYSIS_DETECTOR_NAMES) + len(ORIGINAL_DETECTORS)
    expected_views = 2 * detector_count * 22
    if len(reported) != expected_views:
        raise B1ValidationError(
            f"Performance table must contain {expected_views} frozen views"
        )
    keys = ["subset", "detector", "view_level", "generator", "context_code"]
    if reported.duplicated(keys).any():
        raise B1ValidationError("Performance table contains duplicate views")
    for _, row in reported.iterrows():
        population = (
            scores
            if row["subset"] == "successful_only"
            else scores[scores["description_id"].isin(common_ids)]
        )
        frame = _performance_view(population, row["detector"], row)
        expected = _threshold(frame)
        truth = frame["entry_type"].eq("ai").astype(int)
        include_ranking = row["view_level"] != "generator_context"
        if include_ranking:
            expected["auroc"] = float(roc_auc_score(truth, frame["ai_score"]))
            expected["auprc"] = float(
                average_precision_score(truth, frame["ai_score"])
            )
        stable_seed = RUN_SEED + int(
            hashlib.sha256(
                "|".join(
                    (
                        row["subset"],
                        row["detector"],
                        row["view_level"],
                        row["generator"],
                        row["context_code"],
                    )
                ).encode()
            ).hexdigest()[:8],
            16,
        )
        expected.update(
            _cluster_intervals(
                frame,
                replicates=replicates,
                seed=stable_seed,
                include_ranking=include_ranking,
            )
        )
        if int(row["n_human"]) != int(frame["entry_type"].eq("human").sum()):
            raise B1ValidationError("Performance human count mismatch")
        if int(row["n_ai"]) != int(frame["entry_type"].eq("ai").sum()):
            raise B1ValidationError("Performance AI count mismatch")
        for field, value in expected.items():
            _close(row[field], value, f"performance/{row['detector']}/{field}")


def _pair(
    scores: pd.DataFrame,
    detector_a: str,
    detector_b: str,
    entry_type: str,
) -> pd.DataFrame:
    fields = [
        "description_id",
        "pr_id",
        "model",
        "context_code",
        "predicted_label",
    ]
    left = scores[
        scores["detector_key"].eq(detector_a)
        & scores["entry_type"].eq(entry_type)
    ][fields]
    right = scores[
        scores["detector_key"].eq(detector_b)
        & scores["entry_type"].eq(entry_type)
    ][fields]
    frame = left.merge(
        right,
        on=["description_id", "pr_id", "model", "context_code"],
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    frame["a"] = frame["predicted_label_a"].eq("AI")
    frame["b"] = frame["predicted_label_b"].eq("AI")
    return frame


def _paired_interval(
    frame: pd.DataFrame,
    *,
    specificity: bool,
    replicates: int,
    seed: int,
) -> tuple[float, float, int]:
    pr_ids, pr_codes = np.unique(frame["pr_id"].astype(str), return_inverse=True)
    prediction_a = frame["a"].to_numpy(dtype=bool)
    prediction_b = frame["b"].to_numpy(dtype=bool)
    if specificity:
        prediction_a = ~prediction_a
        prediction_b = ~prediction_b
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        cluster_weights = rng.multinomial(
            len(pr_ids), np.full(len(pr_ids), 1.0 / len(pr_ids))
        )
        weights = cluster_weights[pr_codes]
        denominator = int(weights.sum())
        value_a = float(weights[prediction_a].sum() / denominator)
        value_b = float(weights[prediction_b].sum() / denominator)
        values.append(value_a - value_b)
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(values),
    )


def _filter_reported_view(frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if row.get("view_level") in {"generator", "generator_context"}:
        frame = frame[frame["model"].eq(row["generator"])]
    if row.get("view_level") == "generator_context":
        frame = frame[frame["context_code"].eq(row["context_code"])]
    return frame


def _verify_paired_differences(
    scores: pd.DataFrame,
    recall: pd.DataFrame,
    specificity: pd.DataFrame,
    replicates: int,
) -> None:
    expected_recall = len(ANALYSIS_DETECTOR_NAMES) * len(ORIGINAL_DETECTORS) * 22
    expected_specificity = len(ANALYSIS_DETECTOR_NAMES) * len(ORIGINAL_DETECTORS)
    if len(recall) != expected_recall or len(specificity) != expected_specificity:
        raise B1ValidationError("Paired detector comparison family size mismatch")
    recall_keys = [
        "modern_detector",
        "existing_detector",
        "view_level",
        "generator",
        "context_code",
    ]
    if recall.duplicated(recall_keys).any() or specificity.duplicated(
        ["modern_detector", "existing_detector"]
    ).any():
        raise B1ValidationError("Paired detector table contains duplicate views")
    for _, row in recall.iterrows():
        frame = _pair(
            scores,
            row["modern_detector"],
            row["existing_detector"],
            "ai",
        )
        frame = _filter_reported_view(frame, row)
        modern = float(frame["a"].mean())
        existing = float(frame["b"].mean())
        seed = RUN_SEED + int(
            hashlib.sha256(
                "|".join(
                    (
                        "recall",
                        row["modern_detector"],
                        row["existing_detector"],
                        row["view_level"],
                        row["generator"],
                        row["context_code"],
                    )
                ).encode()
            ).hexdigest()[:8],
            16,
        )
        low, high, valid = _paired_interval(
            frame, specificity=False, replicates=replicates, seed=seed
        )
        expected = {
            "modern_recall": modern,
            "existing_recall": existing,
            "recall_difference_modern_minus_existing": modern - existing,
            "difference_ci_low": low,
            "difference_ci_high": high,
            "bootstrap_valid_replicates": valid,
            "modern_only_ai": int((frame["a"] & ~frame["b"]).sum()),
            "existing_only_ai": int((~frame["a"] & frame["b"]).sum()),
        }
        for field, value in expected.items():
            _close(row[field], value, f"paired recall/{field}")
    for _, row in specificity.iterrows():
        frame = _pair(
            scores,
            row["modern_detector"],
            row["existing_detector"],
            "human",
        )
        modern = float((~frame["a"]).mean())
        existing = float((~frame["b"]).mean())
        seed = RUN_SEED + int(
            hashlib.sha256(
                f"specificity|{row['modern_detector']}|{row['existing_detector']}".encode()
            ).hexdigest()[:8],
            16,
        )
        low, high, valid = _paired_interval(
            frame, specificity=True, replicates=replicates, seed=seed
        )
        expected = {
            "modern_specificity": modern,
            "existing_specificity": existing,
            "specificity_difference_modern_minus_existing": modern - existing,
            "difference_ci_low": low,
            "difference_ci_high": high,
            "bootstrap_valid_replicates": valid,
            "modern_only_ai": int((frame["a"] & ~frame["b"]).sum()),
            "existing_only_ai": int((~frame["a"] & frame["b"]).sum()),
        }
        for field, value in expected.items():
            _close(row[field], value, f"paired specificity/{field}")


def _agreement_frame(scores: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if row["stratum_type"] == "entry_type":
        entry_types = [row["generator"]]
    elif row["stratum_type"] == "overall":
        entry_types = ["ai", "human"]
    else:
        entry_types = ["ai"]
    frame = pd.concat(
        [
            _pair(scores, row["detector_a"], row["detector_b"], entry_type)
            for entry_type in entry_types
        ],
        ignore_index=True,
    )
    if row["stratum_type"] in {"generator", "generator_context"}:
        frame = frame[frame["model"].eq(row["generator"])]
    if row["stratum_type"] in {"context", "generator_context"}:
        frame = frame[frame["context_code"].eq(row["context_code"])]
    return frame


def _verify_agreement(scores: pd.DataFrame, reported: pd.DataFrame) -> None:
    detector_count = len(ANALYSIS_DETECTOR_NAMES) + len(ORIGINAL_DETECTORS)
    expected_strata = math.comb(detector_count, 2) * 30
    if len(reported) != expected_strata:
        raise B1ValidationError(
            f"Pairwise agreement table must contain {expected_strata} strata"
        )
    agreement_keys = [
        "detector_a",
        "detector_b",
        "stratum_type",
        "generator",
        "context_code",
    ]
    if reported.duplicated(agreement_keys).any():
        raise B1ValidationError("Pairwise agreement table contains duplicate strata")
    for _, row in reported.iterrows():
        frame = _agreement_frame(scores, row)
        a = frame["a"]
        b = frame["b"]
        expected = {
            "n_descriptions": len(frame),
            "n_prs": frame["pr_id"].nunique(),
            "both_ai": int((a & b).sum()),
            "a_only_ai": int((a & ~b).sum()),
            "b_only_ai": int((~a & b).sum()),
            "both_not_ai": int((~a & ~b).sum()),
            "agreement_rate": float((a == b).mean()),
        }
        for field, value in expected.items():
            _close(row[field], value, f"agreement/{field}")


def _rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    difference = difference[difference != 0]
    if not len(difference):
        return float("nan")
    ranks = rankdata(np.abs(difference))
    positive = float(ranks[difference > 0].sum())
    negative = float(ranks[difference < 0].sum())
    return (positive - negative) / (len(difference) * (len(difference) + 1) / 2)


def _bh(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _context_population(
    scores: pd.DataFrame, common_ids: set[str], subset: str
) -> pd.DataFrame:
    frame = scores[scores["detector_key"].isin(ANALYSIS_DETECTOR_NAMES)]
    if subset != "successful_only":
        frame = frame[frame["description_id"].isin(common_ids)]
    return frame[frame["entry_type"].eq("ai")]


def _verify_context_outputs(
    scores: pd.DataFrame,
    common_ids: set[str],
    friedman: pd.DataFrame,
    pairwise: pd.DataFrame,
    descriptive: pd.DataFrame,
) -> None:
    modern_count = len(ANALYSIS_DETECTOR_NAMES)
    if (
        len(friedman) != 2 * modern_count
        or len(pairwise) != 30 * modern_count
        or len(descriptive) != 36 * modern_count
    ):
        raise B1ValidationError("Modern-detector context output family size mismatch")
    if friedman.duplicated(["subset", "detector"]).any() or pairwise.duplicated(
        ["subset", "detector", "context_a", "context_b"]
    ).any() or descriptive.duplicated(
        ["subset", "detector", "model", "context_code"]
    ).any():
        raise B1ValidationError("Modern-detector context output contains duplicates")
    for _, row in friedman.iterrows():
        frame = _context_population(scores, common_ids, row["subset"])
        frame = frame[frame["detector_key"].eq(row["detector"])]
        pooled = frame.groupby(["pr_id", "context_code"])["ai_score"].mean()
        wide = pooled.unstack().reindex(columns=CONTEXT_CODES).dropna()
        result = friedmanchisquare(
            *(wide[context].to_numpy() for context in CONTEXT_CODES)
        )
        _close(row["n_complete_prs"], len(wide), "Friedman n")
        _close(row["statistic"], result.statistic, "Friedman statistic")
        _close(row["p"], result.pvalue, "Friedman p")
    for (subset, detector), family in pairwise.groupby(
        ["subset", "detector"], sort=False
    ):
        if len(family) != 15:
            raise B1ValidationError("Context pairwise FDR family is not 15")
        frame = _context_population(scores, common_ids, subset)
        frame = frame[frame["detector_key"].eq(detector)]
        pooled = frame.groupby(["pr_id", "context_code"])["ai_score"].mean()
        wide = pooled.unstack().reindex(columns=CONTEXT_CODES).dropna()
        reconstructed: list[dict[str, float]] = []
        for context_a, context_b in combinations(CONTEXT_CODES, 2):
            a = wide[context_a].to_numpy()
            b = wide[context_b].to_numpy()
            if np.all(a == b):
                statistic, p_value = float("nan"), 1.0
            else:
                statistic, p_value = wilcoxon(a, b)
            reconstructed.append(
                {
                    "context_a": CONTEXT_LABELS[context_a],
                    "context_b": CONTEXT_LABELS[context_b],
                    "n": len(wide),
                    "median_a": float(np.median(a)),
                    "median_b": float(np.median(b)),
                    "rank_biserial_a_minus_b": _rank_biserial(a, b),
                    "wilcoxon_stat": float(statistic),
                    "p": float(p_value),
                }
            )
        expected = pd.DataFrame(reconstructed)
        expected["adj_p"] = _bh(expected["p"])
        observed = family.set_index(["context_a", "context_b"])
        for _, expected_row in expected.iterrows():
            key = (expected_row["context_a"], expected_row["context_b"])
            if key not in observed.index:
                raise B1ValidationError(f"Missing context contrast: {key}")
            actual = observed.loc[key]
            for field in (
                "n",
                "median_a",
                "median_b",
                "rank_biserial_a_minus_b",
                "wilcoxon_stat",
                "p",
                "adj_p",
            ):
                _close(actual[field], expected_row[field], f"context/{key}/{field}")
    for _, row in descriptive.iterrows():
        frame = _context_population(scores, common_ids, row["subset"])
        values = frame[
            frame["detector_key"].eq(row["detector"])
            & frame["model"].eq(row["model"])
            & frame["context_code"].eq(row["context_code"])
        ]["ai_score"].to_numpy()
        expected = {
            "n_descriptions": len(values),
            "minimum": np.min(values),
            "q1": np.quantile(values, 0.25),
            "median": np.median(values),
            "q3": np.quantile(values, 0.75),
            "maximum": np.max(values),
        }
        for field, value in expected.items():
            _close(row[field], value, f"context descriptive/{field}")


def _verify_failure_rows(
    manifest: pd.DataFrame,
    scores: pd.DataFrame,
    reported: pd.DataFrame,
) -> None:
    detector_count = len(ANALYSIS_DETECTOR_NAMES) + len(ORIGINAL_DETECTORS)
    if len(reported) != 17 * detector_count or reported.duplicated(
        ["detector", "stratum_type", "stratum"]
    ).any():
        raise B1ValidationError("Failure/truncation table strata are incomplete")
    for _, row in reported.iterrows():
        detector = row["detector"]
        detector_scores = scores[scores["detector_key"].eq(detector)]
        frame = manifest.copy()
        successful = set(detector_scores["description_id"])
        frame["success"] = frame["description_id"].isin(successful)
        truncated = dict(
            zip(
                detector_scores["description_id"],
                detector_scores["truncated"].astype(str).str.lower().eq("true"),
            )
        )
        frame["truncated"] = frame["description_id"].map(truncated).fillna(False)
        frame["length"] = pd.cut(
            frame["char_count"].astype(int),
            bins=[-1, 249, 999, np.inf],
            labels=["under_250_chars", "250_to_999_chars", "1000_or_more_chars"],
        ).astype(str)
        kind = row["stratum_type"]
        label = row["stratum"]
        if kind == "entry_type":
            frame = frame[frame["entry_type"].eq(label)]
        elif kind == "model":
            frame = frame[frame["model"].replace("", "not_applicable").eq(label)]
        elif kind == "context":
            frame = frame[
                frame["context_code"].replace("", "not_applicable").eq(label)
            ]
        elif kind == "length":
            frame = frame[frame["length"].eq(label)]
        elif kind != "overall":
            raise B1ValidationError(f"Unknown failure stratum: {kind}")
        successes = int(frame["success"].sum())
        truncated_successes = int(frame["truncated"].sum())
        expected = {
            "attempted": len(frame),
            "successful": successes,
            "failures": len(frame) - successes,
            "failure_rate": (len(frame) - successes) / len(frame),
            "truncated_successes": truncated_successes,
            "truncation_rate_among_successes": truncated_successes / successes
            if successes
            else float("nan"),
        }
        for field, value in expected.items():
            _close(row[field], value, f"failure/{detector}/{kind}/{label}/{field}")


def main() -> int:
    args = parse_args()
    require_approved_host(require_cuda=False)
    repo_root = args.repo_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    analysis_dir = output_root / "analysis"
    metadata_path = analysis_dir / "metadata.json"
    if not metadata_path.is_file():
        raise B1ValidationError("B1 analysis metadata is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("run_id") != RUN_ID:
        raise B1ValidationError("B1 analysis run ID mismatch")
    expected_sources = {
        "analyse.py": Path(__file__).with_name("analyse.py"),
        "verify_analysis.py": Path(__file__),
        "paired_stats.py": Path(__file__).resolve().parents[2]
        / "analysis"
        / "paired_stats.py",
    }
    if set(metadata.get("analysis_source_hashes", {})) != set(expected_sources):
        raise B1ValidationError("B1 analysis source-hash record is incomplete")
    for name, path in expected_sources.items():
        if sha256_file(path) != metadata["analysis_source_hashes"][name]:
            raise B1ValidationError(f"B1 analysis source hash mismatch: {name}")
    if set(metadata.get("output_hashes", {})) != EXPECTED_FILES:
        raise B1ValidationError("B1 analysis output set is incomplete")
    for name, expected_hash in metadata["output_hashes"].items():
        if sha256_file(analysis_dir / name) != expected_hash:
            raise B1ValidationError(f"B1 analysis output hash mismatch: {name}")

    scores = pd.read_csv(
        analysis_dir / "all_detector_scores.csv",
        dtype={"description_id": str, "pr_id": str},
        keep_default_na=False,
    )
    if "text" in scores.columns or set(scores["detector_key"].unique()) != set(
        (*ANALYSIS_DETECTOR_NAMES, *ORIGINAL_DETECTORS)
    ):
        raise B1ValidationError("Combined detector score table is invalid")
    scores["ai_score"] = pd.to_numeric(scores["ai_score"], errors="raise")
    _verify_score_sources(repo_root, output_root, scores)
    successful_sets = [
        set(group["description_id"])
        for _, group in scores.groupby("detector_key", sort=True)
    ]
    common_ids = set.intersection(*successful_sets)
    if len(common_ids) != int(metadata["strict_common_description_count"]):
        raise B1ValidationError("Strict all-detector common-set count mismatch")
    if scores["detector_key"].nunique() != int(
        metadata["strict_common_detector_count"]
    ):
        raise B1ValidationError("Strict all-detector detector-count mismatch")

    replicates = int(metadata["bootstrap_replicates"])
    _verify_performance(
        scores,
        common_ids,
        pd.read_csv(analysis_dir / "performance.csv", keep_default_na=False),
        replicates,
    )
    _verify_paired_differences(
        scores[scores["description_id"].isin(common_ids)],
        pd.read_csv(
            analysis_dir / "paired_modern_vs_existing_recall.csv",
            keep_default_na=False,
        ),
        pd.read_csv(
            analysis_dir / "paired_modern_vs_existing_specificity.csv",
            keep_default_na=False,
        ),
        replicates,
    )
    _verify_agreement(
        scores[scores["description_id"].isin(common_ids)],
        pd.read_csv(
            analysis_dir / "pairwise_detector_agreement.csv",
            keep_default_na=False,
        ),
    )
    _verify_context_outputs(
        scores,
        common_ids,
        pd.read_csv(analysis_dir / "rq1_friedman.csv", keep_default_na=False),
        pd.read_csv(
            analysis_dir / "rq1_pooled_context_pairwise.csv",
            keep_default_na=False,
        ),
        pd.read_csv(
            analysis_dir / "rq1_generator_context_descriptive.csv",
            keep_default_na=False,
        ),
    )
    manifest_rows, _ = load_manifest(output_root)
    _verify_failure_rows(
        pd.DataFrame(manifest_rows),
        scores,
        pd.read_csv(
            analysis_dir / "failure_and_truncation_rates.csv",
            keep_default_na=False,
        ),
    )
    report = {
        "run_id": RUN_ID,
        "status": "PASS",
        "verified_at_utc": utc_now(),
        "analysis_metadata_sha256": sha256_file(metadata_path),
        "verified_output_hashes": metadata["output_hashes"],
        "detector_count": scores["detector_key"].nunique(),
        "score_rows": len(scores),
        "strict_common_description_count": len(common_ids),
        "bootstrap_replicates": replicates,
        "independence": "does not import production analysis or paired-statistics code",
    }
    report_path = output_root / "analysis_verification.json"
    if report_path.exists():
        raise B1ValidationError(
            f"Refusing to overwrite existing analysis verification: {report_path}"
        )
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
