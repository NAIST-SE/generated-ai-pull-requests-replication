#!/usr/bin/env python3
"""Server-only confirmatory B1 performance and paired context analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import paired_stats as PS  # noqa: E402

from .common import (  # noqa: E402
    B1ValidationError,
    atomic_write_json,
    load_manifest,
    require_approved_host,
    sha256_file,
    utc_now,
)
from .config import (  # noqa: E402
    ANALYSIS_DETECTOR_NAMES,
    BOOTSTRAP_REPLICATES,
    CORPUS_SHA256,
    CONTEXT_CODES,
    CONTEXT_LABELS,
    RUN_ID,
    RUN_SEED,
    ORIGINAL_DETECTION_RESULTS_RELATIVE_PATH,
    ORIGINAL_DETECTION_RESULTS_SHA256,
    ORIGINAL_DETECTION_VERIFICATION_RELATIVE_PATH,
    ORIGINAL_DETECTION_VERIFICATION_SHA256,
    ORIGINAL_DETECTORS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase3/analysis/outputs/tier_b/b1-modern-detectors-v1"),
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES
    )
    return parser.parse_args()


def _read_score_table(path: Path) -> pd.DataFrame:
    """Read accepted scores without converting blank non-applicable fields to NaN."""
    return pd.read_csv(
        path,
        dtype={"pr_id": str, "description_id": str},
        keep_default_na=False,
    )


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _strict_bool(series: pd.Series, field: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0", "yes", "no"}).all():
        raise B1ValidationError(f"Invalid boolean value in {field}")
    return normalized.isin({"true", "1", "yes"})


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _threshold_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    truth = frame["entry_type"].eq("ai")
    prediction = frame["predicted_label"].eq("AI")
    tp = int((truth & prediction).sum())
    fn = int((truth & ~prediction).sum())
    tn = int((~truth & ~prediction).sum())
    fp = int((~truth & prediction).sum())
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    # Match the study-wide zero_division=0 policy for a detector that predicts
    # no positive rows. The count-based form also avoids propagating 0/0 NaNs.
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    f1 = float((2 * tp) / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "accuracy": _safe_divide(tp + tn, len(frame)),
        "balanced_accuracy": (recall + specificity) / 2.0,
    }


def _cluster_bootstrap_ci(
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
    rng = np.random.default_rng(seed)
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
    estimates: dict[str, list[float]] = {name: [] for name in metric_names}
    valid_replicates = 0
    for _ in range(replicates):
        cluster_counts = rng.multinomial(
            len(pr_ids), np.full(len(pr_ids), 1.0 / len(pr_ids))
        )
        weights = cluster_counts[pr_codes]
        positive_weight = int(weights[truth].sum())
        negative_weight = int(weights[~truth].sum())
        if not positive_weight or not negative_weight:
            continue
        tp = int(weights[truth & prediction].sum())
        fn = positive_weight - tp
        fp = int(weights[~truth & prediction].sum())
        tn = negative_weight - fp
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        current = {
            "recall": recall,
            "specificity": specificity,
            "precision": precision,
            "f1": (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "accuracy": (tp + tn) / (positive_weight + negative_weight),
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
        if not all(math.isfinite(value) for value in current.values()):
            continue
        for name, value in current.items():
            estimates[name].append(value)
        valid_replicates += 1
    if valid_replicates < replicates * 0.95:
        raise B1ValidationError("Too many invalid cluster-bootstrap replicates")
    intervals: dict[str, float] = {
        "bootstrap_valid_replicates": valid_replicates
    }
    for name, values in estimates.items():
        intervals[f"{name}_ci_low"] = float(np.quantile(values, 0.025))
        intervals[f"{name}_ci_high"] = float(np.quantile(values, 0.975))
    return intervals


def _performance_rows(
    scores: pd.DataFrame,
    *,
    subset: str,
    replicates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for detector, detector_frame in scores.groupby("detector_key", sort=True):
        human = detector_frame[detector_frame["entry_type"].eq("human")]
        ai_frame = detector_frame[detector_frame["entry_type"].eq("ai")]
        model_views = [("pooled", "pooled", "all", ai_frame)]
        model_views.extend(
            ("generator", model, "all", group)
            for model, group in detector_frame[
                detector_frame["entry_type"].eq("ai")
            ].groupby("model", sort=True)
        )
        model_views.extend(
            ("generator_context", model, context, group)
            for (model, context), group in ai_frame.groupby(
                ["model", "context_code"], sort=True
            )
        )
        for view_level, generator, context, ai in model_views:
            evaluation = pd.concat([human, ai], ignore_index=True)
            truth = evaluation["entry_type"].eq("ai").astype(int)
            values = evaluation["ai_score"].astype(float)
            if truth.nunique() != 2:
                raise B1ValidationError(
                    "Performance view lacks both classes: "
                    f"{detector}/{view_level}/{generator}/{context}"
                )
            stable_seed = RUN_SEED + int(
                hashlib.sha256(
                    f"{subset}|{detector}|{view_level}|{generator}|{context}".encode()
                ).hexdigest()[:8],
                16,
            )
            include_ranking = view_level != "generator_context"
            ranking = (
                {
                    "auroc": float(roc_auc_score(truth, values)),
                    "auprc": float(average_precision_score(truth, values)),
                }
                if include_ranking
                else {"auroc": float("nan"), "auprc": float("nan")}
            )
            rows.append(
                {
                    "subset": subset,
                    "detector": detector,
                    "view_level": view_level,
                    "generator": generator,
                    "context_code": context,
                    "context": CONTEXT_LABELS.get(context, context),
                    "n_human": int((truth == 0).sum()),
                    "n_ai": int((truth == 1).sum()),
                    "n_prs": int(evaluation["pr_id"].nunique()),
                    **ranking,
                    **_cluster_bootstrap_ci(
                        evaluation,
                        replicates=replicates,
                        seed=stable_seed,
                        include_ranking=include_ranking,
                    ),
                    **_threshold_metrics(evaluation),
                }
            )
    return rows


def _load_original_scores(
    repo_root: Path,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    results_path = repo_root / ORIGINAL_DETECTION_RESULTS_RELATIVE_PATH
    verification_path = repo_root / ORIGINAL_DETECTION_VERIFICATION_RELATIVE_PATH
    if not results_path.is_file() or not verification_path.is_file():
        raise B1ValidationError("Corrected original-detector results are missing")
    if sha256_file(results_path) != ORIGINAL_DETECTION_RESULTS_SHA256:
        raise B1ValidationError("Corrected original-detector result hash mismatch")
    if sha256_file(verification_path) != ORIGINAL_DETECTION_VERIFICATION_SHA256:
        raise B1ValidationError("Original-detector verification-report hash mismatch")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (
        verification.get("status") != "PASS"
        or verification.get("input_sha256") != ORIGINAL_DETECTION_RESULTS_SHA256
        or verification.get("corpus_sha256") != CORPUS_SHA256
    ):
        raise B1ValidationError("Original-detector independent verification is invalid")

    source = pd.read_csv(results_path, dtype=str, keep_default_na=False)
    required = {
        "entry_key",
        "pr_id",
        "entry_type",
        "model",
        "variation",
        "source_experiment_id",
        "text_sha256",
    }
    for spec in ORIGINAL_DETECTORS.values():
        prefix = spec["prefix"]
        required.update(
            {
                f"{prefix}_ai_probability",
                f"{prefix}_is_ai",
                f"{prefix}_success",
                f"{prefix}_model",
                f"{prefix}_revision",
                f"{prefix}_input_mode",
                f"{prefix}_input_token_count",
                f"{prefix}_used_token_count",
                f"{prefix}_input_truncated",
                f"{prefix}_scored_at_utc",
            }
        )
    if not required.issubset(source.columns):
        raise B1ValidationError(
            f"Original-detector table is missing columns: {sorted(required - set(source))}"
        )
    if len(source) != len(manifest) or not source["entry_key"].is_unique:
        raise B1ValidationError("Original-detector row coverage is invalid")

    internal = manifest.set_index("description_id", verify_integrity=True)
    original = source.set_index("entry_key", verify_integrity=True)
    if set(internal.index) != set(original.index):
        raise B1ValidationError("Original-detector identifiers do not match B1 manifest")
    original = original.reindex(internal.index)
    human = original["entry_type"].str.strip().str.lower().eq("human")
    normalized = pd.DataFrame(
        {
            "pr_id": original["pr_id"],
            "entry_type": original["entry_type"].str.strip().str.lower(),
            "model": original["model"].where(~human, ""),
            "context_code": original["variation"].where(~human, ""),
            "run": original["source_experiment_id"].where(~human, "human_baseline"),
            "text_sha256": original["text_sha256"],
        },
        index=original.index,
    )
    for field in normalized.columns:
        if not normalized[field].astype(str).equals(internal[field].astype(str)):
            raise B1ValidationError(
                f"Original-detector metadata differs from B1 manifest: {field}"
            )

    long_parts: list[pd.DataFrame] = []
    for detector_key, spec in ORIGINAL_DETECTORS.items():
        prefix = spec["prefix"]
        success = _strict_bool(original[f"{prefix}_success"], f"{prefix}_success")
        if not success.all():
            raise B1ValidationError(
                f"Frozen original detector contains failures: {detector_key}"
            )
        for field, expected in (
            (f"{prefix}_model", spec["model"]),
            (f"{prefix}_revision", spec["revision"]),
            (f"{prefix}_input_mode", spec["input_mode"]),
        ):
            if not original[field].eq(expected).all():
                raise B1ValidationError(
                    f"Frozen original-detector configuration mismatch: {field}"
                )
        score = pd.to_numeric(
            original[f"{prefix}_ai_probability"], errors="raise"
        ).astype(float)
        if not np.isfinite(score).all() or not score.between(0.0, 1.0).all():
            raise B1ValidationError(f"Invalid original-detector score: {detector_key}")
        predicted = _strict_bool(
            original[f"{prefix}_is_ai"], f"{prefix}_is_ai"
        )
        input_tokens = pd.to_numeric(
            original[f"{prefix}_input_token_count"], errors="coerce"
        )
        used_tokens = pd.to_numeric(
            original[f"{prefix}_used_token_count"], errors="coerce"
        )
        truncated = _strict_bool(
            original[f"{prefix}_input_truncated"],
            f"{prefix}_input_truncated",
        )
        part = normalized.drop(columns="text_sha256").copy()
        part.insert(0, "description_id", part.index.astype(str))
        part["detector_key"] = detector_key
        part["detector_name"] = spec["display_name"]
        part["detector_version"] = spec["revision"]
        part["ai_score"] = score.to_numpy()
        part["predicted_label"] = np.where(predicted, "AI", "Not AI")
        part["input_tokens"] = input_tokens.to_numpy()
        part["used_input_tokens"] = used_tokens.to_numpy()
        part["truncated"] = truncated.to_numpy()
        part["success"] = True
        part["source_route"] = "corrected-original-detector-rerun"
        long_parts.append(part.reset_index(drop=True))
    combined = pd.concat(long_parts, ignore_index=True)
    if "text" in combined.columns or len(combined) != len(manifest) * len(
        ORIGINAL_DETECTORS
    ):
        raise B1ValidationError("Original-detector long-table construction failed")
    return combined, {
        "results_path": str(results_path),
        "results_sha256": ORIGINAL_DETECTION_RESULTS_SHA256,
        "verification_path": str(verification_path),
        "verification_sha256": ORIGINAL_DETECTION_VERIFICATION_SHA256,
    }


def _paired_cluster_difference_ci(
    paired: pd.DataFrame,
    *,
    outcome: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, int]:
    if outcome not in {"recall", "specificity"}:
        raise ValueError(f"Unsupported paired outcome: {outcome}")
    pr_ids, pr_codes = np.unique(paired["pr_id"].astype(str), return_inverse=True)
    prediction_a = paired["prediction_a"].to_numpy(dtype=bool)
    prediction_b = paired["prediction_b"].to_numpy(dtype=bool)
    if outcome == "specificity":
        prediction_a = ~prediction_a
        prediction_b = ~prediction_b
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(replicates):
        cluster_counts = rng.multinomial(
            len(pr_ids), np.full(len(pr_ids), 1.0 / len(pr_ids))
        )
        weights = cluster_counts[pr_codes]
        denominator = int(weights.sum())
        if not denominator:
            continue
        value_a = float(weights[prediction_a].sum() / denominator)
        value_b = float(weights[prediction_b].sum() / denominator)
        differences.append(value_a - value_b)
    if len(differences) < replicates * 0.95:
        raise B1ValidationError("Too many invalid paired bootstrap replicates")
    return (
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
        len(differences),
    )


def _paired_detector_frame(
    scores: pd.DataFrame,
    detector_a: str,
    detector_b: str,
    *,
    entry_type: str,
) -> pd.DataFrame:
    columns = [
        "description_id",
        "pr_id",
        "entry_type",
        "model",
        "context_code",
        "predicted_label",
    ]
    left = scores[
        scores["detector_key"].eq(detector_a)
        & scores["entry_type"].eq(entry_type)
    ][columns]
    right = scores[
        scores["detector_key"].eq(detector_b)
        & scores["entry_type"].eq(entry_type)
    ][columns]
    paired = left.merge(
        right,
        on="description_id",
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    for field in ("pr_id", "entry_type", "model", "context_code"):
        if not paired[f"{field}_a"].equals(paired[f"{field}_b"]):
            raise B1ValidationError(
                f"Paired detector metadata mismatch: {detector_a}/{detector_b}/{field}"
            )
        paired[field] = paired[f"{field}_a"]
    paired["prediction_a"] = paired["predicted_label_a"].eq("AI")
    paired["prediction_b"] = paired["predicted_label_b"].eq("AI")
    return paired[
        [
            "description_id",
            "pr_id",
            "entry_type",
            "model",
            "context_code",
            "prediction_a",
            "prediction_b",
        ]
    ]


def _paired_detector_difference_rows(
    scores: pd.DataFrame,
    *,
    replicates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recall_rows: list[dict[str, Any]] = []
    specificity_rows: list[dict[str, Any]] = []
    for modern in ANALYSIS_DETECTOR_NAMES:
        for existing in ORIGINAL_DETECTORS:
            ai = _paired_detector_frame(
                scores, modern, existing, entry_type="ai"
            )
            views: list[tuple[str, str, str, pd.DataFrame]] = [
                ("pooled", "pooled", "all", ai)
            ]
            views.extend(
                ("generator", model, "all", group)
                for model, group in ai.groupby("model", sort=True)
            )
            views.extend(
                ("generator_context", model, context, group)
                for (model, context), group in ai.groupby(
                    ["model", "context_code"], sort=True
                )
            )
            for view_level, generator, context, view in views:
                stable_seed = RUN_SEED + int(
                    hashlib.sha256(
                        "|".join(
                            (
                                "recall",
                                modern,
                                existing,
                                view_level,
                                generator,
                                context,
                            )
                        ).encode()
                    ).hexdigest()[:8],
                    16,
                )
                low, high, valid = _paired_cluster_difference_ci(
                    view,
                    outcome="recall",
                    replicates=replicates,
                    seed=stable_seed,
                )
                recall_a = float(view["prediction_a"].mean())
                recall_b = float(view["prediction_b"].mean())
                recall_rows.append(
                    {
                        "modern_detector": modern,
                        "existing_detector": existing,
                        "view_level": view_level,
                        "generator": generator,
                        "context_code": context,
                        "context": CONTEXT_LABELS.get(context, context),
                        "n_descriptions": len(view),
                        "n_prs": view["pr_id"].nunique(),
                        "modern_recall": recall_a,
                        "existing_recall": recall_b,
                        "recall_difference_modern_minus_existing": recall_a
                        - recall_b,
                        "difference_ci_low": low,
                        "difference_ci_high": high,
                        "modern_only_ai": int(
                            (view["prediction_a"] & ~view["prediction_b"]).sum()
                        ),
                        "existing_only_ai": int(
                            (~view["prediction_a"] & view["prediction_b"]).sum()
                        ),
                        "bootstrap_valid_replicates": valid,
                    }
                )

            human = _paired_detector_frame(
                scores, modern, existing, entry_type="human"
            )
            stable_seed = RUN_SEED + int(
                hashlib.sha256(
                    f"specificity|{modern}|{existing}".encode()
                ).hexdigest()[:8],
                16,
            )
            low, high, valid = _paired_cluster_difference_ci(
                human,
                outcome="specificity",
                replicates=replicates,
                seed=stable_seed,
            )
            specificity_a = float((~human["prediction_a"]).mean())
            specificity_b = float((~human["prediction_b"]).mean())
            specificity_rows.append(
                {
                    "modern_detector": modern,
                    "existing_detector": existing,
                    "n_descriptions": len(human),
                    "n_prs": human["pr_id"].nunique(),
                    "modern_specificity": specificity_a,
                    "existing_specificity": specificity_b,
                    "specificity_difference_modern_minus_existing": specificity_a
                    - specificity_b,
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "modern_only_ai": int(
                        (human["prediction_a"] & ~human["prediction_b"]).sum()
                    ),
                    "existing_only_ai": int(
                        (~human["prediction_a"] & human["prediction_b"]).sum()
                    ),
                    "bootstrap_valid_replicates": valid,
                }
            )
    return recall_rows, specificity_rows


def _agreement_rows(scores: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    detector_keys = [*ANALYSIS_DETECTOR_NAMES, *ORIGINAL_DETECTORS]
    for detector_a, detector_b in combinations(detector_keys, 2):
        paired = _paired_detector_frame(
            scores, detector_a, detector_b, entry_type="ai"
        )
        human = _paired_detector_frame(
            scores, detector_a, detector_b, entry_type="human"
        )
        all_paired = pd.concat([paired, human], ignore_index=True)
        strata: list[tuple[str, str, str, pd.DataFrame]] = [
            ("overall", "all", "all", all_paired),
            ("entry_type", "human", "all", human),
            ("entry_type", "ai", "all", paired),
        ]
        strata.extend(
            ("generator", model, "all", group)
            for model, group in paired.groupby("model", sort=True)
        )
        strata.extend(
            ("context", "all", context, group)
            for context, group in paired.groupby("context_code", sort=True)
        )
        strata.extend(
            ("generator_context", model, context, group)
            for (model, context), group in paired.groupby(
                ["model", "context_code"], sort=True
            )
        )
        for stratum_type, generator, context, frame in strata:
            a = frame["prediction_a"]
            b = frame["prediction_b"]
            rows.append(
                {
                    "detector_a": detector_a,
                    "detector_b": detector_b,
                    "stratum_type": stratum_type,
                    "generator": generator,
                    "context_code": context,
                    "context": CONTEXT_LABELS.get(context, context),
                    "n_descriptions": len(frame),
                    "n_prs": frame["pr_id"].nunique(),
                    "both_ai": int((a & b).sum()),
                    "a_only_ai": int((a & ~b).sum()),
                    "b_only_ai": int((~a & b).sum()),
                    "both_not_ai": int((~a & ~b).sum()),
                    "agreement_rate": float((a == b).mean()),
                }
            )
    return rows


def _paired_context(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, Any]:
    left = a.groupby("pr_id")["ai_score"].mean()
    right = b.groupby("pr_id")["ai_score"].mean()
    paired = pd.concat(
        [left.rename("a"), right.rename("b")], axis=1, join="inner"
    ).dropna()
    values_a = paired["a"].to_numpy(dtype=float)
    values_b = paired["b"].to_numpy(dtype=float)
    if not len(paired) or np.all(values_a == values_b):
        statistic, p_value = float("nan"), 1.0
    else:
        statistic, p_value = wilcoxon(values_a, values_b)
    return {
        "n": len(paired),
        "median_a": float(np.median(values_a)) if len(paired) else float("nan"),
        "median_b": float(np.median(values_b)) if len(paired) else float("nan"),
        "rank_biserial_a_minus_b": PS.matched_pairs_rank_biserial(values_a, values_b),
        "wilcoxon_stat": float(statistic),
        "p": float(p_value),
    }


def _rq1_rows(
    scores: pd.DataFrame,
    *,
    subset: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    friedman_rows: list[dict[str, Any]] = []
    pooled_pairwise: list[dict[str, Any]] = []
    descriptive_rows: list[dict[str, Any]] = []
    ai = scores[scores["entry_type"].eq("ai")].copy()
    for detector, detector_frame in ai.groupby("detector_key", sort=True):
        pooled = detector_frame.groupby(["pr_id", "context_code"], as_index=False)[
            "ai_score"
        ].mean()
        wide = pooled.pivot(index="pr_id", columns="context_code", values="ai_score")
        wide = wide.reindex(columns=CONTEXT_CODES).dropna()
        if wide.empty:
            raise B1ValidationError(
                f"No complete six-context cases for {detector}/{subset}"
            )
        test = friedmanchisquare(
            *(wide[context].to_numpy() for context in CONTEXT_CODES)
        )
        friedman_rows.append(
            {
                "subset": subset,
                "detector": detector,
                "aggregation": "mean across available generators per PR/context",
                "n_complete_prs": len(wide),
                "statistic": float(test.statistic),
                "df": len(CONTEXT_CODES) - 1,
                "p": float(test.pvalue),
            }
        )
        for context_a, context_b in combinations(CONTEXT_CODES, 2):
            stats = _paired_context(
                wide[context_a]
                .rename("ai_score")
                .reset_index()[["pr_id", "ai_score"]],
                wide[context_b]
                .rename("ai_score")
                .reset_index()[["pr_id", "ai_score"]],
            )
            pooled_pairwise.append(
                {
                    "subset": subset,
                    "detector": detector,
                    "context_a": CONTEXT_LABELS[context_a],
                    "context_b": CONTEXT_LABELS[context_b],
                    "sign": "positive rank-biserial means context_a is more AI-like",
                    **stats,
                }
            )
        for model, model_frame in detector_frame.groupby("model", sort=True):
            for context, context_frame in model_frame.groupby(
                "context_code", sort=True
            ):
                values = context_frame["ai_score"].to_numpy(dtype=float)
                descriptive_rows.append(
                    {
                        "subset": subset,
                        "detector": detector,
                        "model": model,
                        "context_code": context,
                        "context": CONTEXT_LABELS[context],
                        "n_descriptions": len(context_frame),
                        "n_prs": context_frame["pr_id"].nunique(),
                        "minimum": float(np.min(values)),
                        "q1": float(np.quantile(values, 0.25)),
                        "median": float(np.median(values)),
                        "q3": float(np.quantile(values, 0.75)),
                        "maximum": float(np.max(values)),
                    }
                )
    return friedman_rows, pooled_pairwise, descriptive_rows


def _length_stratum(char_count: int) -> str:
    if char_count < 250:
        return "under_250_chars"
    if char_count < 1000:
        return "250_to_999_chars"
    return "1000_or_more_chars"


def _failure_rows(manifest: pd.DataFrame, scores: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for detector in (*ANALYSIS_DETECTOR_NAMES, *ORIGINAL_DETECTORS):
        detector_scores = scores[scores["detector_key"].eq(detector)].copy()
        successful = set(detector_scores["description_id"])
        truncated = dict(
            zip(
                detector_scores["description_id"],
                _as_bool(detector_scores["truncated"]),
            )
        )
        frame = manifest.copy()
        frame["success"] = frame["description_id"].isin(successful)
        frame["truncated"] = frame["description_id"].map(truncated)
        frame["length_stratum"] = frame["char_count"].astype(int).map(_length_stratum)
        strata: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", frame)]
        for column, kind in (
            ("entry_type", "entry_type"),
            ("model", "model"),
            ("context_code", "context"),
            ("length_stratum", "length"),
        ):
            strata.extend(
                (kind, str(value) if str(value) else "not_applicable", group)
                for value, group in frame.groupby(column, dropna=False, sort=True)
            )
        for kind, label, group in strata:
            successes = int(group["success"].sum())
            rows.append(
                {
                    "detector": detector,
                    "stratum_type": kind,
                    "stratum": label,
                    "attempted": len(group),
                    "successful": successes,
                    "failures": len(group) - successes,
                    "failure_rate": (len(group) - successes) / len(group),
                    "truncated_successes": int(group["truncated"].fillna(False).sum()),
                    "truncation_rate_among_successes": _safe_divide(
                        int(group["truncated"].fillna(False).sum()), successes
                    ),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    require_approved_host(require_cuda=False)
    if args.bootstrap_replicates < 1000:
        raise B1ValidationError("Use at least 1,000 cluster-bootstrap replicates")
    repo_root = args.repo_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    verification_path = output_root / "verified" / "verification.json"
    if not verification_path.is_file():
        raise B1ValidationError("Run verify_full before B1 analysis")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "verified":
        raise B1ValidationError("B1 full verification has not passed")
    manifest_rows, preparation = load_manifest(output_root)
    manifest = pd.DataFrame(manifest_rows)
    accepted_path = output_root / "verified" / "accepted_scores.csv"
    common_path = output_root / "verified" / "strict_common_scores.csv"
    if sha256_file(accepted_path) != verification["accepted_scores_sha256"]:
        raise B1ValidationError("Accepted B1 score hash mismatch")
    if sha256_file(common_path) != verification["strict_common_scores_sha256"]:
        raise B1ValidationError("Strict-common B1 score hash mismatch")
    accepted = _read_score_table(accepted_path)
    common = _read_score_table(common_path)
    display_to_key = {
        "Binoculars": "binoculars",
        "Fast-DetectGPT": "fast-detectgpt",
    }
    for frame in (accepted, common):
        frame["detector_key"] = frame["detector_name"].map(display_to_key)
        if frame["detector_key"].isna().any():
            raise B1ValidationError("Unknown detector display name in accepted scores")
        frame["ai_score"] = pd.to_numeric(frame["ai_score"], errors="raise")
        frame["source_route"] = np.where(
            frame["detector_key"].eq("binoculars")
            & frame["detector_version"].astype(str).str.contains("macos-mps"),
            "collaborator-apple-mps",
            "internal-modern-detector-run",
        )

    original, original_provenance = _load_original_scores(repo_root, manifest)
    all_scores = pd.concat([accepted, original], ignore_index=True, sort=False)
    expected_detectors = set(ANALYSIS_DETECTOR_NAMES).union(ORIGINAL_DETECTORS)
    if set(all_scores["detector_key"].unique()) != expected_detectors:
        raise B1ValidationError(
            "Combined score table does not contain the active detector set"
        )
    successful_sets = {
        detector: set(group["description_id"].astype(str))
        for detector, group in all_scores.groupby("detector_key", sort=True)
    }
    all_common_ids = set.intersection(*successful_sets.values())
    all_common = all_scores[
        all_scores["description_id"].astype(str).isin(all_common_ids)
    ].copy()
    modern_common_ids = set(common["description_id"].astype(str))
    if all_common_ids != modern_common_ids:
        raise B1ValidationError(
            "All-detector common set differs from the verified active-detector common set"
        )
    for detector, group in all_common.groupby("detector_key", sort=True):
        if len(group) != len(all_common_ids):
            raise B1ValidationError(f"Strict-common coverage mismatch for {detector}")

    analysis_dir = output_root / "analysis"
    if analysis_dir.exists():
        raise B1ValidationError(
            f"Refusing to overwrite existing B1 analysis directory: {analysis_dir}"
        )
    performance = pd.DataFrame(
        _performance_rows(
            all_scores,
            subset="successful_only",
            replicates=args.bootstrap_replicates,
        )
        + _performance_rows(
            all_common,
            subset="strict_common_all_detectors",
            replicates=args.bootstrap_replicates,
        )
    )
    failure_rates = pd.DataFrame(_failure_rows(manifest, all_scores))
    friedman_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    descriptive_rows: list[dict[str, Any]] = []
    for subset, frame in (
        ("successful_only", accepted),
        ("strict_common_all_detectors", common),
    ):
        friedman, pooled, descriptive = _rq1_rows(frame, subset=subset)
        friedman_rows.extend(friedman)
        pooled_rows.extend(pooled)
        descriptive_rows.extend(descriptive)
    friedman = pd.DataFrame(friedman_rows)
    pooled_pairwise = PS.apply_fdr_by(
        pd.DataFrame(pooled_rows), by=["subset", "detector"], pcol="p"
    )
    family_sizes = pooled_pairwise.groupby(["subset", "detector"]).size()
    if not family_sizes.eq(15).all():
        raise B1ValidationError("Every modern-detector context FDR family must contain 15 tests")
    context_descriptive = pd.DataFrame(descriptive_rows)
    recall_rows, specificity_rows = _paired_detector_difference_rows(
        all_common, replicates=args.bootstrap_replicates
    )
    paired_recall = pd.DataFrame(recall_rows)
    paired_specificity = pd.DataFrame(specificity_rows)
    agreement = pd.DataFrame(_agreement_rows(all_common))
    analysis_dir.mkdir(parents=True, exist_ok=True)
    score_columns = [
        "description_id",
        "pr_id",
        "entry_type",
        "model",
        "context_code",
        "run",
        "detector_key",
        "detector_name",
        "detector_version",
        "ai_score",
        "predicted_label",
        "input_tokens",
        "used_input_tokens",
        "truncated",
        "success",
        "source_route",
    ]
    all_scores[score_columns].to_csv(
        analysis_dir / "all_detector_scores.csv", index=False
    )
    performance.to_csv(analysis_dir / "performance.csv", index=False)
    failure_rates.to_csv(analysis_dir / "failure_and_truncation_rates.csv", index=False)
    paired_recall.to_csv(
        analysis_dir / "paired_modern_vs_existing_recall.csv", index=False
    )
    paired_specificity.to_csv(
        analysis_dir / "paired_modern_vs_existing_specificity.csv", index=False
    )
    agreement.to_csv(analysis_dir / "pairwise_detector_agreement.csv", index=False)
    friedman.to_csv(analysis_dir / "rq1_friedman.csv", index=False)
    pooled_pairwise.to_csv(
        analysis_dir / "rq1_pooled_context_pairwise.csv", index=False
    )
    context_descriptive.to_csv(
        analysis_dir / "rq1_generator_context_descriptive.csv", index=False
    )
    metadata = {
        "run_id": RUN_ID,
        "created_at_utc": utc_now(),
        "accepted_scores_sha256": verification["accepted_scores_sha256"],
        "strict_common_scores_sha256": verification["strict_common_scores_sha256"],
        "input_manifest_sha256": preparation["input_manifest_sha256"],
        "original_detector_provenance": original_provenance,
        "detector_set": sorted(expected_detectors),
        "strict_common_detector_count": len(expected_detectors),
        "strict_common_description_count": len(all_common_ids),
        "continuous_score_policy": (
            "higher ai_score is more AI-like: -raw Binoculars ratio and the "
            "verified original-detector AI probabilities; no threshold was tuned"
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": RUN_SEED,
        "performance_ci": (
            f"95% percentile cluster bootstrap by PR; {args.bootstrap_replicates} replicates; "
            "deterministic per-view seeds derived from base seed 42"
        ),
        "rq1_friedman": "per-PR/context mean across available generators; six-context complete cases",
        "rq1_pooled_fdr_family": "15 context pairs within each subset/detector",
        "rq1_generator_context_policy": "descriptive only; no unplanned generator-specific inference",
        "paired_detector_policy": (
            "modern-minus-existing recall and specificity differences with PR-clustered "
            "percentile intervals; no row-independent pooled McNemar test"
        ),
        "failure_policy": "successful-only primary; strict common subset sensitivity; no imputation",
        "analysis_source_hashes": {
            "analyse.py": sha256_file(Path(__file__)),
            "verify_analysis.py": sha256_file(Path(__file__).with_name("verify_analysis.py")),
            "paired_stats.py": sha256_file(ANALYSIS_DIR / "paired_stats.py"),
        },
        "output_hashes": {
            path.name: sha256_file(path) for path in sorted(analysis_dir.glob("*.csv"))
        },
    }
    atomic_write_json(analysis_dir / "metadata.json", metadata)
    print(performance.to_string(index=False))
    print(friedman.to_string(index=False))
    print(f"B1 analysis outputs written to {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
