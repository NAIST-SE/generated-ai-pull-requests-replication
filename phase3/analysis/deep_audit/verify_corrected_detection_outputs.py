#!/usr/bin/env python3
"""Independently recompute and verify corrected Preliminary-RQ/RQ1 outputs.

SERVER ONLY. This verifier is pinned to the finalized corrected detector corpus and
recomputes every detector metric and inferential result reported for Preliminary-RQ and
RQ1. It intentionally does not import the production analysis scripts or paired_stats.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, friedmanchisquare, rankdata, wilcoxon

from server_guard import require_nokogiri


EXPECTED_INPUT_SHA256 = "b995b02848b4e7487682379afa22c6b7d6a8a06a404f35dac3ebca60daf9aa91"
EXPECTED_CORPUS_SHA256 = "07136384785d6d02d17762535ab51a2007a7a47b8c41b77fbb311ce9c8780306"
MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "gpt-5-mini-2025-08-07",
    "meta-llama/llama-3.1-70b-instruct",
]
CONTEXTS = {
    "P-1": "Repo Path",
    "P-2": "PR Title",
    "P-3": "File Diff",
    "P-4": "File Contents",
    "P-5": "Linked Issue",
    "P-6": "Without File Contents",
}
EXPECTED_CELL_COUNTS = {
    "Qwen/Qwen2.5-72B-Instruct": {
        "P-1": 308,
        "P-2": 308,
        "P-3": 308,
        "P-4": 302,
        "P-5": 273,
        "P-6": 273,
    },
    "gpt-5-mini-2025-08-07": {
        "P-1": 308,
        "P-2": 308,
        "P-3": 308,
        "P-4": 302,
        "P-5": 273,
        "P-6": 273,
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "P-1": 308,
        "P-2": 308,
        "P-3": 308,
        "P-4": 308,
        "P-5": 273,
        "P-6": 273,
    },
}
DETECTORS = {
    "ZeroGPT": ("zerogpt_is_ai", "zerogpt_ai_probability", "zerogpt_success"),
    "ChatGPT Detector RoBERTa": (
        "roberta_is_ai",
        "roberta_ai_probability",
        "roberta_success",
    ),
    "ChatGPT-QA Detector RoBERTa": (
        "roberta-qa_is_ai",
        "roberta-qa_ai_probability",
        "roberta-qa_success",
    ),
}
EXPECTED_DETECTOR_CONFIG = {
    "ZeroGPT": {
        "prefix": "zerogpt",
        "model": "ZeroGPT proprietary service",
        "revision": "not exposed by response",
        "input_mode": "single",
        "truncated": 0,
        "truncated_by_source": {
            "Human": 0,
            "Qwen/Qwen2.5-72B-Instruct": 0,
            "gpt-5-mini-2025-08-07": 0,
            "meta-llama/llama-3.1-70b-instruct": 0,
        },
    },
    "ChatGPT Detector RoBERTa": {
        "prefix": "roberta",
        "model": "Hello-SimpleAI/chatgpt-detector-roberta",
        "revision": "d2b342c61775d5dd0221808a79983ed3b86ffd86",
        "input_mode": "single",
        "truncated": 274,
        "truncated_by_source": {
            "Human": 16,
            "Qwen/Qwen2.5-72B-Instruct": 1,
            "gpt-5-mini-2025-08-07": 257,
            "meta-llama/llama-3.1-70b-instruct": 0,
        },
    },
    "ChatGPT-QA Detector RoBERTa": {
        "prefix": "roberta-qa",
        "model": "Hello-SimpleAI/chatgpt-qa-detector-roberta",
        "revision": "25aad95e922d92631f86164642a3db104deee087",
        "input_mode": "pair",
        "truncated": 302,
        "truncated_by_source": {
            "Human": 19,
            "Qwen/Qwen2.5-72B-Instruct": 1,
            "gpt-5-mini-2025-08-07": 282,
            "meta-llama/llama-3.1-70b-instruct": 0,
        },
    },
}
FLOAT_TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def close(actual: float, expected: float, label: str) -> None:
    if np.isnan(actual) and np.isnan(expected):
        return
    if not np.isclose(actual, expected, rtol=FLOAT_TOLERANCE, atol=FLOAT_TOLERANCE):
        raise RuntimeError(f"{label}: expected {expected!r}, observed {actual!r}")


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    comparisons = np.sign(
        np.asarray(a, dtype=float)[:, None] - np.asarray(b, dtype=float)[None, :]
    )
    return float(comparisons.sum() / comparisons.size)


def delta_magnitude(delta: float) -> str:
    absolute = abs(delta)
    if absolute < 0.147:
        return "Negligible"
    if absolute < 0.33:
        return "Small"
    if absolute < 0.474:
        return "Medium"
    return "Large"


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    differences = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    differences = differences[differences != 0]
    if len(differences) == 0:
        return float("nan")
    ranks = rankdata(np.abs(differences))
    total = len(differences) * (len(differences) + 1) / 2
    return float(
        (ranks[differences > 0].sum() - ranks[differences < 0].sum()) / total
    )


def sig_mark(adjusted_p: float) -> str:
    if adjusted_p < 1e-3:
        return "***"
    if adjusted_p < 1e-2:
        return "**"
    if adjusted_p < 0.05:
        return "*"
    return "n.s."


def cochrans_q_independent(matrix: np.ndarray) -> tuple[float, float]:
    matrix = np.asarray(matrix, dtype=float)
    k = matrix.shape[1]
    column_sums = matrix.sum(axis=0)
    row_sums = matrix.sum(axis=1)
    total = column_sums.sum()
    denominator = k * total - np.square(row_sums).sum()
    require(denominator > 0, "Cochran Q denominator is not positive")
    statistic = (
        (k - 1)
        * (k * np.square(column_sums).sum() - total**2)
        / denominator
    )
    return float(statistic), float(chi2.sf(statistic, k - 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_and_verify_input(
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    completion = json.loads((run_dir / "completion.json").read_text())
    require(
        completion["status"] == "complete_and_verified", "Detector run is not verified"
    )
    require(
        completion["detection_results_sha256"] == EXPECTED_INPUT_SHA256,
        "Completion manifest is not pinned to the finalized detector result",
    )
    require(
        completion["corpus_sha256"] == EXPECTED_CORPUS_SHA256,
        "Completion manifest has an unexpected corpus hash",
    )
    require(
        all(completion["integrity_gates"].values()),
        "A completion integrity gate failed",
    )

    input_path = run_dir / "detection_results.csv"
    require(
        sha256(input_path) == EXPECTED_INPUT_SHA256, "Final detector CSV hash mismatch"
    )
    require(
        sha256(run_dir / "corpus.csv") == EXPECTED_CORPUS_SHA256,
        "Corpus hash mismatch",
    )
    frame = pd.read_csv(input_path)
    frame["entry_type"] = frame["entry_type"].astype(str).str.lower()
    ai = frame[frame["entry_type"].eq("ai")].copy()
    human = frame[frame["entry_type"].eq("human")].copy()
    ai["pr_id"] = ai["pr_id"].astype(str).str.strip()
    ai["model"] = ai["model"].astype(str).str.strip()
    ai["variation"] = ai["variation"].astype(str).str.strip()
    require(len(frame) == 5630, f"Expected 5,630 rows, observed {len(frame)}")
    require(len(ai) == 5322, f"Expected 5,322 AI rows, observed {len(ai)}")
    require(len(human) == 308, f"Expected 308 human rows, observed {len(human)}")
    require(
        not ai.duplicated(["pr_id", "model", "variation"]).any(), "Duplicate AI keys"
    )
    require(
        frame["text"].fillna("").astype(str).str.strip().ne("").all(), "Empty text"
    )
    require(set(ai["model"]) == set(MODELS), "Unexpected model set")
    require(set(ai["variation"]) == set(CONTEXTS), "Unexpected context set")

    counts = ai.groupby(["model", "variation"]).size()
    for model, expected_contexts in EXPECTED_CELL_COUNTS.items():
        for context, expected in expected_contexts.items():
            require(
                int(counts.loc[(model, context)]) == expected,
                f"Unexpected cell size for {model}, {context}",
            )

    for detector, (prediction_col, probability_col, success_col) in DETECTORS.items():
        require(as_bool(frame[success_col]).all(), f"{detector} has a failed call")
        probabilities = pd.to_numeric(frame[probability_col], errors="coerce")
        require(probabilities.notna().all(), f"{detector} has a missing probability")
        require(
            probabilities.between(0, 1).all(),
            f"{detector} probability outside [0,1]",
        )
        require(prediction_col in frame, f"{detector} flag column is missing")
    return frame, ai, human


def verify_metadata(output_root: Path) -> None:
    for metadata_path in sorted(output_root.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        require(
            metadata.get("input_sha256") == EXPECTED_INPUT_SHA256,
            f"Input hash mismatch in {metadata_path}",
        )


def verify_detector_input_diagnostics(frame: pd.DataFrame) -> dict:
    source = frame["model"].where(frame["entry_type"].eq("ai"), "Human")
    diagnostics = {}
    for detector, expected in EXPECTED_DETECTOR_CONFIG.items():
        prefix = expected["prefix"]
        model_values = frame[f"{prefix}_model"].dropna().astype(str).unique().tolist()
        revision_values = (
            frame[f"{prefix}_revision"].dropna().astype(str).unique().tolist()
        )
        input_mode_values = (
            frame[f"{prefix}_input_mode"].dropna().astype(str).unique().tolist()
        )
        require(model_values == [expected["model"]], f"Unexpected model for {detector}")
        require(
            revision_values == [expected["revision"]],
            f"Unexpected revision for {detector}",
        )
        require(
            input_mode_values == [expected["input_mode"]],
            f"Unexpected input mode for {detector}",
        )
        truncated = as_bool(frame[f"{prefix}_input_truncated"])
        truncated_by_source = {
            source_name: int(truncated[source.eq(source_name)].sum())
            for source_name in ["Human", *MODELS]
        }
        require(
            int(truncated.sum()) == expected["truncated"],
            f"Unexpected truncation count for {detector}",
        )
        require(
            truncated_by_source == expected["truncated_by_source"],
            f"Unexpected truncation distribution for {detector}",
        )
        input_tokens = pd.to_numeric(
            frame[f"{prefix}_input_token_count"], errors="coerce"
        )
        used_tokens = pd.to_numeric(
            frame[f"{prefix}_used_token_count"], errors="coerce"
        )
        diagnostics[detector] = {
            "model": expected["model"],
            "revision": expected["revision"],
            "input_mode": expected["input_mode"],
            "successful": int(as_bool(frame[f"{prefix}_success"]).sum()),
            "failed": int((~as_bool(frame[f"{prefix}_success"])).sum()),
            "truncated": int(truncated.sum()),
            "truncated_pct_of_5630": float(truncated.mean() * 100),
            "truncated_by_source": truncated_by_source,
            "max_input_tokens": (
                int(input_tokens.max()) if input_tokens.notna().any() else None
            ),
            "max_used_tokens": (
                int(used_tokens.max()) if used_tokens.notna().any() else None
            ),
        }
    return diagnostics


def independently_verify_descriptive(
    ai: pd.DataFrame, human: pd.DataFrame, output_root: Path
) -> dict:
    output_dir = output_root / "detection_descriptive"
    metrics = pd.read_csv(output_dir / "detector_metrics_by_view_model_context.csv")
    prelim = pd.read_csv(output_dir / "prelim_file_diff_metrics.csv")
    extrema = pd.read_csv(output_dir / "rq1_f1_extrema.csv")
    require(len(metrics) == 108, "Expected 108 descriptive rows")
    require(len(prelim) == 9, "Expected nine Preliminary metric rows")
    require(len(extrema) == 9, "Expected nine RQ1 extrema rows")

    equality_columns = [column for column in metrics.columns if column != "view"]
    operational = metrics[metrics["view"].eq("operational")].sort_values(
        ["detector", "model", "variation"]
    )
    successful = metrics[metrics["view"].eq("successful_only")].sort_values(
        ["detector", "model", "variation"]
    )
    pd.testing.assert_frame_equal(
        operational[equality_columns].reset_index(drop=True),
        successful[equality_columns].reset_index(drop=True),
        check_dtype=False,
        atol=FLOAT_TOLERANCE,
        rtol=FLOAT_TOLERANCE,
    )

    recomputed = []
    for detector, (prediction_col, _, _) in DETECTORS.items():
        human_prediction = as_bool(human[prediction_col])
        for model in MODELS:
            for variation, context in CONTEXTS.items():
                ai_cell = ai[
                    ai["model"].eq(model) & ai["variation"].eq(variation)
                ]
                ai_prediction = as_bool(ai_cell[prediction_col])
                tp = int(ai_prediction.sum())
                fn = int((~ai_prediction).sum())
                fp = int(human_prediction.sum())
                tn = int((~human_prediction).sum())
                total = tp + fn + fp + tn
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                recomputed.append(
                    {
                        "detector": detector,
                        "model": model,
                        "variation": variation,
                        "context": context,
                        "n_human": len(human),
                        "n_ai": len(ai_cell),
                        "human_detector_failures": 0,
                        "ai_detector_failures": 0,
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        "accuracy": (tp + tn) / total,
                        "precision": precision,
                        "recall": recall,
                        "f1": (
                            2 * precision * recall / (precision + recall)
                            if precision + recall
                            else 0.0
                        ),
                        "specificity": tn / (tn + fp),
                    }
                )
    expected = pd.DataFrame(recomputed)
    merged = operational.merge(
        expected,
        on=["detector", "model", "variation", "context"],
        suffixes=("_observed", "_expected"),
        validate="one_to_one",
    )
    numeric_columns = [
        "n_human",
        "n_ai",
        "human_detector_failures",
        "ai_detector_failures",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
    ]
    for column in numeric_columns:
        for index, row in merged.iterrows():
            close(
                float(row[f"{column}_observed"]),
                float(row[f"{column}_expected"]),
                f"descriptive {index} {column}",
            )

    observed_prelim = prelim.drop(columns=["view"])
    pd.testing.assert_frame_equal(
        observed_prelim.sort_values(["detector", "model"]).reset_index(drop=True),
        operational[operational["variation"].eq("P-3")]
        .drop(columns=["view"])
        .sort_values(["detector", "model"])
        .reset_index(drop=True),
        check_dtype=False,
    )

    for _, observed in extrema.iterrows():
        cell = expected[
            expected["detector"].eq(observed["detector"])
            & expected["model"].eq(observed["model"])
        ]
        best = float(cell["f1"].max())
        worst = float(cell["f1"].min())
        close(float(observed["best_f1"]), best, "RQ1 best F1")
        close(float(observed["worst_f1"]), worst, "RQ1 worst F1")
        require(
            observed["best_contexts"]
            == "; ".join(cell.loc[cell["f1"].eq(best), "context"]),
            "RQ1 best context mismatch",
        )
        require(
            observed["worst_contexts"]
            == "; ".join(cell.loc[cell["f1"].eq(worst), "context"]),
            "RQ1 worst context mismatch",
        )
    return {
        "all_54_cells_recomputed": True,
        "operational_equals_successful_only": True,
        "prelim_metric_cells": 9,
        "rq1_extrema_cells": 9,
    }


def independently_verify_prelim(ai: pd.DataFrame, output_root: Path) -> dict:
    output_dir = output_root / "prelim"
    omnibus = pd.read_csv(output_dir / "prelim_cochrans_q.csv")
    pairs = pd.read_csv(output_dir / "prelim_mcnemar_pairwise.csv")
    require(len(omnibus) == 6, "Expected six Preliminary omnibus rows")
    require(len(pairs) == 18, "Expected 18 Preliminary pairwise rows")

    for frame, keys in [
        (omnibus, ["detector"]),
        (pairs, ["detector", "model_a", "model_b"]),
    ]:
        operational = frame[frame["view"].eq("operational")].sort_values(keys)
        successful = frame[frame["view"].eq("successful_only")].sort_values(keys)
        columns = [column for column in frame.columns if column != "view"]
        pd.testing.assert_frame_equal(
            operational[columns].reset_index(drop=True),
            successful[columns].reset_index(drop=True),
            check_dtype=False,
            atol=FLOAT_TOLERANCE,
            rtol=FLOAT_TOLERANCE,
        )

    file_diff = ai[ai["variation"].eq("P-3")].copy()
    expected_omnibus = []
    expected_pairs = []
    for detector, (prediction_col, _, _) in DETECTORS.items():
        wide = file_diff.pivot(index="pr_id", columns="model", values=prediction_col)
        wide = wide.reindex(columns=MODELS).apply(as_bool).astype(int).dropna()
        require(len(wide) == 308, f"Preliminary pairing is not n=308 for {detector}")
        statistic, p_value = cochrans_q_independent(wide.to_numpy())
        expected_omnibus.append((detector, statistic, p_value, wide))
        for model_a, model_b in combinations(MODELS, 2):
            a = wide[model_a]
            b = wide[model_b]
            a1_b0 = int(((a == 1) & (b == 0)).sum())
            a0_b1 = int(((a == 0) & (b == 1)).sum())
            discordant = a1_b0 + a0_b1
            p_pair = float(
                binomtest(min(a1_b0, a0_b1), discordant, 0.5).pvalue
            )
            expected_pairs.append(
                {
                    "detector": detector,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_prs": 308,
                    "rate_a": float(a.mean()),
                    "rate_b": float(b.mean()),
                    "paired_rate_difference_a_minus_b": float((a - b).mean()),
                    "discordant_a1_b0": a1_b0,
                    "discordant_a0_b1": a0_b1,
                    "mcnemar_stat": float(min(a1_b0, a0_b1)),
                    "p": p_pair,
                }
            )

    observed_omnibus = omnibus[omnibus["view"].eq("operational")]
    for detector, statistic, p_value, wide in expected_omnibus:
        observed = observed_omnibus[
            observed_omnibus["detector"].eq(detector)
        ].iloc[0]
        require(int(observed["n_prs"]) == 308, "Preliminary omnibus n mismatch")
        close(float(observed["q_stat"]), statistic, f"{detector} Cochran Q")
        close(float(observed["p"]), p_value, f"{detector} Cochran p")
        for model in MODELS:
            close(
                float(observed[f"rate__{model}"]),
                float(wide[model].mean()),
                "flag rate",
            )

    expected_pair_frame = pd.DataFrame(expected_pairs)
    expected_pair_frame["adj_p"] = bh_adjust(expected_pair_frame["p"].to_numpy())
    expected_pair_frame["sig"] = expected_pair_frame["adj_p"].lt(0.05)
    observed_pairs = pairs[pairs["view"].eq("operational")]
    merged = observed_pairs.merge(
        expected_pair_frame,
        on=["detector", "model_a", "model_b"],
        suffixes=("_observed", "_expected"),
        validate="one_to_one",
    )
    for column in [
        "n_prs",
        "rate_a",
        "rate_b",
        "paired_rate_difference_a_minus_b",
        "discordant_a1_b0",
        "discordant_a0_b1",
        "mcnemar_stat",
        "p",
        "adj_p",
    ]:
        for index, row in merged.iterrows():
            close(
                float(row[f"{column}_observed"]),
                float(row[f"{column}_expected"]),
                f"McNemar {index} {column}",
            )
    require(
        (
            as_bool(merged["sig_observed"])
            == as_bool(merged["sig_expected"])
        ).all(),
        "McNemar adjusted-significance mismatch",
    )
    return {
        "cochran_tests_recomputed": 3,
        "mcnemar_tests_recomputed": 9,
        "bh_family_size": 9,
        "operational_equals_successful_only": True,
    }


def independently_verify_rq1(ai: pd.DataFrame, output_root: Path) -> dict:
    output_dir = output_root / "rq1"
    friedman = pd.read_csv(output_dir / "rq1_friedman.csv")
    pairs = pd.read_csv(output_dir / "rq1_model_context_pairwise.csv")
    robustness = pd.read_csv(output_dir / "rq1_operational_vs_successful_only.csv")
    require(len(friedman) == 6, "Expected six RQ1 Friedman rows")
    require(len(pairs) == 270, "Expected 270 RQ1 paired rows")
    require(len(robustness) == 135, "Expected 135 RQ1 view-comparison rows")

    for frame, keys in [
        (friedman, ["detector"]),
        (pairs, ["detector", "model", "context_a", "context_b"]),
    ]:
        operational = frame[frame["view"].eq("operational")].sort_values(keys)
        successful = frame[frame["view"].eq("successful_only")].sort_values(keys)
        columns = [column for column in frame.columns if column != "view"]
        pd.testing.assert_frame_equal(
            operational[columns].reset_index(drop=True),
            successful[columns].reset_index(drop=True),
            check_dtype=False,
            atol=FLOAT_TOLERANCE,
            rtol=FLOAT_TOLERANCE,
        )
    require(
        not as_bool(robustness["direction_reversal"]).any(),
        "Direction reversal remains",
    )
    require(
        not as_bool(robustness["significance_change"]).any(),
        "Significance change remains",
    )
    require(
        (robustness["n_operational"] == robustness["n_successful_only"]).all(),
        "RQ1 view sample sizes differ",
    )

    expected_friedman = []
    expected_pairs = []
    for detector, (_, probability_col, _) in DETECTORS.items():
        detector_ai = ai[["pr_id", "model", "variation", probability_col]].copy()
        detector_ai["probability"] = pd.to_numeric(
            detector_ai[probability_col], errors="raise"
        )
        pooled = detector_ai.groupby(["pr_id", "variation"])[
            "probability"
        ].mean().unstack()
        complete = pooled.reindex(columns=list(CONTEXTS)).dropna()
        result = friedmanchisquare(*(complete[context] for context in CONTEXTS))
        expected_friedman.append(
            {
                "detector": detector,
                "n_complete_prs": len(complete),
                "statistic": float(result.statistic),
                "p": float(result.pvalue),
            }
        )

        for model in MODELS:
            model_frame = detector_ai[detector_ai["model"].eq(model)]
            for context_a, context_b in combinations(CONTEXTS, 2):
                a = model_frame[model_frame["variation"].eq(context_a)].set_index(
                    "pr_id"
                )["probability"]
                b = model_frame[model_frame["variation"].eq(context_b)].set_index(
                    "pr_id"
                )["probability"]
                aligned = pd.concat(
                    [a.rename("a"), b.rename("b")], axis=1, join="inner"
                ).dropna()
                a_values = aligned["a"].to_numpy()
                b_values = aligned["b"].to_numpy()
                differences = a_values - b_values
                if np.all(differences == 0):
                    statistic, p_value = float("nan"), 1.0
                else:
                    result_pair = wilcoxon(a_values, b_values)
                    statistic = float(result_pair.statistic)
                    p_value = float(result_pair.pvalue)
                expected_pairs.append(
                    {
                        "detector": detector,
                        "model": model,
                        "context_a": CONTEXTS[context_a],
                        "context_b": CONTEXTS[context_b],
                        "n": len(aligned),
                        "median_human": float(np.median(a_values)),
                        "median_ai": float(np.median(b_values)),
                        "cliffs_delta": cliffs_delta(a_values, b_values),
                        "rank_biserial": rank_biserial(a_values, b_values),
                        "wilcoxon_stat": statistic,
                        "p": p_value,
                    }
                )

    observed_friedman = friedman[friedman["view"].eq("operational")]
    for expected in expected_friedman:
        observed = observed_friedman[
            observed_friedman["detector"].eq(expected["detector"])
        ].iloc[0]
        require(
            int(observed["n_complete_prs"]) == expected["n_complete_prs"],
            "Friedman n mismatch",
        )
        close(float(observed["statistic"]), expected["statistic"], "Friedman statistic")
        close(float(observed["p"]), expected["p"], "Friedman p")

    expected_pair_frame = pd.DataFrame(expected_pairs)
    expected_parts = []
    for _, family in expected_pair_frame.groupby(["detector", "model"], sort=False):
        family = family.copy()
        family["adj_p"] = bh_adjust(family["p"].to_numpy())
        family["sig"] = family["adj_p"].lt(0.05)
        family["Sig"] = family["adj_p"].apply(sig_mark)
        family["magnitude"] = family["cliffs_delta"].apply(delta_magnitude)
        expected_parts.append(family)
    expected_pair_frame = pd.concat(expected_parts, ignore_index=True)
    observed_pairs = pairs[pairs["view"].eq("operational")]
    keys = ["detector", "model", "context_a", "context_b"]
    merged = observed_pairs.merge(
        expected_pair_frame,
        on=keys,
        suffixes=("_observed", "_expected"),
        validate="one_to_one",
    )
    for column in [
        "n",
        "median_human",
        "median_ai",
        "cliffs_delta",
        "rank_biserial",
        "wilcoxon_stat",
        "p",
        "adj_p",
    ]:
        for index, row in merged.iterrows():
            close(
                float(row[f"{column}_observed"]),
                float(row[f"{column}_expected"]),
                f"RQ1 pair {index} {column}",
            )
    for column in ["sig", "Sig", "magnitude"]:
        require(
            merged[f"{column}_observed"].astype(str).tolist()
            == merged[f"{column}_expected"].astype(str).tolist(),
            f"RQ1 pair {column} mismatch",
        )
    return {
        "friedman_tests_recomputed": 3,
        "paired_contrasts_recomputed": 135,
        "bh_families": 9,
        "operational_equals_successful_only": True,
        "complete_prs_by_detector": {
            row["detector"]: int(row["n_complete_prs"])
            for row in expected_friedman
        },
    }


def main() -> None:
    require_nokogiri()
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_root = args.output_root.resolve()
    frame, ai, human = load_and_verify_input(run_dir)
    verify_metadata(output_root)
    contracts = {
        "detector_inputs": verify_detector_input_diagnostics(frame),
        "descriptive": independently_verify_descriptive(ai, human, output_root),
        "preliminary_inference": independently_verify_prelim(ai, output_root),
        "rq1_inference": independently_verify_rq1(ai, output_root),
    }

    figure_dir = output_root / "figures"
    figure_metadata = json.loads((figure_dir / "metadata.json").read_text())
    pdf_path = figure_dir / "rq1_probability_boxplots_paired.pdf"
    png_path = figure_dir / "rq1_probability_boxplots_paired.png"
    require(pdf_path.stat().st_size > 0, "Missing/empty RQ1 PDF")
    require(png_path.stat().st_size > 0, "Missing/empty RQ1 PNG")
    require(
        sha256(pdf_path) == figure_metadata["pdf_sha256"], "RQ1 PDF hash mismatch"
    )
    require(
        sha256(png_path) == figure_metadata["png_sha256"], "RQ1 PNG hash mismatch"
    )

    output_hashes = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "verification_report.json":
            output_hashes[str(path.relative_to(output_root))] = sha256(path)
    report = {
        "status": "PASS",
        "verifier": "independent full recomputation of Preliminary-RQ and RQ1",
        "input_sha256": EXPECTED_INPUT_SHA256,
        "corpus_sha256": EXPECTED_CORPUS_SHA256,
        "rows": {"ai": 5322, "human": 308, "total": 5630},
        "ai_rows_by_model_context": EXPECTED_CELL_COUNTS,
        "all_detector_calls_successful": True,
        "contracts": contracts,
        "figure_sha256": {
            "pdf": sha256(pdf_path),
            "png": sha256(png_path),
        },
        "output_sha256": output_hashes,
    }
    report_path = output_root / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
