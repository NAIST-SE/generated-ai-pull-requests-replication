#!/usr/bin/env python3
"""Paired statistical analysis and claim-level sensitivity summary."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import (
    CONTEXTS,
    EXPECTED,
    PROTOCOL,
    bootstrap_median_interval,
    output_dir,
    project_root,
    require_nokogiri,
    sha256_file,
    utc_now,
    write_json,
)


STRUCTURAL_LEXICAL = {
    "word_count": "human_word_count",
    "sentence_count": "human_sentence_count",
    "flesch_kincaid_grade": "human_flesch_kincaid_grade",
    "ttr": "human_ttr",
    "mtld_original": "human_mtld_original",
    "mtld_bidir": "human_mtld_bidir",
    "lexical_density": "human_lexical_density",
}
SIMILARITY = (
    "rougeL_fmeasure",
    "bertscore_f1",
    "bm25_reciprocal_rank",
)
MAGNITUDE_TIER = {"Negligible": 0, "Small": 1, "Medium": 2, "Large": 3, "NA": -1}


def stable_seed(*values: str) -> int:
    digest = hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def sensitivity_test(
    frame: pd.DataFrame, metric: str, paired_stats: Any
) -> dict[str, Any]:
    wide = frame.pivot(index="pr_id", columns="version", values=metric).dropna()
    if not {"historical", "current"}.issubset(wide.columns):
        raise RuntimeError(f"Missing version for {metric}")
    current = wide["current"].to_numpy(float)
    historical = wide["historical"].to_numpy(float)
    result = paired_stats.paired_test(current, historical)
    difference = current - historical
    low, high = bootstrap_median_interval(
        difference,
        stable_seed(metric, str(frame["model"].iloc[0]), str(frame["variation"].iloc[0])),
    )
    return {
        "n": len(wide),
        "historical_median": float(np.median(historical)),
        "current_median": float(np.median(current)),
        "median_current_minus_historical": float(np.median(difference)),
        "median_difference_ci_low": low,
        "median_difference_ci_high": high,
        "current_vs_historical_cliffs_delta": result["cliffs_delta"],
        "current_vs_historical_rank_biserial": result["rank_biserial"],
        "wilcoxon_stat": result["wilcoxon_stat"],
        "p": result["p"],
    }


def performance(tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int]:
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def boolean_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise RuntimeError("Unexpected detector boolean encoding")
    return normalized.isin({"true", "1"}).to_numpy(bool)


def main() -> None:
    require_nokogiri()
    root = project_root(Path(__file__))
    sys.path.insert(0, str(root / "phase3/analysis"))
    import paired_stats as PS

    out = output_dir(root)
    main_metrics = pd.read_csv(out / "main_metrics.csv", dtype={"pr_id": str})
    py312 = pd.read_csv(out / "py312_metrics.csv", dtype={"pr_id": str})
    bertscore = pd.read_csv(out / "bertscore_metrics.csv", dtype={"pr_id": str})
    keys = ["model_label", "model", "pr_id", "variation", "context", "version"]
    metrics = main_metrics.merge(py312, on=keys, validate="one_to_one").merge(
        bertscore[keys + ["bertscore_f1", "candidate_truncated", "reference_truncated"]],
        on=keys,
        validate="one_to_one",
    )
    if len(metrics) != 778:
        raise RuntimeError(f"Expected 778 merged metric rows, observed {len(metrics)}")

    effect_rows: list[dict[str, Any]] = []
    for metric, human_metric in STRUCTURAL_LEXICAL.items():
        for (version, model, variation, context), group in metrics.groupby(
            ["version", "model", "variation", "context"], sort=True
        ):
            result = PS.paired_test(
                group[human_metric].to_numpy(float), group[metric].to_numpy(float)
            )
            effect_rows.append(
                {
                    "metric": metric,
                    "version": version,
                    "model": model,
                    "variation": variation,
                    "context": context,
                    **result,
                }
            )
    effects = PS.apply_fdr_by(
        pd.DataFrame(effect_rows), ["metric", "version"], pcol="p"
    )
    effects["significance"] = effects["adj_p"].apply(PS.sig_mark)
    effects_path = out / "structural_lexical_human_ai.csv"
    effects.to_csv(effects_path, index=False)

    join_keys = ["metric", "model", "variation", "context"]
    historical_effects = effects[effects["version"].eq("historical")].drop(
        columns=["version"]
    )
    current_effects = effects[effects["version"].eq("current")].drop(
        columns=["version"]
    )
    stability = historical_effects.merge(
        current_effects,
        on=join_keys,
        suffixes=("_historical", "_current"),
        validate="one_to_one",
    )
    stability["cliffs_delta_shift"] = (
        stability["cliffs_delta_current"] - stability["cliffs_delta_historical"]
    )
    stability["direction_reversal"] = (
        np.sign(stability["cliffs_delta_current"])
        != np.sign(stability["cliffs_delta_historical"])
    ) & stability["cliffs_delta_current"].ne(0) & stability[
        "cliffs_delta_historical"
    ].ne(0)
    stability["magnitude_category_change"] = stability["magnitude_current"].ne(
        stability["magnitude_historical"]
    )
    stability["magnitude_tier_shift"] = stability["magnitude_current"].map(
        MAGNITUDE_TIER
    ) - stability["magnitude_historical"].map(MAGNITUDE_TIER)
    stability["significance_change"] = stability["sig_current"].ne(
        stability["sig_historical"]
    )
    stability_path = out / "structural_lexical_stability.csv"
    stability.to_csv(stability_path, index=False)

    direct_rows: list[dict[str, Any]] = []
    for metric in list(STRUCTURAL_LEXICAL) + list(SIMILARITY):
        for (model, variation, context), group in metrics.groupby(
            ["model", "variation", "context"], sort=True
        ):
            direct_rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "variation": variation,
                    "context": context,
                    **sensitivity_test(group, metric, PS),
                }
            )
    direct = PS.apply_fdr_by(pd.DataFrame(direct_rows), "metric", pcol="p")
    direct["significance"] = direct["adj_p"].apply(PS.sig_mark)
    direct_path = out / "historical_vs_current_outcomes.csv"
    direct.to_csv(direct_path, index=False)

    ranking_rows: list[dict[str, Any]] = []
    for metric in SIMILARITY:
        summary = (
            metrics.groupby(["version", "model", "variation", "context"], as_index=False)[
                metric
            ]
            .median()
            .rename(columns={metric: "median"})
        )
        summary["rank_descending"] = summary.groupby(["version", "model"])[
            "median"
        ].rank(method="average", ascending=False)
        for row in summary.itertuples(index=False):
            ranking_rows.append({"metric": metric, **row._asdict()})
    rankings = pd.DataFrame(ranking_rows)
    rankings_path = out / "p1_p4_similarity_rankings.csv"
    rankings.to_csv(rankings_path, index=False)

    detector_scores = pd.read_csv(
        out / "detector_scores.csv", dtype={"pr_id": str}
    )
    human_scores = pd.read_csv(
        out / "human_detector_scores.csv", dtype={"pr_id": str}
    )
    detector_rows: list[dict[str, Any]] = []
    for (detector, model, variation, context), group in detector_scores.groupby(
        ["detector", "model", "variation", "context"], sort=True
    ):
        wide = group.pivot(
            index="pr_id", columns="version", values=["ai_probability", "is_ai"]
        ).dropna()
        current_probability = wide[("ai_probability", "current")].to_numpy(float)
        historical_probability = wide[("ai_probability", "historical")].to_numpy(float)
        result = PS.paired_test(current_probability, historical_probability)
        differences = current_probability - historical_probability
        low, high = bootstrap_median_interval(
            differences, stable_seed("detector", detector, model, variation)
        )
        current_flag = boolean_values(wide[("is_ai", "current")])
        historical_flag = boolean_values(wide[("is_ai", "historical")])
        historical_yes_current_no = int((historical_flag & ~current_flag).sum())
        historical_no_current_yes = int((~historical_flag & current_flag).sum())
        discordant = historical_yes_current_no + historical_no_current_yes
        mcnemar_p = (
            float(
                binomtest(
                    min(historical_yes_current_no, historical_no_current_yes),
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if discordant
            else 1.0
        )
        detector_rows.append(
            {
                "detector": detector,
                "model": model,
                "variation": variation,
                "context": context,
                "n": len(wide),
                "historical_probability_median": float(np.median(historical_probability)),
                "current_probability_median": float(np.median(current_probability)),
                "median_probability_change": float(np.median(differences)),
                "median_probability_change_ci_low": low,
                "median_probability_change_ci_high": high,
                "current_vs_historical_cliffs_delta": result["cliffs_delta"],
                "current_vs_historical_rank_biserial": result["rank_biserial"],
                "wilcoxon_stat": result["wilcoxon_stat"],
                "p": result["p"],
                "historical_flag_rate": float(historical_flag.mean()),
                "current_flag_rate": float(current_flag.mean()),
                "flag_rate_change": float(current_flag.mean() - historical_flag.mean()),
                "historical_yes_current_no": historical_yes_current_no,
                "historical_no_current_yes": historical_no_current_yes,
                "discordant": discordant,
                "mcnemar_exact_p": mcnemar_p,
            }
        )
    detector_direct = PS.apply_fdr_by(
        pd.DataFrame(detector_rows), "detector", pcol="p"
    )
    detector_direct["probability_significance"] = detector_direct["adj_p"].apply(
        PS.sig_mark
    )
    detector_direct_path = out / "detector_historical_vs_current.csv"
    detector_direct.to_csv(detector_direct_path, index=False)

    performance_rows: list[dict[str, Any]] = []
    for (detector, model, variation, context, version), group in detector_scores.groupby(
        ["detector", "model", "variation", "context", "version"], sort=True
    ):
        pair_ids = set(group["pr_id"])
        human = human_scores[
            human_scores["detector"].eq(detector) & human_scores["pr_id"].isin(pair_ids)
        ]
        if len(human) != len(pair_ids):
            raise RuntimeError(f"Human detector denominator mismatch for {detector} {model} {variation}")
        ai_flags = boolean_values(group["is_ai"])
        human_flags = boolean_values(human["is_ai"])
        values = performance(
            tp=int(ai_flags.sum()),
            fp=int(human_flags.sum()),
            tn=int((~human_flags).sum()),
            fn=int((~ai_flags).sum()),
        )
        performance_rows.append(
            {
                "detector": detector,
                "model": model,
                "variation": variation,
                "context": context,
                "version": version,
                "n_ai": len(group),
                "n_human": len(human),
                **values,
            }
        )
    detector_performance = pd.DataFrame(performance_rows)
    performance_path = out / "detector_performance.csv"
    detector_performance.to_csv(performance_path, index=False)
    perf_keys = ["detector", "model", "variation", "context"]
    historical_perf = detector_performance[
        detector_performance["version"].eq("historical")
    ].drop(columns=["version"])
    current_perf = detector_performance[
        detector_performance["version"].eq("current")
    ].drop(columns=["version"])
    performance_stability = historical_perf.merge(
        current_perf,
        on=perf_keys,
        suffixes=("_historical", "_current"),
        validate="one_to_one",
    )
    for metric in ["accuracy", "precision", "recall", "specificity", "f1"]:
        performance_stability[f"{metric}_change"] = (
            performance_stability[f"{metric}_current"]
            - performance_stability[f"{metric}_historical"]
        )
    performance_stability_path = out / "detector_performance_stability.csv"
    performance_stability.to_csv(performance_stability_path, index=False)

    top_context_changes: list[dict[str, Any]] = []
    for (metric, model), group in rankings.groupby(["metric", "model"]):
        tops = {}
        for version in ("historical", "current"):
            version_group = group[group["version"].eq(version)]
            best = version_group[version_group["median"].eq(version_group["median"].max())]
            tops[version] = sorted(best["context"].tolist())
        top_context_changes.append(
            {
                "metric": metric,
                "model": model,
                "historical_top_contexts": tops["historical"],
                "current_top_contexts": tops["current"],
                "changed": tops["historical"] != tops["current"],
            }
        )

    summary = {
        "protocol": PROTOCOL,
        "created_at_utc": utc_now(),
        "scope": {
            "models": sorted(metrics["model"].unique()),
            "contexts": [CONTEXTS[key] for key in CONTEXTS],
            "successful_pairs": int(len(metrics) / 2),
            "expected_cell_counts": {
                f"{model}|{variation}": count
                for (model, variation), count in EXPECTED.items()
            },
            "p5_p6_retested": False,
            "zerogpt_retested": False,
        },
        "structural_lexical": {
            "cells": len(stability),
            "direction_reversals": int(stability["direction_reversal"].sum()),
            "magnitude_category_changes": int(
                stability["magnitude_category_change"].sum()
            ),
            "two_or_more_tier_changes": int(
                stability["magnitude_tier_shift"].abs().ge(2).sum()
            ),
            "significance_changes": int(stability["significance_change"].sum()),
            "maximum_absolute_cliffs_delta_shift": float(
                stability["cliffs_delta_shift"].abs().max()
            ),
        },
        "p1_p4_similarity_top_context_changes": top_context_changes,
        "detectors": {
            "cells": len(detector_direct),
            "total_binary_flips": int(detector_direct["discordant"].sum()),
            "maximum_absolute_flag_rate_change": float(
                detector_direct["flag_rate_change"].abs().max()
            ),
            "maximum_absolute_f1_change": float(
                performance_stability["f1_change"].abs().max()
            ),
        },
        "interpretation_guardrails": [
            "This is a diagnostic on a frozen 50-PR sample, not an equivalence test or replacement for the 308-PR study.",
            "The 11 missing Qwen P4 cases are non-random with respect to prompt length and were not replaced or imputed.",
            "Only P1-P4 were regenerated; six-context and P5/P6 claims are not retested.",
            "Significance changes are secondary to directions, effect sizes, continuous changes, and detector flag/F1 changes.",
        ],
    }
    write_json(out / "sensitivity_summary.json", summary)

    changed_rankings = sum(item["changed"] for item in top_context_changes)
    report = f"""# Generation-session outcome-sensitivity results

Status: analysis complete; final integrity verification is performed separately.

## Scope

- 389 successful per-PR/model/context pairs, each evaluated historically and currently.
- GPT: 50 PRs in each P1--P4 cell; Qwen: 50 in P1--P3 and 39 in P4.
- Structural/lexical RQ2 outcomes, ROUGE-L, BERTScore, fixed-corpus BM25, and the two pinned local RoBERTa detectors.
- ZeroGPT and P5/P6 were not rerun. This diagnostic cannot retest six-context or P5/P6 claims.

## Mechanical sensitivity flags

- Structural/lexical cells: {len(stability)}.
- Cliff-delta direction reversals: {int(stability['direction_reversal'].sum())}.
- Cliff-delta magnitude-category changes: {int(stability['magnitude_category_change'].sum())}.
- Changes of at least two magnitude tiers: {int(stability['magnitude_tier_shift'].abs().ge(2).sum())}.
- BH-adjusted significance-status changes: {int(stability['significance_change'].sum())} (secondary diagnostic only).
- P1--P4 top-context changes across the six model-by-similarity-metric comparisons: {changed_rankings}.
- Detector binary flips across all paired AI descriptions and both detectors: {int(detector_direct['discordant'].sum())}.
- Largest absolute detector flag-rate change in one detector/model/context cell: {detector_direct['flag_rate_change'].abs().max():.3f}.
- Largest absolute detector F1 change in one detector/model/context cell: {performance_stability['f1_change'].abs().max():.3f}.

These counts are transparent flags, not post-hoc equivalence thresholds. Interpret the
underlying continuous estimates and tables before deciding whether a manuscript claim is
stable. The diagnostic cannot identify whether any difference is caused by nondeterminism,
provider/backend evolution, or another generation-protocol detail.
"""
    report_path = out / "SENSITIVITY_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    outputs = [
        effects_path,
        stability_path,
        direct_path,
        rankings_path,
        detector_direct_path,
        performance_path,
        performance_stability_path,
        out / "sensitivity_summary.json",
        report_path,
    ]
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": utc_now(),
        "stage": "analysis",
        "merged_metric_rows": len(metrics),
        "output_sha256": {path.name: sha256_file(path) for path in outputs},
        "raw_text_in_outputs": False,
    }
    write_json(out / "analysis_metadata.json", payload)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
