#!/usr/bin/env python3
"""Build and summarize the no-API six-context session bridge."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


PROTOCOL = "generation-session-six-context-bridge-v1"
EXPECTED_COMPLETE = {"gpt": 50, "qwen": 39}
P14 = ("P-1", "P-2", "P-3", "P-4")
P56 = ("P-5", "P-6")
ALL_CONTEXTS = P14 + P56
VERSIONS = ("historical_bridge", "current_bridge")
FORBIDDEN_COLUMNS = {
    "text",
    "prompt",
    "full_prompt",
    "generated_text",
    "human_text",
    "title",
    "description",
    "body",
}


def project_root(script: Path) -> Path:
    for candidate in (script.resolve(), *script.resolve().parents):
        if (candidate / ".git").exists() and (candidate / "phase3").is_dir():
            return candidate
    raise RuntimeError("Could not locate llm-in-prs repository root")


def require_nokogiri() -> None:
    hostname = platform.node().lower()
    if "nokogiri" not in hostname:
        raise RuntimeError(f"This analysis must run on Nokogiri, not {hostname!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def exact_one(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        example = frame.loc[duplicated, keys].head(5).to_dict("records")
        raise RuntimeError(f"{label} has duplicate keys: {example}")


def aggregate_value(values: pd.Series, aggregation: str) -> float:
    clean = values.dropna().astype(float)
    if clean.empty:
        raise RuntimeError("Cannot aggregate an empty metric cell")
    if aggregation == "median":
        return float(clean.median())
    if aggregation == "mean":
        return float(clean.mean())
    raise ValueError(f"Unknown aggregation {aggregation!r}")


def complete_ids(manifest: pd.DataFrame) -> dict[str, set[str]]:
    exact_one(manifest, ["model_label", "pr_id", "variation"], "pair manifest")
    if not set(manifest["variation"]).issubset(P14):
        raise RuntimeError("Pair manifest contains a context outside P1--P4")
    result: dict[str, set[str]] = {}
    for model_label, group in manifest.groupby("model_label", sort=True):
        counts = group.groupby("pr_id")["variation"].nunique()
        ids = set(counts[counts.eq(len(P14))].index.astype(str))
        expected = EXPECTED_COMPLETE.get(str(model_label))
        if expected is None:
            raise RuntimeError(f"Unexpected model label {model_label!r}")
        if len(ids) != expected:
            raise RuntimeError(
                f"{model_label}: expected {expected} complete P1--P4 PRs, observed {len(ids)}"
            )
        result[str(model_label)] = ids
    if set(result) != set(EXPECTED_COMPLETE):
        raise RuntimeError(f"Expected model labels {sorted(EXPECTED_COMPLETE)}, got {sorted(result)}")
    return result


def validate_matrix(
    frame: pd.DataFrame,
    model_label: str,
    expected_ids: set[str],
    variations: tuple[str, ...],
    label: str,
) -> None:
    subset = frame[frame["model_label"].eq(model_label)]
    exact_one(subset, ["pr_id", "variation"], label)
    observed_ids = set(subset["pr_id"].astype(str))
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)[:5]
        extra = sorted(observed_ids - expected_ids)[:5]
        raise RuntimeError(f"{label}: PR coverage mismatch; missing={missing}, extra={extra}")
    expected_rows = len(expected_ids) * len(variations)
    if len(subset) != expected_rows:
        raise RuntimeError(f"{label}: expected {expected_rows} rows, observed {len(subset)}")
    counts = subset.groupby("pr_id")["variation"].nunique()
    if not counts.eq(len(variations)).all() or set(subset["variation"]) != set(variations):
        raise RuntimeError(f"{label}: incomplete context matrix")


def prepare_p14(
    frame: pd.DataFrame,
    value_column: str,
    metric: str,
    aggregation: str,
) -> pd.DataFrame:
    required = {"model_label", "model", "pr_id", "variation", "context", "version", value_column}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"P1--P4 source for {metric} lacks {sorted(missing)}")
    result = frame.loc[
        frame["variation"].isin(P14) & frame["version"].isin(("historical", "current")),
        list(required),
    ].copy()
    result = result.rename(columns={value_column: "value"})
    result["bridge_version"] = result["version"].map(
        {"historical": "historical_bridge", "current": "current_bridge"}
    )
    result["metric"] = metric
    result["aggregation"] = aggregation
    result["source_segment"] = "session-specific P1-P4"
    return result[
        [
            "model_label",
            "model",
            "pr_id",
            "variation",
            "context",
            "bridge_version",
            "metric",
            "aggregation",
            "value",
            "source_segment",
        ]
    ]


def prepare_p56(
    frame: pd.DataFrame,
    value_column: str,
    metric: str,
    aggregation: str,
    model_map: dict[str, str],
) -> pd.DataFrame:
    id_column = "normalized_pr_id"
    required = {id_column, "model", "variation", value_column}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"P5--P6 source for {metric} lacks {sorted(missing)}")
    result = frame.loc[frame["variation"].isin(P56), list(required) + [c for c in ("context", "Context_Name") if c in frame.columns]].copy()
    if "entry_type" in frame.columns:
        result = frame.loc[
            frame["entry_type"].eq("ai") & frame["variation"].isin(P56),
            list(required) + [c for c in ("context", "Context_Name") if c in frame.columns],
        ].copy()
    result = result.rename(columns={id_column: "pr_id", value_column: "value"})
    context_column = "Context_Name" if "Context_Name" in result.columns else "context"
    if context_column not in result.columns:
        raise RuntimeError(f"P5--P6 source for {metric} has no context label")
    result = result.rename(columns={context_column: "context"})
    # The corrected full-study files also contain Llama. This frozen diagnostic
    # has regenerated P1--P4 data only for the GPT and Qwen manifest models, so
    # unrelated model rows are excluded before mapping and coverage validation.
    result = result[result["model"].isin(model_map)].copy()
    result["model_label"] = result["model"].map(model_map)
    if result["model_label"].isna().any():
        unknown = sorted(result.loc[result["model_label"].isna(), "model"].unique())
        raise RuntimeError(f"P5--P6 source for {metric} has unknown models: {unknown}")
    result["metric"] = metric
    result["aggregation"] = aggregation
    result["source_segment"] = "fixed corrected P5-P6"
    base_columns = [
        "model_label",
        "model",
        "pr_id",
        "variation",
        "context",
        "metric",
        "aggregation",
        "value",
        "source_segment",
    ]
    result = result[base_columns]
    return pd.concat(
        [result.assign(bridge_version=version) for version in VERSIONS],
        ignore_index=True,
    )[
        [
            "model_label",
            "model",
            "pr_id",
            "variation",
            "context",
            "bridge_version",
            "metric",
            "aggregation",
            "value",
            "source_segment",
        ]
    ]


def render_report(
    rankings: pd.DataFrame,
    headline: pd.DataFrame,
    coverage: pd.DataFrame,
    decision: str,
) -> str:
    lines = [
        "# Six-context generation-session bridge report",
        "",
        "This no-API diagnostic replaces only P1--P4 with the newer session and holds corrected P5--P6 fixed.",
        "It is a robustness bridge, not a unified-session replication of the full 308-PR study.",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        "The predeclared trigger is whether P6 loses top aggregate rank in any current model-by-metric cell.",
        "",
        "## Complete-case coverage",
        "",
        "| Model | PRs | Rows per bridge version and metric |",
        "|---|---:|---:|",
    ]
    for row in coverage.sort_values("model_label").itertuples(index=False):
        lines.append(f"| {row.model_label} | {row.complete_prs} | {row.rows_per_version_metric} |")

    lines.extend(["", "## Headline stability", ""])
    for row in headline.sort_values(["model_label", "metric"]).itertuples(index=False):
        historical_top = bool(row.historical_p6_is_top)
        current_top = bool(row.current_p6_is_top)
        if historical_top and current_top:
            status = "retained top rank"
        elif not historical_top and current_top:
            status = "became top-ranked"
        elif historical_top and not current_top:
            status = "lost top rank"
        else:
            status = "was not top-ranked in either bridge"
        lines.extend(
            [
                f"### {row.model_label} — {row.metric}",
                "",
                f"P6 top status: **{status}** (historical bridge: {historical_top}; current bridge: {current_top}).",
                f"Current top context(s): {row.current_top_contexts}. P6 gap from best non-P6 context: {row.current_p6_gap_to_best_other:.8f}.",
                f"Within-PR P6 tied-top share: {row.current_p6_tied_top_share:.3f}; strict-top share: {row.current_p6_strict_top_share:.3f} (n={int(row.n)}).",
                "",
            ]
        )

    lines.extend(["## Current bridge rankings", ""])
    current = rankings[rankings["bridge_version"].eq("current_bridge")]
    for (model_label, metric), group in current.groupby(["model_label", "metric"], sort=True):
        lines.extend(
            [
                f"### {model_label} — {metric}",
                "",
                "| Rank | Context | Code | Aggregate |",
                "|---:|---|---|---:|",
            ]
        )
        for row in group.sort_values(["rank_descending", "variation"]).itertuples(index=False):
            lines.append(
                f"| {row.rank_descending:g} | {row.context} | {row.variation} | {row.aggregate_value:.8f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "P5 and P6 are identical in both bridge versions by design. The diagnostic therefore answers whether the newer P1--P4 results displace P6 in the frozen sample; it does not estimate all six contexts from one newly generated session and does not establish statistical equivalence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    require_nokogiri()
    root = project_root(Path(__file__))
    sensitivity = root / "phase3/analysis/outputs/tier_b/generation-session-outcome-sensitivity-v1"
    corrected = root / "phase3/analysis/outputs/corrected_rq23_v1/notebooks/outputs"
    out = root / "phase3/analysis/outputs/tier_b/generation-session-six-context-bridge-v1"
    out.mkdir(parents=True, exist_ok=True)

    sources = {
        "pair_manifest": sensitivity / "pair_manifest.csv",
        "main_metrics": sensitivity / "main_metrics.csv",
        "session_bertscore": sensitivity / "bertscore_metrics.csv",
        "corrected_rouge": corrected / "row_level/rouge_metrics.csv",
        "corrected_bertscore": corrected / "row_level/bertscore_metrics.csv",
        "corrected_bm25": corrected / "bm25_retrieval/bm25_retrieval_scores.csv",
    }
    missing_sources = [str(path) for path in sources.values() if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Missing authoritative sources: {missing_sources}")

    manifest = pd.read_csv(sources["pair_manifest"], dtype={"pr_id": str})
    ids_by_model = complete_ids(manifest)
    model_rows = manifest[["model_label", "model"]].drop_duplicates()
    exact_one(model_rows, ["model_label"], "model mapping")
    model_map = dict(zip(model_rows["model"], model_rows["model_label"]))

    main_metrics = pd.read_csv(sources["main_metrics"], dtype={"pr_id": str})
    session_bert = pd.read_csv(sources["session_bertscore"], dtype={"pr_id": str})
    corrected_rouge = pd.read_csv(sources["corrected_rouge"], dtype={"normalized_pr_id": str})
    corrected_bert = pd.read_csv(sources["corrected_bertscore"], dtype={"normalized_pr_id": str})
    corrected_bm25 = pd.read_csv(sources["corrected_bm25"], dtype={"normalized_pr_id": str})

    metric_specs: list[tuple[str, str, str, pd.DataFrame, str, pd.DataFrame]] = [
        ("rougeL_fmeasure", "median", "rougeL_fmeasure", main_metrics, "rougeL_fmeasure", corrected_rouge),
        ("bertscore_f1", "median", "bertscore_f1", session_bert, "bertscore_f1", corrected_bert),
        ("bm25_reciprocal_rank", "mean", "bm25_reciprocal_rank", main_metrics, "reciprocal_rank", corrected_bm25),
    ]

    bridge_parts: list[pd.DataFrame] = []
    for metric, aggregation, p14_column, p14_source, p56_column, p56_source in metric_specs:
        p14 = prepare_p14(p14_source, p14_column, metric, aggregation)
        p56 = prepare_p56(p56_source, p56_column, metric, aggregation, model_map)
        for model_label, expected_ids in ids_by_model.items():
            p14_model = p14[p14["pr_id"].isin(expected_ids)]
            p56_model = p56[p56["pr_id"].isin(expected_ids)]
            for version in VERSIONS:
                validate_matrix(
                    p14_model[p14_model["bridge_version"].eq(version)],
                    model_label,
                    expected_ids,
                    P14,
                    f"{metric} {model_label} {version} P1-P4",
                )
                validate_matrix(
                    p56_model[p56_model["bridge_version"].eq(version)],
                    model_label,
                    expected_ids,
                    P56,
                    f"{metric} {model_label} {version} P5-P6",
                )
        bridge_parts.extend(
            [
                p14[p14.apply(lambda row: row["pr_id"] in ids_by_model[row["model_label"]], axis=1)],
                p56[p56.apply(lambda row: row["pr_id"] in ids_by_model[row["model_label"]], axis=1)],
            ]
        )

    bridge = pd.concat(bridge_parts, ignore_index=True)
    exact_one(
        bridge,
        ["model_label", "pr_id", "variation", "bridge_version", "metric"],
        "bridge rows",
    )
    if bridge["value"].isna().any():
        raise RuntimeError("Bridge contains missing metric values")

    coverage_rows = []
    for model_label, expected_ids in ids_by_model.items():
        coverage_rows.append(
            {
                "model_label": model_label,
                "complete_prs": len(expected_ids),
                "contexts": len(ALL_CONTEXTS),
                "bridge_versions": len(VERSIONS),
                "metrics": len(metric_specs),
                "rows_per_version_metric": len(expected_ids) * len(ALL_CONTEXTS),
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    ranking_rows = []
    headline_rows = []
    for (model_label, model, metric, aggregation, version), group in bridge.groupby(
        ["model_label", "model", "metric", "aggregation", "bridge_version"], sort=True
    ):
        pivot = group.pivot(index="pr_id", columns="variation", values="value")
        if list(sorted(pivot.columns)) != list(ALL_CONTEXTS):
            raise RuntimeError(f"{model_label} {metric} {version}: context columns differ")
        if pivot.isna().any().any():
            raise RuntimeError(f"{model_label} {metric} {version}: incomplete matrix")
        context_labels = group[["variation", "context"]].drop_duplicates()
        exact_one(context_labels, ["variation"], f"{model_label} {metric} context labels")
        label_map = dict(zip(context_labels["variation"], context_labels["context"]))
        aggregates = {
            variation: aggregate_value(pivot[variation], aggregation)
            for variation in ALL_CONTEXTS
        }
        aggregate_series = pd.Series(aggregates, dtype=float)
        ranks = aggregate_series.rank(method="average", ascending=False)
        maximum = float(aggregate_series.max())
        top_variations = [
            variation for variation in ALL_CONTEXTS if np.isclose(aggregates[variation], maximum, rtol=0, atol=1e-15)
        ]
        other_max = max(aggregates[variation] for variation in ALL_CONTEXTS if variation != "P-6")
        row_max = pivot.max(axis=1)
        p6_tied = np.isclose(pivot["P-6"].to_numpy(float), row_max.to_numpy(float), rtol=0, atol=1e-15)
        other_row_max = pivot[[variation for variation in ALL_CONTEXTS if variation != "P-6"]].max(axis=1)
        p6_strict = pivot["P-6"].to_numpy(float) > other_row_max.to_numpy(float)
        for variation in ALL_CONTEXTS:
            ranking_rows.append(
                {
                    "model_label": model_label,
                    "model": model,
                    "metric": metric,
                    "aggregation": aggregation,
                    "bridge_version": version,
                    "variation": variation,
                    "context": label_map[variation],
                    "n": len(pivot),
                    "aggregate_value": aggregates[variation],
                    "rank_descending": float(ranks[variation]),
                    "is_top": variation in top_variations,
                }
            )
        headline_rows.append(
            {
                "model_label": model_label,
                "model": model,
                "metric": metric,
                "aggregation": aggregation,
                "bridge_version": version,
                "n": len(pivot),
                "top_variations": ";".join(top_variations),
                "top_contexts": ";".join(label_map[variation] for variation in top_variations),
                "p6_context": label_map["P-6"],
                "p6_aggregate": aggregates["P-6"],
                "p6_rank": float(ranks["P-6"]),
                "p6_is_top": "P-6" in top_variations,
                "p6_gap_to_best_other": aggregates["P-6"] - other_max,
                "p6_tied_top_share": float(np.mean(p6_tied)),
                "p6_strict_top_share": float(np.mean(p6_strict)),
            }
        )

    rankings = pd.DataFrame(ranking_rows)
    headline_by_version = pd.DataFrame(headline_rows)
    historical = headline_by_version[headline_by_version["bridge_version"].eq("historical_bridge")].drop(columns="bridge_version")
    current = headline_by_version[headline_by_version["bridge_version"].eq("current_bridge")].drop(columns="bridge_version")
    headline = historical.merge(
        current,
        on=["model_label", "model", "metric", "aggregation", "n", "p6_context"],
        suffixes=("_historical", "_current"),
        validate="one_to_one",
    )
    headline = headline.rename(
        columns={
            "top_variations_historical": "historical_top_variations",
            "top_contexts_historical": "historical_top_contexts",
            "p6_aggregate_historical": "historical_p6_aggregate",
            "p6_rank_historical": "historical_p6_rank",
            "p6_is_top_historical": "historical_p6_is_top",
            "p6_gap_to_best_other_historical": "historical_p6_gap_to_best_other",
            "p6_tied_top_share_historical": "historical_p6_tied_top_share",
            "p6_strict_top_share_historical": "historical_p6_strict_top_share",
            "top_variations_current": "current_top_variations",
            "top_contexts_current": "current_top_contexts",
            "p6_aggregate_current": "current_p6_aggregate",
            "p6_rank_current": "current_p6_rank",
            "p6_is_top_current": "current_p6_is_top",
            "p6_gap_to_best_other_current": "current_p6_gap_to_best_other",
            "p6_tied_top_share_current": "current_p6_tied_top_share",
            "p6_strict_top_share_current": "current_p6_strict_top_share",
        }
    )
    headline["p6_top_status_stable"] = headline["historical_p6_is_top"].eq(headline["current_p6_is_top"])
    headline["p6_aggregate_change"] = headline["current_p6_aggregate"] - headline["historical_p6_aggregate"]

    rank_historical = rankings[rankings["bridge_version"].eq("historical_bridge")].drop(columns="bridge_version")
    rank_current = rankings[rankings["bridge_version"].eq("current_bridge")].drop(columns="bridge_version")
    comparison = rank_historical.merge(
        rank_current,
        on=["model_label", "model", "metric", "aggregation", "variation", "context", "n"],
        suffixes=("_historical", "_current"),
        validate="one_to_one",
    )
    comparison["aggregate_change"] = comparison["aggregate_value_current"] - comparison["aggregate_value_historical"]
    comparison["rank_change"] = comparison["rank_descending_current"] - comparison["rank_descending_historical"]

    all_current_top = bool(headline["current_p6_is_top"].all())
    decision_code = "does_not_trigger_complete_rerun" if all_current_top else "review_required_complete_rerun_triggered"
    decision_text = (
        "P6 remains top-ranked in every current bridge cell; this diagnostic does not trigger a complete rerun."
        if all_current_top
        else "P6 loses top rank in at least one current bridge cell; review the flagged cell(s) before deciding on a complete rerun."
    )

    output_frames = {
        "bridge_metric_rows.csv": bridge.sort_values(["metric", "model_label", "bridge_version", "pr_id", "variation"]),
        "coverage.csv": coverage.sort_values("model_label"),
        "context_rankings.csv": rankings.sort_values(["metric", "model_label", "bridge_version", "rank_descending", "variation"]),
        "headline_stability.csv": headline.sort_values(["metric", "model_label"]),
        "context_version_comparison.csv": comparison.sort_values(["metric", "model_label", "variation"]),
    }
    for name, frame in output_frames.items():
        leaked = FORBIDDEN_COLUMNS & set(frame.columns)
        if leaked:
            raise RuntimeError(f"{name} exposes forbidden raw columns: {sorted(leaked)}")
        frame.to_csv(out / name, index=False)

    report = render_report(rankings, headline, coverage, decision_text)
    (out / "BRIDGE_REPORT.md").write_text(report, encoding="utf-8")
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    metadata = {
        "protocol": PROTOCOL,
        "created_on_hostname": platform.node(),
        "git_commit": git_commit,
        "decision_code": decision_code,
        "decision_text": decision_text,
        "complete_prs": {row["model_label"]: int(row["complete_prs"]) for row in coverage_rows},
        "bridge_definition": {
            "historical_bridge": "historical P1-P4 plus fixed corrected P5-P6",
            "current_bridge": "current P1-P4 plus the same fixed corrected P5-P6",
        },
        "aggregations": {metric: aggregation for metric, aggregation, *_ in metric_specs},
        "p6_current_top_in_all_cells": all_current_top,
        "external_api_calls": False,
        "new_generation": False,
        "new_metric_scoring": False,
        "raw_text_in_outputs": False,
        "manuscript_modified": False,
        "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
    }
    write_json(out / "analysis_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
