#!/usr/bin/env python3
"""Paired inference for the established correct-diff retrieval outcomes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from paired_stats import align_pairs, apply_fdr_by, paired_test  # noqa: E402


PROTOCOL = "rq3-retrieval-inference-v1"
SOURCE_PROTOCOL = "rq3-mismatched-calibration-v1"
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260914
COMPLETION_SHA256 = "032e69cac929ee2745aaed5b3d78b04a096cb7f83a138b95cdd6dc76eff5926e"
ROW_HASHES = {
    "codebert": "90038f1deec7449427c5ac012a0320ad9b16e1f5c0d66d6c080aab80a1add17e",
    "graphcodebert": "571760f6e4c05c077372b5c390bc9fdf950c41a5a47b58a56d1d3daee1ca7f79",
}
TARGET_MODELS = [
    "meta-llama/llama-3.1-70b-instruct",
    "gpt-5-mini-2025-08-07",
    "Qwen/Qwen2.5-72B-Instruct",
]
CONTEXT_MAPPING = {
    "P-1": "Repo Path",
    "P-2": "PR Title",
    "P-3": "File Diff",
    "P-4": "File Contents",
    "P-5": "Linked Issue",
    "P-6": "Without File Contents",
}
KEY_COLUMNS = ["normalized_pr_id", "entry_type", "model", "variation"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def verify_source(calibration_root: Path) -> dict[str, str]:
    completion_path = calibration_root / "completion.json"
    if sha256_file(completion_path) != COMPLETION_SHA256:
        raise RuntimeError("Frozen calibration completion-manifest hash mismatch")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("protocol") != SOURCE_PROTOCOL:
        raise RuntimeError("Unexpected source protocol")
    if completion.get("status") != "complete_and_verified":
        raise RuntimeError("Source calibration is not complete and verified")
    observed: dict[str, str] = {"completion.json": sha256_file(completion_path)}
    for encoder, expected in ROW_HASHES.items():
        relative = f"row_level/{encoder}_calibration.csv"
        path = calibration_root / relative
        actual = sha256_file(path)
        if actual != expected or completion["output_hashes"].get(relative) != expected:
            raise RuntimeError(f"Frozen source hash mismatch for {relative}")
        observed[relative] = actual
    return observed


def load_rows(calibration_root: Path, encoder: str) -> pd.DataFrame:
    frame = pd.read_csv(
        calibration_root / "row_level" / f"{encoder}_calibration.csv",
        dtype={"normalized_pr_id": "string"},
    )
    required = {
        *KEY_COLUMNS,
        "Context_Name",
        "correct_rank",
        "reciprocal_rank",
        "top_1",
        "top_5",
        "top_10",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{encoder}: missing source columns: {sorted(missing)}")
    if len(frame) != 5630 or frame[KEY_COLUMNS].duplicated().any():
        raise RuntimeError(f"{encoder}: unexpected source population or duplicate keys")
    if not np.allclose(
        frame["reciprocal_rank"].to_numpy(dtype=float),
        1.0 / frame["correct_rank"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(f"{encoder}: reciprocal ranks do not match correct ranks")
    for k in (1, 5, 10):
        expected = (frame["correct_rank"] <= k).astype(int)
        if not expected.equals(frame[f"top_{k}"].astype(int)):
            raise RuntimeError(f"{encoder}: top-{k} flags do not match correct ranks")
    if len(frame[frame["entry_type"] == "human"]) != 308:
        raise RuntimeError(f"{encoder}: expected 308 human rows")
    if len(frame[frame["entry_type"] == "ai"]) != 5322:
        raise RuntimeError(f"{encoder}: expected 5,322 AI rows")
    return frame


def bootstrap_seed(encoder: str, model: str) -> int:
    payload = f"{BOOTSTRAP_SEED}|{encoder}|{model}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def paired_bootstrap_mean_difference(
    human: np.ndarray, ai: np.ndarray, seed: int
) -> tuple[float, float]:
    differences = np.asarray(human, dtype=float) - np.asarray(ai, dtype=float)
    if differences.ndim != 1 or differences.size == 0 or not np.isfinite(differences).all():
        raise RuntimeError("Bootstrap requires a finite one-dimensional paired difference")
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    batch_size = 1_000
    for start in range(0, BOOTSTRAP_RESAMPLES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, differences.size, size=(stop - start, differences.size))
        estimates[start:stop] = differences[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975], method="linear")
    return float(lower), float(upper)


def per_pr_ai(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    subset = frame[(frame["entry_type"] == "ai") & (frame["model"] == model)]
    grouped = subset.groupby("normalized_pr_id", sort=True)
    result = grouped[
        ["correct_rank", "reciprocal_rank", "top_1", "top_5", "top_10"]
    ].mean()
    result["context_count"] = grouped.size()
    if len(result) != 308 or result["context_count"].min() < 4 or result["context_count"].max() > 6:
        raise RuntimeError(f"{model}: unexpected per-PR context coverage")
    return result.reset_index()


def descriptive_row(
    encoder: str,
    source: str,
    model: str,
    values: pd.DataFrame,
    description_rows: int,
    context_min: int,
    context_max: int,
) -> dict[str, object]:
    return {
        "encoder": encoder,
        "source": source,
        "model": model,
        "n_pr": int(len(values)),
        "n_description_rows": int(description_rows),
        "contexts_per_pr_min": int(context_min),
        "contexts_per_pr_max": int(context_max),
        "mean_reciprocal_rank": float(values["reciprocal_rank"].mean()),
        "median_reciprocal_rank": float(values["reciprocal_rank"].median()),
        "median_pr_average_correct_rank": float(values["correct_rank"].median()),
        "recall_at_1": float(values["top_1"].mean()),
        "recall_at_5": float(values["top_5"].mean()),
        "recall_at_10": float(values["top_10"].mean()),
    }


def build_pooled_outputs(frame: pd.DataFrame, encoder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    human = frame[frame["entry_type"] == "human"].sort_values("normalized_pr_id")
    summaries = [
        descriptive_row(encoder, "human", "Human", human, 308, 1, 1)
    ]
    comparisons: list[dict[str, object]] = []
    for model in TARGET_MODELS:
        ai_rows = frame[(frame["entry_type"] == "ai") & (frame["model"] == model)]
        ai = per_pr_ai(frame, model)
        summaries.append(
            descriptive_row(
                encoder,
                "ai_context_average",
                model,
                ai,
                len(ai_rows),
                int(ai["context_count"].min()),
                int(ai["context_count"].max()),
            )
        )
        human_rr, ai_rr = align_pairs(human, ai, "reciprocal_rank")
        stats = paired_test(human_rr, ai_rr)
        ci_low, ci_high = paired_bootstrap_mean_difference(
            human_rr, ai_rr, bootstrap_seed(encoder, model)
        )
        comparisons.append({
            "encoder": encoder,
            "metric": "reciprocal_rank",
            "model": model,
            "n": int(stats["n"]),
            "human_mrr": float(np.mean(human_rr)),
            "ai_mrr": float(np.mean(ai_rr)),
            "delta_mrr_human_minus_ai": float(np.mean(human_rr - ai_rr)),
            "delta_mrr_ci95_low": ci_low,
            "delta_mrr_ci95_high": ci_high,
            "human_median_reciprocal_rank": float(stats["median_human"]),
            "ai_median_pr_average_reciprocal_rank": float(stats["median_ai"]),
            "wilcoxon_statistic": float(stats["wilcoxon_stat"]),
            "p": float(stats["p"]),
            "rank_biserial_human_gt_ai": float(stats["rank_biserial"]),
            "ai_contexts_per_pr_min": int(ai["context_count"].min()),
            "ai_contexts_per_pr_max": int(ai["context_count"].max()),
        })
    comparison_frame = apply_fdr_by(pd.DataFrame(comparisons), by="encoder")
    return pd.DataFrame(summaries), comparison_frame.rename(
        columns={"adj_p": "p_adj_bh", "sig": "significant_bh_0_05"}
    )


def build_context_comparisons(frame: pd.DataFrame, encoder: str) -> pd.DataFrame:
    human = frame[frame["entry_type"] == "human"]
    rows: list[dict[str, object]] = []
    for model in TARGET_MODELS:
        for variation, context in CONTEXT_MAPPING.items():
            ai = frame[
                (frame["entry_type"] == "ai")
                & (frame["model"] == model)
                & (frame["variation"] == variation)
            ]
            human_rr, ai_rr = align_pairs(human, ai, "reciprocal_rank")
            stats = paired_test(human_rr, ai_rr)
            rows.append({
                "encoder": encoder,
                "metric": "reciprocal_rank",
                "model": model,
                "variation": variation,
                "context": context,
                "n": int(stats["n"]),
                "human_mrr": float(np.mean(human_rr)),
                "ai_mrr": float(np.mean(ai_rr)),
                "delta_mrr_human_minus_ai": float(np.mean(human_rr - ai_rr)),
                "human_median_reciprocal_rank": float(stats["median_human"]),
                "ai_median_reciprocal_rank": float(stats["median_ai"]),
                "wilcoxon_statistic": float(stats["wilcoxon_stat"]),
                "p": float(stats["p"]),
                "rank_biserial_human_gt_ai": float(stats["rank_biserial"]),
            })
    result = apply_fdr_by(pd.DataFrame(rows), by="encoder")
    return result.rename(columns={"adj_p": "p_adj_bh", "sig": "significant_bh_0_05"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_root = args.calibration_root.resolve()
    observed_hashes = verify_source(calibration_root)
    rows_by_encoder = {
        encoder: load_rows(calibration_root, encoder) for encoder in ROW_HASHES
    }
    if args.preflight:
        print(json.dumps({
            "status": "preflight_pass",
            "protocol": PROTOCOL,
            "source_protocol": SOURCE_PROTOCOL,
            "model_inference": False,
            "encoders": list(rows_by_encoder),
            "rows_per_encoder": 5630,
            "candidate_diffs": 308,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        }, indent=2, sort_keys=True))
        return
    if args.output_root is None:
        raise RuntimeError("--output-root is required outside preflight")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pooled_summaries: list[pd.DataFrame] = []
    pooled_comparisons: list[pd.DataFrame] = []
    context_comparisons: list[pd.DataFrame] = []
    for encoder, frame in rows_by_encoder.items():
        summary, comparison = build_pooled_outputs(frame, encoder)
        pooled_summaries.append(summary)
        pooled_comparisons.append(comparison)
        context_comparisons.append(build_context_comparisons(frame, encoder))

    summary_dir = output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(pooled_summaries, ignore_index=True).to_csv(
        summary_dir / "pooled_retrieval_summary.csv", index=False
    )
    pd.concat(pooled_comparisons, ignore_index=True).to_csv(
        summary_dir / "pooled_human_ai_reciprocal_rank_comparisons.csv", index=False
    )
    pd.concat(context_comparisons, ignore_index=True).to_csv(
        summary_dir / "context_human_ai_reciprocal_rank_comparisons.csv", index=False
    )

    candidate_count = 308
    metadata = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "paired_stats": str((ANALYSIS_DIR / "paired_stats.py").resolve()),
            "paired_stats_sha256": sha256_file(ANALYSIS_DIR / "paired_stats.py"),
            "pip": package_version("pip"),
        },
        "source_root": str(calibration_root),
        "source_hashes": observed_hashes,
        "population": {
            "human_descriptions": 308,
            "ai_descriptions": 5322,
            "candidate_diffs": candidate_count,
        },
        "random_retrieval_reference": {
            "expected_mrr": float(sum(1.0 / rank for rank in range(1, candidate_count + 1)) / candidate_count),
            "expected_mean_rank": (candidate_count + 1) / 2,
            "recall_at_1": 1.0 / candidate_count,
            "recall_at_5": 5.0 / candidate_count,
            "recall_at_10": 10.0 / candidate_count,
        },
        "analysis": {
            "primary_outcome": "per-description reciprocal rank of the correct PR diff",
            "ai_pooling": "mean across available contexts within model and PR",
            "paired_test": "two-sided Wilcoxon signed-rank by PR",
            "effect_size": "matched-pairs rank-biserial; positive means human > AI",
            "multiple_comparison_correction": {
                "pooled": "BH across three generators within encoder",
                "context": "BH across 18 model-context comparisons within encoder",
            },
            "bootstrap": {
                "estimand": "human MRR minus AI MRR",
                "method": "paired percentile bootstrap",
                "resamples": BOOTSTRAP_RESAMPLES,
                "base_seed": BOOTSTRAP_SEED,
            },
            "recall_inference": "descriptive only",
        },
    }
    (output_root / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Retrieval inference complete; independent verification is required.")


if __name__ == "__main__":
    main()
