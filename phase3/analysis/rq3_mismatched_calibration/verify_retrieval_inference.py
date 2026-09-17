#!/usr/bin/env python3
"""Independently reconstruct and verify paired retrieval-inference outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(left: float, right: float, label: str, tolerance: float = 1e-12) -> None:
    if not np.isclose(left, right, rtol=0.0, atol=tolerance, equal_nan=True):
        raise RuntimeError(f"{label}: observed {left}, expected {right}")


def rank_biserial(human: np.ndarray, ai: np.ndarray) -> float:
    differences = np.asarray(human, dtype=float) - np.asarray(ai, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if differences.size == 0:
        return float("nan")
    ranks = rankdata(np.abs(differences), method="average")
    return float(
        (ranks[differences > 0].sum() - ranks[differences < 0].sum()) / ranks.sum()
    )


def align(human: pd.DataFrame, ai: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    h = human.groupby("normalized_pr_id")["reciprocal_rank"].mean()
    a = ai.groupby("normalized_pr_id")["reciprocal_rank"].mean()
    pair = pd.concat([h.rename("human"), a.rename("ai")], axis=1, join="inner").dropna()
    return pair["human"].to_numpy(dtype=float), pair["ai"].to_numpy(dtype=float)


def bootstrap_seed(encoder: str, model: str) -> int:
    payload = f"{BOOTSTRAP_SEED}|{encoder}|{model}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def bootstrap_interval(human: np.ndarray, ai: np.ndarray, seed: int) -> tuple[float, float]:
    differences = human - ai
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for start in range(0, BOOTSTRAP_RESAMPLES, 1_000):
        stop = min(start + 1_000, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        estimates[start:stop] = differences[indices].mean(axis=1)
    values = np.quantile(estimates, [0.025, 0.975], method="linear")
    return float(values[0]), float(values[1])


def paired_values(human: np.ndarray, ai: np.ndarray) -> dict[str, float | int]:
    if len(human) == 0 or len(human) != len(ai):
        raise RuntimeError("Invalid paired vectors")
    if np.all(human - ai == 0):
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(human, ai, alternative="two-sided", zero_method="wilcox", method="auto")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    return {
        "n": len(human),
        "human_mrr": float(human.mean()),
        "ai_mrr": float(ai.mean()),
        "delta_mrr_human_minus_ai": float((human - ai).mean()),
        "human_median_reciprocal_rank": float(np.median(human)),
        "ai_median_reciprocal_rank": float(np.median(ai)),
        "wilcoxon_statistic": statistic,
        "p": p_value,
        "rank_biserial_human_gt_ai": rank_biserial(human, ai),
    }


def load_rows(calibration_root: Path, encoder: str) -> pd.DataFrame:
    frame = pd.read_csv(
        calibration_root / "row_level" / f"{encoder}_calibration.csv",
        dtype={"normalized_pr_id": "string"},
    )
    if len(frame) != 5630:
        raise RuntimeError(f"{encoder}: unexpected row count")
    return frame


def verify_pooled_summary(observed: pd.DataFrame, rows: dict[str, pd.DataFrame]) -> None:
    if len(observed) != 8:
        raise RuntimeError("Expected eight pooled retrieval-summary rows")
    for record in observed.itertuples(index=False):
        frame = rows[record.encoder]
        if record.source == "human":
            values = frame[frame["entry_type"] == "human"].copy()
            description_rows, context_min, context_max = 308, 1, 1
        else:
            source = frame[(frame["entry_type"] == "ai") & (frame["model"] == record.model)]
            grouped = source.groupby("normalized_pr_id", sort=True)
            values = grouped[["correct_rank", "reciprocal_rank", "top_1", "top_5", "top_10"]].mean()
            counts = grouped.size()
            description_rows = len(source)
            context_min, context_max = int(counts.min()), int(counts.max())
        if record.n_pr != len(values) or record.n_description_rows != description_rows:
            raise RuntimeError("Pooled retrieval population mismatch")
        if record.contexts_per_pr_min != context_min or record.contexts_per_pr_max != context_max:
            raise RuntimeError("Pooled retrieval context coverage mismatch")
        expected = {
            "mean_reciprocal_rank": values["reciprocal_rank"].mean(),
            "median_reciprocal_rank": values["reciprocal_rank"].median(),
            "median_pr_average_correct_rank": values["correct_rank"].median(),
            "recall_at_1": values["top_1"].mean(),
            "recall_at_5": values["top_5"].mean(),
            "recall_at_10": values["top_10"].mean(),
        }
        for column, value in expected.items():
            assert_close(float(getattr(record, column)), float(value), f"summary {column}")


def verify_pooled_comparisons(observed: pd.DataFrame, rows: dict[str, pd.DataFrame]) -> None:
    if len(observed) != 6:
        raise RuntimeError("Expected six pooled reciprocal-rank comparisons")
    for record in observed.itertuples(index=False):
        frame = rows[record.encoder]
        human = frame[frame["entry_type"] == "human"]
        ai_source = frame[(frame["entry_type"] == "ai") & (frame["model"] == record.model)]
        ai = ai_source.groupby("normalized_pr_id", as_index=False)["reciprocal_rank"].mean()
        human_rr, ai_rr = align(human, ai)
        expected = paired_values(human_rr, ai_rr)
        for column in [
            "n", "human_mrr", "ai_mrr", "delta_mrr_human_minus_ai",
            "human_median_reciprocal_rank", "wilcoxon_statistic", "p",
            "rank_biserial_human_gt_ai",
        ]:
            assert_close(float(getattr(record, column)), float(expected[column]), f"pooled {column}")
        assert_close(
            float(record.ai_median_pr_average_reciprocal_rank),
            float(expected["ai_median_reciprocal_rank"]),
            "pooled ai median reciprocal rank",
        )
        low, high = bootstrap_interval(human_rr, ai_rr, bootstrap_seed(record.encoder, record.model))
        assert_close(record.delta_mrr_ci95_low, low, "bootstrap low")
        assert_close(record.delta_mrr_ci95_high, high, "bootstrap high")
    for encoder, group in observed.groupby("encoder", sort=False):
        rejected, adjusted, _, _ = multipletests(group["p"].to_numpy(), method="fdr_bh")
        if not np.allclose(group["p_adj_bh"], adjusted, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{encoder}: pooled BH mismatch")
        if not np.array_equal(group["significant_bh_0_05"].astype(bool), rejected):
            raise RuntimeError(f"{encoder}: pooled BH decision mismatch")


def verify_context_comparisons(observed: pd.DataFrame, rows: dict[str, pd.DataFrame]) -> None:
    if len(observed) != 36:
        raise RuntimeError("Expected 36 context reciprocal-rank comparisons")
    for record in observed.itertuples(index=False):
        frame = rows[record.encoder]
        human = frame[frame["entry_type"] == "human"]
        ai = frame[
            (frame["entry_type"] == "ai")
            & (frame["model"] == record.model)
            & (frame["variation"] == record.variation)
        ]
        human_rr, ai_rr = align(human, ai)
        expected = paired_values(human_rr, ai_rr)
        for column in [
            "n", "human_mrr", "ai_mrr", "delta_mrr_human_minus_ai",
            "human_median_reciprocal_rank", "ai_median_reciprocal_rank",
            "wilcoxon_statistic", "p", "rank_biserial_human_gt_ai",
        ]:
            assert_close(float(getattr(record, column)), float(expected[column]), f"context {column}")
        if record.context != CONTEXT_MAPPING[record.variation]:
            raise RuntimeError("Context label mismatch")
    for encoder, group in observed.groupby("encoder", sort=False):
        rejected, adjusted, _, _ = multipletests(group["p"].to_numpy(), method="fdr_bh")
        if not np.allclose(group["p_adj_bh"], adjusted, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{encoder}: context BH mismatch")
        if not np.array_equal(group["significant_bh_0_05"].astype(bool), rejected):
            raise RuntimeError(f"{encoder}: context BH decision mismatch")


def output_hashes(output_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"completion.json", "pipeline.log"}:
            hashes[str(path.relative_to(output_root))] = sha256_file(path)
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_root = args.calibration_root.resolve()
    output_root = args.output_root.resolve()
    metadata = json.loads((output_root / "analysis_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("protocol") != PROTOCOL or metadata.get("analysis", {}).get("primary_outcome") != "per-description reciprocal rank of the correct PR diff":
        raise RuntimeError("Unexpected retrieval analysis protocol")
    source_completion_path = calibration_root / "completion.json"
    if sha256_file(source_completion_path) != COMPLETION_SHA256:
        raise RuntimeError("Source completion-manifest hash mismatch")
    source_completion = json.loads(source_completion_path.read_text(encoding="utf-8"))
    if source_completion.get("protocol") != SOURCE_PROTOCOL or source_completion.get("status") != "complete_and_verified":
        raise RuntimeError("Unexpected or incomplete source calibration")
    for encoder, expected_hash in ROW_HASHES.items():
        relative = f"row_level/{encoder}_calibration.csv"
        if source_completion.get("output_hashes", {}).get(relative) != expected_hash:
            raise RuntimeError(f"Source manifest does not freeze {relative}")
        if sha256_file(calibration_root / relative) != expected_hash:
            raise RuntimeError(f"Source row hash mismatch for {relative}")

    rows = {encoder: load_rows(calibration_root, encoder) for encoder in ["codebert", "graphcodebert"]}
    summary_dir = output_root / "summaries"
    verify_pooled_summary(pd.read_csv(summary_dir / "pooled_retrieval_summary.csv"), rows)
    verify_pooled_comparisons(
        pd.read_csv(summary_dir / "pooled_human_ai_reciprocal_rank_comparisons.csv"), rows
    )
    verify_context_comparisons(
        pd.read_csv(summary_dir / "context_human_ai_reciprocal_rank_comparisons.csv"), rows
    )

    verification_dir = output_root / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": PROTOCOL,
        "status": "pass",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "source_calibration_complete_and_verified": "pass",
            "source_hashes": "pass",
            "row_level_rank_identities": "pass",
            "pooled_descriptive_summaries": "pass",
            "pooled_paired_wilcoxon": "pass",
            "pooled_matched_pairs_rank_biserial": "pass",
            "pooled_paired_bootstrap_intervals": "pass",
            "context_paired_comparisons": "pass",
            "bh_corrections": "pass",
            "ordinary_cliffs_delta_reported": False,
        },
    }
    report_path = verification_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completion = {
        "protocol": PROTOCOL,
        "status": "complete_and_verified",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_report": str(report_path.relative_to(output_root)),
        "output_hashes": output_hashes(output_root),
    }
    (output_root / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
