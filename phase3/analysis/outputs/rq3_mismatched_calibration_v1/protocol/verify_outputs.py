#!/usr/bin/env python3
"""Independently reconstruct and verify the RQ3 calibration outputs."""

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


PROTOCOL = "rq3-mismatched-calibration-v1"
EXPECTED_ROW_HASHES = {
    "codebert": "932dbb166dfcc221e48a7696161ed541e24057e059bc79e057bd1e331e5c1550",
    "graphcodebert": "2c96eb065c971f07fc21ffd145f9cbab097695b532454303504b8c1875b2820b",
}
EXPECTED_SCORE_COLUMNS = {
    "codebert": "codebert_similarity",
    "graphcodebert": "graphcodebert_similarity",
}
TARGET_MODELS = [
    "meta-llama/llama-3.1-70b-instruct",
    "gpt-5-mini-2025-08-07",
    "Qwen/Qwen2.5-72B-Instruct",
]
ROW_KEYS = ["normalized_pr_id", "entry_type", "model", "variation"]
NUMERIC_ROW_COLUMNS = [
    "matched_similarity", "mismatch_mean", "mismatch_median", "mismatch_std",
    "mismatch_q95", "mismatch_max", "margin_mean", "margin_median",
    "correct_rank", "reciprocal_rank", "top_1", "top_5", "top_10",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(observed: float, expected: float, label: str, tolerance: float = 1e-10) -> None:
    if not np.isclose(observed, expected, rtol=0.0, atol=tolerance, equal_nan=True):
        raise RuntimeError(f"{label}: observed {observed}, expected {expected}")


def rank_biserial(values: np.ndarray) -> float:
    differences = np.asarray(values, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if differences.size == 0:
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    return float(
        (ranks[differences > 0].sum() - ranks[differences < 0].sum()) / ranks.sum()
    )


def reconstruct_rows(
    row_metadata: pd.DataFrame,
    diff_ids: list[str],
    description_embeddings: np.ndarray,
    diff_embeddings: np.ndarray,
) -> pd.DataFrame:
    desc_norms = np.linalg.norm(description_embeddings.astype(np.float64), axis=1, keepdims=True)
    diff_norms = np.linalg.norm(diff_embeddings.astype(np.float64), axis=1, keepdims=True)
    if np.any(desc_norms == 0) or np.any(diff_norms == 0):
        raise RuntimeError("Zero-norm embedding in retained output")
    similarities = (
        description_embeddings.astype(np.float64) / desc_norms
    ) @ (
        diff_embeddings.astype(np.float64) / diff_norms
    ).T
    id_to_index = {pr_id: index for index, pr_id in enumerate(diff_ids)}
    correct = np.array([id_to_index[pr_id] for pr_id in row_metadata["normalized_pr_id"]])
    rows = np.arange(len(row_metadata))
    matched = similarities[rows, correct]
    nonmatch = similarities.copy()
    nonmatch[rows, correct] = np.nan
    greater = np.sum(nonmatch > matched[:, None], axis=1)
    tied = np.sum(nonmatch == matched[:, None], axis=1)
    correct_rank = 1.0 + greater + 0.5 * tied

    result = row_metadata.copy()
    result["matched_similarity"] = matched
    result["mismatch_mean"] = np.nanmean(nonmatch, axis=1)
    result["mismatch_median"] = np.nanmedian(nonmatch, axis=1)
    result["mismatch_std"] = np.nanstd(nonmatch, axis=1, ddof=1)
    result["mismatch_q95"] = np.nanquantile(nonmatch, 0.95, axis=1)
    result["mismatch_max"] = np.nanmax(nonmatch, axis=1)
    result["margin_mean"] = matched - result["mismatch_mean"].to_numpy()
    result["margin_median"] = matched - result["mismatch_median"].to_numpy()
    result["correct_rank"] = correct_rank
    result["reciprocal_rank"] = 1.0 / correct_rank
    result["top_1"] = (correct_rank <= 1).astype(int)
    result["top_5"] = (correct_rank <= 5).astype(int)
    result["top_10"] = (correct_rank <= 10).astype(int)
    return result


def check_row_reconstruction(observed: pd.DataFrame, expected: pd.DataFrame, encoder: str) -> None:
    if len(observed) != 5630 or len(expected) != 5630:
        raise RuntimeError(f"{encoder}: expected 5,630 row diagnostics")
    for column in [*ROW_KEYS, "Context_Name"]:
        if not observed[column].astype(str).equals(expected[column].astype(str)):
            raise RuntimeError(f"{encoder}: metadata mismatch in {column}")
    for column in NUMERIC_ROW_COLUMNS:
        left = observed[column].to_numpy(dtype=float)
        right = expected[column].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0.0, atol=1e-10, equal_nan=True):
            maximum = float(np.nanmax(np.abs(left - right)))
            raise RuntimeError(f"{encoder}: {column} reconstruction differs by up to {maximum}")


def check_existing_scores(
    calibrated: pd.DataFrame, expected_path: Path, encoder: str
) -> dict[str, float | int]:
    if sha256_file(expected_path) != EXPECTED_ROW_HASHES[encoder]:
        raise RuntimeError(f"{encoder}: existing row-score hash mismatch")
    expected = pd.read_csv(expected_path, dtype={"normalized_pr_id": "string"})
    for column in ROW_KEYS:
        expected[column] = expected[column].astype("string").fillna("").str.strip()
    observed = calibrated.copy()
    for column in ROW_KEYS:
        observed[column] = observed[column].astype("string").fillna("").str.strip()
    score_column = EXPECTED_SCORE_COLUMNS[encoder]
    joined = observed[ROW_KEYS + ["matched_similarity"]].merge(
        expected[ROW_KEYS + [score_column]], on=ROW_KEYS, how="outer", validate="one_to_one",
        indicator=True,
    )
    if not (joined["_merge"] == "both").all() or len(joined) != 5630:
        raise RuntimeError(f"{encoder}: keys do not match the verified corrected run")
    differences = np.abs(
        joined["matched_similarity"].to_numpy(dtype=float)
        - joined[score_column].to_numpy(dtype=float)
    )
    maximum = float(differences.max())
    if maximum > 5e-6:
        raise RuntimeError(f"{encoder}: matched-score drift is {maximum}, exceeding 5e-6")
    return {"n": int(len(joined)), "max_abs_difference": maximum}


def check_group_summary(summary: pd.DataFrame, rows_by_encoder: dict[str, pd.DataFrame]) -> None:
    if len(summary) != 38:
        raise RuntimeError(f"Expected 38 group-summary rows, observed {len(summary)}")
    for record in summary.itertuples(index=False):
        frame = rows_by_encoder[record.encoder]
        group = frame[
            (frame["entry_type"] == record.entry_type)
            & (frame["model"] == record.model)
            & (frame["variation"] == record.variation)
            & (frame["Context_Name"] == record.context)
        ]
        if len(group) != record.n:
            raise RuntimeError(f"Group size mismatch for {record.encoder}/{record.model}/{record.variation}")
        checks = {
            "matched_mean": group["matched_similarity"].mean(),
            "matched_median": group["matched_similarity"].median(),
            "mismatch_mean": group["mismatch_mean"].mean(),
            "mismatch_median_of_row_medians": group["mismatch_median"].median(),
            "margin_mean_average": group["margin_mean"].mean(),
            "margin_mean_median": group["margin_mean"].median(),
            "margin_median_median": group["margin_median"].median(),
            "positive_margin_rate": (group["margin_mean"] > 0).mean(),
            "correct_rank_median": group["correct_rank"].median(),
            "mean_reciprocal_rank": group["reciprocal_rank"].mean(),
            "recall_at_1": group["top_1"].mean(),
            "recall_at_5": group["top_5"].mean(),
            "recall_at_10": group["top_10"].mean(),
        }
        for column, expected in checks.items():
            assert_close(float(getattr(record, column)), float(expected), f"group summary {column}")


def check_calibration_tests(table: pd.DataFrame, rows_by_encoder: dict[str, pd.DataFrame]) -> None:
    if len(table) != 38:
        raise RuntimeError(f"Expected 38 calibration tests, observed {len(table)}")
    for record in table.itertuples(index=False):
        frame = rows_by_encoder[record.encoder]
        values = frame[
            (frame["entry_type"] == record.entry_type)
            & (frame["model"] == record.model)
            & (frame["variation"] == record.variation)
            & (frame["Context_Name"] == record.context)
        ]["margin_mean"].to_numpy(dtype=float)
        test = wilcoxon(values, alternative="greater", zero_method="wilcox", method="auto")
        assert_close(record.wilcoxon_statistic, float(test.statistic), "calibration statistic")
        assert_close(record.p, float(test.pvalue), "calibration p")
        assert_close(record.rank_biserial_positive, rank_biserial(values), "calibration rrb")
    for encoder, group in table.groupby("encoder", sort=False):
        rejected, adjusted, _, _ = multipletests(group["p"].to_numpy(), method="fdr_bh")
        if not np.allclose(group["p_adj_bh"], adjusted, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{encoder}: calibration BH values differ")
        if not np.array_equal(group["significant_bh_0_05"].astype(bool), rejected):
            raise RuntimeError(f"{encoder}: calibration BH decisions differ")


def paired_reconstruction(human: pd.Series, ai: pd.Series) -> dict[str, float | int]:
    pair = pd.concat([human.rename("human"), ai.rename("ai")], axis=1, join="inner").dropna()
    differences = pair["human"].to_numpy(dtype=float) - pair["ai"].to_numpy(dtype=float)
    test = wilcoxon(pair["human"], pair["ai"], alternative="two-sided", zero_method="wilcox", method="auto")
    return {
        "n": len(pair),
        "human_median": pair["human"].median(),
        "ai_median": pair["ai"].median(),
        "median_human_minus_ai": np.median(differences),
        "wilcoxon_statistic": float(test.statistic),
        "p": float(test.pvalue),
        "rank_biserial_human_gt_ai": rank_biserial(differences),
    }


def check_pooled_tests(table: pd.DataFrame, rows_by_encoder: dict[str, pd.DataFrame]) -> None:
    if len(table) != 12:
        raise RuntimeError(f"Expected 12 pooled comparisons, observed {len(table)}")
    numeric = ["human_median", "ai_median", "median_human_minus_ai", "wilcoxon_statistic", "p", "rank_biserial_human_gt_ai"]
    for record in table.itertuples(index=False):
        frame = rows_by_encoder[record.encoder]
        human = frame[frame["entry_type"] == "human"].set_index("normalized_pr_id")[record.metric]
        ai = frame[(frame["entry_type"] == "ai") & (frame["model"] == record.model)].groupby("normalized_pr_id")[record.metric].mean()
        expected = paired_reconstruction(human, ai)
        if record.n != expected["n"]:
            raise RuntimeError("Pooled paired sample size mismatch")
        for column in numeric:
            assert_close(float(getattr(record, column)), float(expected[column]), f"pooled {column}")
    for _, group in table.groupby(["encoder", "metric"], sort=False):
        rejected, adjusted, _, _ = multipletests(group["p"].to_numpy(), method="fdr_bh")
        if not np.allclose(group["p_adj_bh"], adjusted, rtol=0.0, atol=1e-12):
            raise RuntimeError("Pooled BH values differ")
        if not np.array_equal(group["significant_bh_0_05"].astype(bool), rejected):
            raise RuntimeError("Pooled BH decisions differ")


def check_context_tests(table: pd.DataFrame, rows_by_encoder: dict[str, pd.DataFrame]) -> None:
    if len(table) != 36:
        raise RuntimeError(f"Expected 36 context comparisons, observed {len(table)}")
    numeric = ["human_median", "ai_median", "median_human_minus_ai", "wilcoxon_statistic", "p", "rank_biserial_human_gt_ai"]
    for record in table.itertuples(index=False):
        frame = rows_by_encoder[record.encoder]
        human = frame[frame["entry_type"] == "human"].set_index("normalized_pr_id")["margin_mean"]
        ai = frame[
            (frame["entry_type"] == "ai")
            & (frame["model"] == record.model)
            & (frame["variation"] == record.variation)
        ].set_index("normalized_pr_id")["margin_mean"]
        expected = paired_reconstruction(human, ai)
        if record.n != expected["n"]:
            raise RuntimeError("Context paired sample size mismatch")
        for column in numeric:
            assert_close(float(getattr(record, column)), float(expected[column]), f"context {column}")
    for encoder, group in table.groupby("encoder", sort=False):
        rejected, adjusted, _, _ = multipletests(group["p"].to_numpy(), method="fdr_bh")
        if not np.allclose(group["p_adj_bh"], adjusted, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{encoder}: context BH values differ")
        if not np.array_equal(group["significant_bh_0_05"].astype(bool), rejected):
            raise RuntimeError(f"{encoder}: context BH decisions differ")


def output_hashes(output_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output_root.rglob("*")):
        # pipeline.log remains open in the parent shell until after verification exits.
        if path.is_file() and path.name not in {"completion.json", "pipeline.log"}:
            hashes[str(path.relative_to(output_root))] = sha256_file(path)
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-codebert", type=Path, required=True)
    parser.add_argument("--expected-graphcodebert", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    metadata = json.loads((output_root / "analysis_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("protocol") != PROTOCOL:
        raise RuntimeError("Unexpected analysis protocol")

    row_metadata = pd.read_csv(output_root / "row_metadata.csv", dtype={"normalized_pr_id": "string"})
    diff_ids = pd.read_csv(output_root / "diff_ids.csv", dtype={"normalized_pr_id": "string"})["normalized_pr_id"].tolist()
    if len(row_metadata) != 5630 or len(diff_ids) != 308 or len(set(diff_ids)) != 308:
        raise RuntimeError("Retained row/diff metadata have unexpected dimensions")

    rows_by_encoder: dict[str, pd.DataFrame] = {}
    drift: dict[str, dict[str, float | int]] = {}
    expected_paths = {
        "codebert": args.expected_codebert,
        "graphcodebert": args.expected_graphcodebert,
    }
    for encoder in ["codebert", "graphcodebert"]:
        description_embeddings = np.load(
            output_root / "embeddings" / f"{encoder}_descriptions.npy", allow_pickle=False
        )
        diff_embeddings = np.load(
            output_root / "embeddings" / f"{encoder}_diffs.npy", allow_pickle=False
        )
        if description_embeddings.shape != (5630, 768) or diff_embeddings.shape != (308, 768):
            raise RuntimeError(f"{encoder}: unexpected embedding shapes")
        reconstructed = reconstruct_rows(row_metadata, diff_ids, description_embeddings, diff_embeddings)
        observed = pd.read_csv(
            output_root / "row_level" / f"{encoder}_calibration.csv",
            dtype={"normalized_pr_id": "string"},
        )
        check_row_reconstruction(observed, reconstructed, encoder)
        rows_by_encoder[encoder] = observed
        drift[encoder] = check_existing_scores(observed, expected_paths[encoder], encoder)

    check_group_summary(pd.read_csv(output_root / "summaries" / "group_summary.csv"), rows_by_encoder)
    check_calibration_tests(pd.read_csv(output_root / "summaries" / "calibration_tests.csv"), rows_by_encoder)
    check_pooled_tests(pd.read_csv(output_root / "summaries" / "pooled_human_ai_comparisons.csv"), rows_by_encoder)
    check_context_tests(pd.read_csv(output_root / "summaries" / "context_human_ai_margin_comparisons.csv"), rows_by_encoder)

    verification_dir = output_root / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": PROTOCOL,
        "status": "pass",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "row_metadata_count": len(row_metadata),
            "diff_count": len(diff_ids),
            "nonmatching_diffs_per_description": len(diff_ids) - 1,
            "embedding_shapes": "pass",
            "row_metrics_reconstructed": "pass",
            "group_summaries_reconstructed": "pass",
            "calibration_tests_reconstructed": "pass",
            "pooled_paired_tests_reconstructed": "pass",
            "context_paired_tests_reconstructed": "pass",
            "bh_corrections_reconstructed": "pass",
            "existing_matched_score_drift": drift,
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
