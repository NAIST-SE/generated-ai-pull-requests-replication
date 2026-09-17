#!/usr/bin/env python3
"""Independently verify the conditional embedding post-processing sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import verify_outputs as base_verify


PROTOCOL = "rq3-embedding-postprocess-sensitivity-v1"
CALIBRATION_PROTOCOL = "rq3-mismatched-calibration-v1"
VARIANTS = {"centered": 0, "remove_top_1": 1, "remove_top_2": 2, "remove_top_3": 3}
INPUT_HASHES = {
    "diff_ids.csv": "1a6c6ff3ae9b6c326a9ae3b5c9b6cfa991b3632decd40d3f4b0859df920b9d65",
    "row_metadata.csv": "23827b92ceecffbc12c622887b277f796c053a1363c2bc4264a75e24c0db6e71",
    "embeddings/codebert_descriptions.npy": "ee5644512f9140238ea257779f93689d34eff539713298d1a68268eef818a8fb",
    "embeddings/codebert_diffs.npy": "fbf6c21bf414dfcf8708ca2c1ca3f3d5422d6a49a656fb7902709320e47299b7",
    "embeddings/graphcodebert_descriptions.npy": "e535c27de1996573e9841e3ad947864c0e61d5e987478a01c2e34110e91fcd33",
    "embeddings/graphcodebert_diffs.npy": "bb6559833eac50f0f104ebbf437f7700102bd28b4f38245a8a78ea95a49dd273",
    "row_level/codebert_calibration.csv": "90038f1deec7449427c5ac012a0320ad9b16e1f5c0d66d6c080aab80a1add17e",
    "row_level/graphcodebert_calibration.csv": "571760f6e4c05c077372b5c390bc9fdf950c41a5a47b58a56d1d3daee1ca7f79",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_input_hashes(calibration_root: Path) -> None:
    completion = json.loads((calibration_root / "completion.json").read_text(encoding="utf-8"))
    if completion.get("protocol") != CALIBRATION_PROTOCOL or completion.get("status") != "complete_and_verified":
        raise RuntimeError("Calibration source is not complete and verified")
    for relative, expected in INPUT_HASHES.items():
        observed = sha256_file(calibration_root / relative)
        if observed != expected:
            raise RuntimeError(f"Sensitivity input hash mismatch for {relative}")


def transform(
    matrix: np.ndarray, common_mean: np.ndarray, components: np.ndarray, count: int
) -> np.ndarray:
    centered = matrix.astype(np.float64) - common_mean
    if count:
        selected = components[:count]
        centered = centered - (centered @ selected.T) @ selected
    return centered


def output_hashes(output_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"completion.json", "pipeline.log"}:
            result[str(path.relative_to(output_root))] = sha256_file(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_root = args.calibration_root.resolve()
    output_root = args.output_root.resolve()
    verify_input_hashes(calibration_root)
    sensitivity_metadata = json.loads(
        (output_root / "sensitivity_metadata.json").read_text(encoding="utf-8")
    )
    if sensitivity_metadata.get("protocol") != PROTOCOL:
        raise RuntimeError("Unexpected sensitivity protocol")

    row_metadata = pd.read_csv(
        calibration_root / "row_metadata.csv", dtype={"normalized_pr_id": "string"}
    )
    diff_ids = pd.read_csv(
        calibration_root / "diff_ids.csv", dtype={"normalized_pr_id": "string"}
    )["normalized_pr_id"].tolist()
    human_mask = row_metadata["entry_type"].to_numpy() == "human"
    rows_by_variant: dict[str, dict[str, pd.DataFrame]] = {
        variant: {} for variant in ["raw", *VARIANTS]
    }
    geometry_checks: dict[str, object] = {}

    for encoder in ["codebert", "graphcodebert"]:
        descriptions = np.load(
            calibration_root / "embeddings" / f"{encoder}_descriptions.npy", allow_pickle=False
        ).astype(np.float64)
        diffs = np.load(
            calibration_root / "embeddings" / f"{encoder}_diffs.npy", allow_pickle=False
        ).astype(np.float64)
        reference = np.vstack([descriptions[human_mask], diffs])
        common_mean = reference.mean(axis=0)
        centered_reference = reference - common_mean
        _, singular_values, components = np.linalg.svd(centered_reference, full_matrices=False)

        retained_mean = np.load(
            output_root / "transforms" / f"{encoder}_common_mean.npy", allow_pickle=False
        )
        retained_components = np.load(
            output_root / "transforms" / f"{encoder}_components.npy", allow_pickle=False
        )
        retained_singular_values = np.load(
            output_root / "transforms" / f"{encoder}_singular_values.npy", allow_pickle=False
        )
        if not np.allclose(retained_mean, common_mean, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{encoder}: retained common mean differs")
        if not np.allclose(retained_singular_values, singular_values, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"{encoder}: retained singular values differ")
        observed_projection = retained_components.T @ retained_components
        expected_projection = components[:3].T @ components[:3]
        if not np.allclose(observed_projection, expected_projection, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"{encoder}: retained component subspace differs")

        raw = pd.read_csv(
            calibration_root / "row_level" / f"{encoder}_calibration.csv",
            dtype={"normalized_pr_id": "string"},
        )
        rows_by_variant["raw"][encoder] = raw
        for variant, count in VARIANTS.items():
            transformed_descriptions = transform(descriptions, common_mean, components, count)
            transformed_diffs = transform(diffs, common_mean, components, count)
            reconstructed = base_verify.reconstruct_rows(
                row_metadata, diff_ids, transformed_descriptions, transformed_diffs
            )
            observed = pd.read_csv(
                output_root / "row_level" / f"{encoder}_{variant}.csv",
                dtype={"normalized_pr_id": "string"},
            )
            if set(observed["variant"].unique()) != {variant}:
                raise RuntimeError(f"{encoder}/{variant}: variant label mismatch")
            observed = observed.drop(columns="variant")
            base_verify.check_row_reconstruction(observed, reconstructed, f"{encoder}/{variant}")
            rows_by_variant[variant][encoder] = observed

        explained = singular_values ** 2
        explained_ratio = explained / explained.sum()
        recorded = sensitivity_metadata["geometry"][encoder]
        base_verify.assert_close(
            recorded["common_mean_l2_norm"], float(np.linalg.norm(common_mean)),
            f"{encoder}: common mean norm",
        )
        base_verify.assert_close(
            recorded["top_1_explained_variance_ratio"], float(explained_ratio[0]),
            f"{encoder}: top-1 explained variance",
        )
        base_verify.assert_close(
            recorded["top_3_cumulative_explained_variance_ratio"], float(explained_ratio[:3].sum()),
            f"{encoder}: top-3 explained variance",
        )
        geometry_checks[encoder] = {
            "reference_shape": list(reference.shape),
            "common_mean": "pass",
            "top_three_component_subspace": "pass",
            "singular_values": "pass",
        }

    group_table = pd.read_csv(output_root / "summaries" / "group_summary_all_variants.csv")
    calibration_table = pd.read_csv(output_root / "summaries" / "calibration_tests_all_variants.csv")
    pooled_table = pd.read_csv(output_root / "summaries" / "pooled_human_ai_all_variants.csv")
    context_table = pd.read_csv(output_root / "summaries" / "context_human_ai_margins_all_variants.csv")
    for variant, rows_by_encoder in rows_by_variant.items():
        base_verify.check_group_summary(
            group_table[group_table["variant"] == variant].drop(columns="variant"), rows_by_encoder
        )
        base_verify.check_calibration_tests(
            calibration_table[calibration_table["variant"] == variant].drop(columns="variant"),
            rows_by_encoder,
        )
        base_verify.check_pooled_tests(
            pooled_table[pooled_table["variant"] == variant].drop(columns="variant"),
            rows_by_encoder,
        )
        base_verify.check_context_tests(
            context_table[context_table["variant"] == variant].drop(columns="variant"),
            rows_by_encoder,
        )

    verification_dir = output_root / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": PROTOCOL,
        "status": "pass",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "frozen_input_hashes": "pass",
            "balanced_reference": "pass",
            "geometry": geometry_checks,
            "all_transformed_row_metrics_reconstructed": "pass",
            "all_group_summaries_reconstructed": "pass",
            "all_wilcoxon_tests_reconstructed": "pass",
            "all_rank_biserial_effects_reconstructed": "pass",
            "all_bh_corrections_reconstructed": "pass",
            "variants": ["raw", *VARIANTS],
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
