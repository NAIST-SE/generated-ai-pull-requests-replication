#!/usr/bin/env python3
"""Run the frozen centering/top-component sensitivity on retained RQ3 embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import analysis as core


PROTOCOL = "rq3-embedding-postprocess-sensitivity-v1"
VARIANTS = {
    "centered": 0,
    "remove_top_1": 1,
    "remove_top_2": 2,
    "remove_top_3": 3,
}
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


def verify_inputs(calibration_root: Path) -> None:
    completion = json.loads((calibration_root / "completion.json").read_text(encoding="utf-8"))
    if completion.get("protocol") != core.PROTOCOL or completion.get("status") != "complete_and_verified":
        raise RuntimeError("The retained calibration input is not complete and verified")
    for relative, expected in INPUT_HASHES.items():
        path = calibration_root / relative
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"Retained input hash mismatch for {relative}: expected {expected}, observed {observed}"
            )


def remove_components(centered: np.ndarray, components: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return centered.copy()
    selected = components[:count]
    return centered - (centered @ selected.T) @ selected


def add_variant(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    result = frame.copy()
    result.insert(1, "variant", variant)
    return result


def run_encoder(
    encoder: str,
    calibration_root: Path,
    output_root: Path,
    row_metadata: pd.DataFrame,
    diff_ids: list[str],
) -> tuple[list[pd.DataFrame], dict[str, object]]:
    descriptions = np.load(
        calibration_root / "embeddings" / f"{encoder}_descriptions.npy", allow_pickle=False
    ).astype(np.float64)
    diffs = np.load(
        calibration_root / "embeddings" / f"{encoder}_diffs.npy", allow_pickle=False
    ).astype(np.float64)
    if descriptions.shape != (5630, 768) or diffs.shape != (308, 768):
        raise RuntimeError(f"{encoder}: unexpected retained embedding shapes")

    human_mask = row_metadata["entry_type"].to_numpy() == "human"
    human_descriptions = descriptions[human_mask]
    if human_descriptions.shape != (308, 768):
        raise RuntimeError(f"{encoder}: balanced reference requires 308 human descriptions")
    reference = np.vstack([human_descriptions, diffs])
    common_mean = reference.mean(axis=0)
    reference_centered = reference - common_mean
    _, singular_values, components = np.linalg.svd(reference_centered, full_matrices=False)
    explained = singular_values ** 2
    explained_ratio = explained / explained.sum()

    transform_dir = output_root / "transforms"
    transform_dir.mkdir(parents=True, exist_ok=True)
    np.save(transform_dir / f"{encoder}_common_mean.npy", common_mean, allow_pickle=False)
    np.save(transform_dir / f"{encoder}_components.npy", components[:3], allow_pickle=False)
    np.save(transform_dir / f"{encoder}_singular_values.npy", singular_values, allow_pickle=False)

    description_centered = descriptions - common_mean
    diff_centered = diffs - common_mean
    frames: list[pd.DataFrame] = []
    row_dir = output_root / "row_level"
    row_dir.mkdir(parents=True, exist_ok=True)
    for variant, component_count in VARIANTS.items():
        print(f"{encoder}: {variant}", flush=True)
        transformed_descriptions = remove_components(
            description_centered, components, component_count
        )
        transformed_diffs = remove_components(diff_centered, components, component_count)
        calibrated = core.calibration_frame(
            row_metadata, diff_ids, transformed_descriptions, transformed_diffs
        )
        calibrated.insert(1, "variant", variant)
        calibrated.to_csv(row_dir / f"{encoder}_{variant}.csv", index=False)
        frames.append(calibrated)

    geometry = {
        "reference": "308 human-description embeddings + 308 PR-diff embeddings",
        "reference_shape": list(reference.shape),
        "common_mean_l2_norm": float(np.linalg.norm(common_mean)),
        "top_10_explained_variance_ratio": [float(value) for value in explained_ratio[:10]],
        "top_1_explained_variance_ratio": float(explained_ratio[0]),
        "top_3_cumulative_explained_variance_ratio": float(explained_ratio[:3].sum()),
    }
    return frames, geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_root = args.calibration_root.resolve()
    verify_inputs(calibration_root)
    row_metadata = pd.read_csv(
        calibration_root / "row_metadata.csv", dtype={"normalized_pr_id": "string"}
    )
    diff_ids = pd.read_csv(
        calibration_root / "diff_ids.csv", dtype={"normalized_pr_id": "string"}
    )["normalized_pr_id"].tolist()
    if len(row_metadata) != 5630 or len(diff_ids) != 308:
        raise RuntimeError("Unexpected retained metadata dimensions")
    if args.preflight:
        print(json.dumps({
            "status": "preflight_pass",
            "protocol": PROTOCOL,
            "calibration_protocol": core.PROTOCOL,
            "variants": ["raw", *VARIANTS],
            "reference_vectors_per_encoder": 616,
            "model_inference": False,
        }, indent=2, sort_keys=True))
        return
    if args.output_root is None:
        raise RuntimeError("--output-root is required outside preflight")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    group_summaries: list[pd.DataFrame] = []
    calibration_tests: list[pd.DataFrame] = []
    pooled_comparisons: list[pd.DataFrame] = []
    context_comparisons: list[pd.DataFrame] = []
    geometry: dict[str, object] = {}

    for encoder in core.ENCODERS:
        raw = pd.read_csv(
            calibration_root / "row_level" / f"{encoder}_calibration.csv",
            dtype={"normalized_pr_id": "string"},
        )
        raw.insert(1, "variant", "raw")
        frames, encoder_geometry = run_encoder(
            encoder, calibration_root, output_root, row_metadata, diff_ids
        )
        geometry[encoder] = encoder_geometry
        for frame in [raw, *frames]:
            variant = str(frame["variant"].iloc[0])
            without_variant = frame.drop(columns="variant")
            group_summaries.append(add_variant(core.build_group_summary(without_variant, encoder), variant))
            calibration_tests.append(add_variant(core.build_calibration_tests(without_variant, encoder), variant))
            pooled_comparisons.append(add_variant(core.build_pooled_comparisons(without_variant, encoder), variant))
            context_comparisons.append(add_variant(core.build_context_comparisons(without_variant, encoder), variant))

    summary_dir = output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(group_summaries, ignore_index=True).to_csv(
        summary_dir / "group_summary_all_variants.csv", index=False
    )
    pd.concat(calibration_tests, ignore_index=True).to_csv(
        summary_dir / "calibration_tests_all_variants.csv", index=False
    )
    pd.concat(pooled_comparisons, ignore_index=True).to_csv(
        summary_dir / "pooled_human_ai_all_variants.csv", index=False
    )
    pd.concat(context_comparisons, ignore_index=True).to_csv(
        summary_dir / "context_human_ai_margins_all_variants.csv", index=False
    )

    metadata = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "input_protocol": core.PROTOCOL,
        "input_root": str(calibration_root),
        "input_hashes": INPUT_HASHES,
        "fit_reference": "balanced common reference: 308 human descriptions + 308 diffs",
        "fit_uses_ai_descriptions": False,
        "same_transform_applied_to_both_modalities": True,
        "variants": {"raw": None, **VARIANTS},
        "selection_rule": "report all variants; no post-result selection",
        "geometry": geometry,
    }
    (output_root / "sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Post-processing sensitivity complete; independent verification is required.")


if __name__ == "__main__":
    main()
