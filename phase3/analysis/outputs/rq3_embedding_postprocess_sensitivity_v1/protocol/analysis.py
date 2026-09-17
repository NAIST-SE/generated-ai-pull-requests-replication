#!/usr/bin/env python3
"""Run deterministic matched-versus-mismatched calibration for RQ3 embeddings."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
import torch
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests
from transformers import AutoModel, AutoTokenizer
import transformers


PROTOCOL = "rq3-mismatched-calibration-v1"
CORPUS_SHA256 = "07136384785d6d02d17762535ab51a2007a7a47b8c41b77fbb311ce9c8780306"
DIFFS_SHA256 = "45c71d0170fc9343d999998ad6515e9b5b8298f63b66215f4110b3911f681d0c"
EXPECTED_CODEBERT_SHA256 = "932dbb166dfcc221e48a7696161ed541e24057e059bc79e057bd1e331e5c1550"
EXPECTED_GRAPHCODEBERT_SHA256 = "2c96eb065c971f07fc21ffd145f9cbab097695b532454303504b8c1875b2820b"

CONTEXT_MAPPING = {
    "P-1": "Repo Path",
    "P-2": "PR Title",
    "P-3": "File Diff",
    "P-4": "File Contents",
    "P-5": "Linked Issue",
    "P-6": "Without File Contents",
}
TARGET_MODELS = [
    "meta-llama/llama-3.1-70b-instruct",
    "gpt-5-mini-2025-08-07",
    "Qwen/Qwen2.5-72B-Instruct",
]
MODEL_ALIASES = {
    "meta-llama_llama-3.1-70b-instruct": "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-70b-instruct": "meta-llama/llama-3.1-70b-instruct",
    "gpt-5-mini-2025-08-07": "gpt-5-mini-2025-08-07",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
}
ENCODERS = {
    "codebert": {
        "model_name": "microsoft/codebert-base",
        "revision": "3b0952feddeffad0063f274080e3c23d75e7eb39",
    },
    "graphcodebert": {
        "model_name": "microsoft/graphcodebert-base",
        "revision": "2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d",
    },
}
EXPECTED_CELL_COUNTS = {
    (model, variation): (
        273 if variation in {"P-5", "P-6"}
        else 302 if variation == "P-4" and model != TARGET_MODELS[0]
        else 308
    )
    for model in TARGET_MODELS
    for variation in CONTEXT_MAPPING
}
MAX_LENGTH = 512
BATCH_SIZE = 16
ROW_KEY_COLUMNS = ["normalized_pr_id", "entry_type", "model", "variation"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sha256(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected}, observed {observed}"
        )


def clean_diff(patch_text: str) -> str:
    """Reproduce the RQ3 changed-line extraction exactly."""
    if not isinstance(patch_text, str) or not patch_text.strip():
        return ""
    changed_lines: list[str] = []
    for line in patch_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@@"):
            continue
        if stripped.startswith("---") or stripped.startswith("+++"):
            continue
        if stripped.startswith("diff --git"):
            continue
        if stripped.startswith("+") or stripped.startswith("-"):
            changed_lines.append(stripped[1:].strip())
    return "\n".join(changed_lines)


def canonical_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def load_inputs(corpus_path: Path, diffs_path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    check_sha256(corpus_path, CORPUS_SHA256)
    check_sha256(diffs_path, DIFFS_SHA256)

    corpus = pd.read_csv(
        corpus_path,
        usecols=["pr_id", "entry_type", "variation", "model", "text"],
        dtype={"pr_id": "string", "entry_type": "string", "variation": "string", "model": "string"},
    ).copy()
    corpus["normalized_pr_id"] = canonical_string(corpus["pr_id"])
    corpus["entry_type"] = canonical_string(corpus["entry_type"])
    corpus["model"] = canonical_string(corpus["model"]).map(lambda value: MODEL_ALIASES.get(value, value))
    corpus["variation"] = canonical_string(corpus["variation"])

    diffs = pd.read_csv(diffs_path, usecols=["id", "patch"], dtype={"id": "string"})
    diffs["normalized_pr_id"] = canonical_string(diffs["id"])
    combined_diffs = (
        diffs.groupby("normalized_pr_id", sort=True)["patch"]
        .apply(lambda patches: "\n".join(str(patch) for patch in patches if pd.notna(patch)))
        .to_dict()
    )
    clean_diffs = {pr_id: clean_diff(text) for pr_id, text in combined_diffs.items()}

    human = corpus[corpus["entry_type"] == "human"].copy()
    human["model"] = "Human"
    human["variation"] = "original"
    human["Context_Name"] = "Original"

    ai = corpus[
        (corpus["entry_type"] == "ai")
        & corpus["model"].isin(TARGET_MODELS)
        & corpus["variation"].isin(CONTEXT_MAPPING)
    ].copy()
    ai["Context_Name"] = ai["variation"].map(CONTEXT_MAPPING)

    valid_ids = set(human["normalized_pr_id"]) & set(ai["normalized_pr_id"]) & set(clean_diffs)
    valid_ids = {pr_id for pr_id in valid_ids if clean_diffs[pr_id].strip()}
    human = human[human["normalized_pr_id"].isin(valid_ids)].copy()
    ai = ai[ai["normalized_pr_id"].isin(valid_ids)].copy()

    if len(valid_ids) != 308 or len(human) != 308 or len(ai) != 5322:
        raise RuntimeError(
            f"Unexpected population: PRs={len(valid_ids)}, human={len(human)}, AI={len(ai)}"
        )
    if human["normalized_pr_id"].duplicated().any():
        raise RuntimeError("Human descriptions are not unique by PR")
    observed_counts = ai.groupby(["model", "variation"]).size().to_dict()
    if observed_counts != EXPECTED_CELL_COUNTS:
        raise RuntimeError(f"Unexpected model/context counts: {observed_counts}")

    combined = pd.concat([human, ai], ignore_index=True)
    combined["_entry_order"] = combined["entry_type"].map({"human": 0, "ai": 1})
    combined = combined.sort_values(
        ["_entry_order", "model", "variation", "normalized_pr_id"], kind="stable"
    ).drop(columns="_entry_order").reset_index(drop=True)
    if combined[ROW_KEY_COLUMNS].duplicated().any():
        raise RuntimeError("Description keys are not unique")
    if combined["text"].isna().any():
        raise RuntimeError("Description corpus contains missing text")

    diff_ids = sorted(valid_ids)
    diff_texts = [clean_diffs[pr_id] for pr_id in diff_ids]
    if set(combined["normalized_pr_id"]) != set(diff_ids):
        raise RuntimeError("Description and diff PR identifiers do not match")
    return combined, diff_ids, diff_texts


def encode_texts(
    texts: list[str], tokenizer: AutoTokenizer, model: AutoModel, device: torch.device
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = [text if text and text.strip() else " " for text in texts[start : start + BATCH_SIZE]]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            output = model(**encoded).last_hidden_state[:, 0, :]
        batches.append(output.detach().cpu().numpy().astype(np.float32, copy=False))
        if (start // BATCH_SIZE + 1) % 50 == 0 or start + BATCH_SIZE >= len(texts):
            print(f"  encoded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(batches, axis=0)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("Encountered a zero-norm embedding")
    return matrix / norms


def calibration_frame(
    metadata: pd.DataFrame,
    diff_ids: list[str],
    description_embeddings: np.ndarray,
    diff_embeddings: np.ndarray,
) -> pd.DataFrame:
    desc = l2_normalize(description_embeddings.astype(np.float64, copy=False))
    diffs = l2_normalize(diff_embeddings.astype(np.float64, copy=False))
    similarities = desc @ diffs.T
    id_to_index = {pr_id: index for index, pr_id in enumerate(diff_ids)}
    correct_indices = np.array([id_to_index[pr_id] for pr_id in metadata["normalized_pr_id"]])
    row_indices = np.arange(len(metadata))
    matched = similarities[row_indices, correct_indices]

    nonmatch = similarities.copy()
    nonmatch[row_indices, correct_indices] = np.nan
    mismatch_mean = np.nanmean(nonmatch, axis=1)
    mismatch_median = np.nanmedian(nonmatch, axis=1)
    mismatch_std = np.nanstd(nonmatch, axis=1, ddof=1)
    mismatch_q95 = np.nanquantile(nonmatch, 0.95, axis=1)
    mismatch_max = np.nanmax(nonmatch, axis=1)

    greater = np.sum(nonmatch > matched[:, None], axis=1)
    tied = np.sum(nonmatch == matched[:, None], axis=1)
    correct_rank = 1.0 + greater + 0.5 * tied

    result = metadata.copy()
    result["matched_similarity"] = matched
    result["mismatch_mean"] = mismatch_mean
    result["mismatch_median"] = mismatch_median
    result["mismatch_std"] = mismatch_std
    result["mismatch_q95"] = mismatch_q95
    result["mismatch_max"] = mismatch_max
    result["margin_mean"] = matched - mismatch_mean
    result["margin_median"] = matched - mismatch_median
    result["correct_rank"] = correct_rank
    result["reciprocal_rank"] = 1.0 / correct_rank
    result["top_1"] = (correct_rank <= 1).astype(int)
    result["top_5"] = (correct_rank <= 5).astype(int)
    result["top_10"] = (correct_rank <= 10).astype(int)
    return result


def signed_rank_biserial(values: np.ndarray) -> float:
    differences = np.asarray(values, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if differences.size == 0:
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    denominator = float(ranks.sum())
    return float((ranks[differences > 0].sum() - ranks[differences < 0].sum()) / denominator)


def one_sample_test(values: pd.Series) -> dict[str, float | int]:
    data = values.to_numpy(dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise RuntimeError("Cannot test an empty margin series")
    if np.all(data == 0):
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(data, alternative="greater", zero_method="wilcox", method="auto")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    return {
        "n": int(data.size),
        "n_nonzero": int(np.count_nonzero(data)),
        "median_margin": float(np.median(data)),
        "mean_margin": float(np.mean(data)),
        "positive_margin_rate": float(np.mean(data > 0)),
        "wilcoxon_statistic": statistic,
        "p": p_value,
        "rank_biserial_positive": signed_rank_biserial(data),
    }


def paired_test(human: pd.Series, ai: pd.Series) -> dict[str, float | int]:
    pair = pd.concat([human.rename("human"), ai.rename("ai")], axis=1, join="inner").dropna()
    differences = pair["human"].to_numpy(dtype=float) - pair["ai"].to_numpy(dtype=float)
    if np.all(differences == 0):
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(
            pair["human"].to_numpy(dtype=float),
            pair["ai"].to_numpy(dtype=float),
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        statistic, p_value = float(test.statistic), float(test.pvalue)
    return {
        "n": int(len(pair)),
        "human_median": float(pair["human"].median()),
        "ai_median": float(pair["ai"].median()),
        "median_human_minus_ai": float(np.median(differences)),
        "wilcoxon_statistic": statistic,
        "p": p_value,
        "rank_biserial_human_gt_ai": signed_rank_biserial(differences),
    }


def apply_bh(frame: pd.DataFrame, family_columns: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    grouper: str | list[str] = family_columns[0] if len(family_columns) == 1 else family_columns
    for _, group in frame.groupby(grouper, sort=False, dropna=False):
        adjusted = group.copy()
        rejected, p_adj, _, _ = multipletests(adjusted["p"].to_numpy(), method="fdr_bh")
        adjusted["p_adj_bh"] = p_adj
        adjusted["significant_bh_0_05"] = rejected
        parts.append(adjusted)
    return pd.concat(parts, ignore_index=True)


def build_group_summary(frame: pd.DataFrame, encoder: str) -> pd.DataFrame:
    keys = ["entry_type", "model", "variation", "Context_Name"]
    rows: list[dict[str, float | int | str]] = []
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        entry_type, model, variation, context = key
        rows.append({
            "encoder": encoder,
            "entry_type": entry_type,
            "model": model,
            "variation": variation,
            "context": context,
            "n": int(len(group)),
            "matched_mean": float(group["matched_similarity"].mean()),
            "matched_median": float(group["matched_similarity"].median()),
            "mismatch_mean": float(group["mismatch_mean"].mean()),
            "mismatch_median_of_row_medians": float(group["mismatch_median"].median()),
            "margin_mean_average": float(group["margin_mean"].mean()),
            "margin_mean_median": float(group["margin_mean"].median()),
            "margin_median_median": float(group["margin_median"].median()),
            "positive_margin_rate": float((group["margin_mean"] > 0).mean()),
            "correct_rank_median": float(group["correct_rank"].median()),
            "mean_reciprocal_rank": float(group["reciprocal_rank"].mean()),
            "recall_at_1": float(group["top_1"].mean()),
            "recall_at_5": float(group["top_5"].mean()),
            "recall_at_10": float(group["top_10"].mean()),
        })
    return pd.DataFrame(rows)


def build_calibration_tests(frame: pd.DataFrame, encoder: str) -> pd.DataFrame:
    keys = ["entry_type", "model", "variation", "Context_Name"]
    rows: list[dict[str, float | int | str]] = []
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        entry_type, model, variation, context = key
        rows.append({
            "encoder": encoder,
            "entry_type": entry_type,
            "model": model,
            "variation": variation,
            "context": context,
            **one_sample_test(group["margin_mean"]),
        })
    result = pd.DataFrame(rows)
    return apply_bh(result, ["encoder"])


def build_pooled_comparisons(frame: pd.DataFrame, encoder: str) -> pd.DataFrame:
    human = frame[frame["entry_type"] == "human"].set_index("normalized_pr_id")
    rows: list[dict[str, float | int | str]] = []
    for metric in ["matched_similarity", "margin_mean"]:
        human_metric = human[metric]
        for model in TARGET_MODELS:
            ai_metric = (
                frame[(frame["entry_type"] == "ai") & (frame["model"] == model)]
                .groupby("normalized_pr_id")[metric]
                .mean()
            )
            rows.append({
                "encoder": encoder,
                "metric": metric,
                "model": model,
                **paired_test(human_metric, ai_metric),
            })
    return apply_bh(pd.DataFrame(rows), ["encoder", "metric"])


def build_context_comparisons(frame: pd.DataFrame, encoder: str) -> pd.DataFrame:
    human = (
        frame[frame["entry_type"] == "human"]
        .set_index("normalized_pr_id")["margin_mean"]
    )
    rows: list[dict[str, float | int | str]] = []
    for model in TARGET_MODELS:
        for variation, context in CONTEXT_MAPPING.items():
            ai = (
                frame[
                    (frame["entry_type"] == "ai")
                    & (frame["model"] == model)
                    & (frame["variation"] == variation)
                ]
                .set_index("normalized_pr_id")["margin_mean"]
            )
            rows.append({
                "encoder": encoder,
                "metric": "margin_mean",
                "model": model,
                "variation": variation,
                "context": context,
                **paired_test(human, ai),
            })
    return apply_bh(pd.DataFrame(rows), ["encoder"])


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def run_encoder(
    encoder: str,
    config: dict[str, str],
    combined: pd.DataFrame,
    diff_ids: list[str],
    diff_texts: list[str],
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    print(f"\n=== {encoder}: {config['model_name']} @ {config['revision']} ===", flush=True)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], revision=config["revision"], local_files_only=True
    )
    model = AutoModel.from_pretrained(
        config["model_name"], revision=config["revision"], local_files_only=True
    ).to(device)
    model.eval()

    started = time.monotonic()
    print("Encoding PR diffs...", flush=True)
    diff_embeddings = encode_texts(diff_texts, tokenizer, model, device)
    print("Encoding descriptions...", flush=True)
    description_embeddings = encode_texts(
        [str(text) for text in combined["text"].tolist()], tokenizer, model, device
    )
    elapsed = time.monotonic() - started

    embedding_dir = output_root / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    np.save(embedding_dir / f"{encoder}_descriptions.npy", description_embeddings, allow_pickle=False)
    np.save(embedding_dir / f"{encoder}_diffs.npy", diff_embeddings, allow_pickle=False)

    metadata = combined[
        ["normalized_pr_id", "entry_type", "model", "variation", "Context_Name"]
    ].copy()
    calibrated = calibration_frame(metadata, diff_ids, description_embeddings, diff_embeddings)
    row_dir = output_root / "row_level"
    row_dir.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(row_dir / f"{encoder}_calibration.csv", index=False)

    del model, tokenizer
    torch.cuda.empty_cache()
    return calibrated, {
        "model_name": config["model_name"],
        "revision": config["revision"],
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "representation": "first token of final hidden state",
        "data_flow_graph_supplied": False if encoder == "graphcodebert" else None,
        "description_embedding_shape": list(description_embeddings.shape),
        "diff_embedding_shape": list(diff_embeddings.shape),
        "elapsed_seconds": elapsed,
    }


def self_test() -> None:
    metadata = pd.DataFrame({
        "normalized_pr_id": ["a", "b", "c"],
        "entry_type": ["human"] * 3,
        "model": ["Human"] * 3,
        "variation": ["original"] * 3,
        "Context_Name": ["Original"] * 3,
    })
    identity = np.eye(3, dtype=np.float32)
    result = calibration_frame(metadata, ["a", "b", "c"], identity, identity)
    if not np.allclose(result["matched_similarity"], 1.0):
        raise RuntimeError("Self-test failed: matched similarities")
    if not np.allclose(result["mismatch_mean"], 0.0):
        raise RuntimeError("Self-test failed: mismatch baseline")
    if not np.allclose(result["correct_rank"], 1.0):
        raise RuntimeError("Self-test failed: correct ranks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--diffs", type=Path, required=True)
    parser.add_argument("--expected-codebert", type=Path, required=True)
    parser.add_argument("--expected-graphcodebert", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    self_test()
    check_sha256(args.expected_codebert, EXPECTED_CODEBERT_SHA256)
    check_sha256(args.expected_graphcodebert, EXPECTED_GRAPHCODEBERT_SHA256)
    combined, diff_ids, diff_texts = load_inputs(args.corpus, args.diffs)

    if args.preflight:
        print(json.dumps({
            "status": "preflight_pass",
            "protocol": PROTOCOL,
            "descriptions": len(combined),
            "human": int((combined["entry_type"] == "human").sum()),
            "ai": int((combined["entry_type"] == "ai").sum()),
            "diffs": len(diff_ids),
            "incorrect_diffs_per_description": len(diff_ids) - 1,
        }, indent=2, sort_keys=True))
        return

    if args.output_root is None:
        raise RuntimeError("--output-root is required outside preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full calibration run")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"normalized_pr_id": diff_ids}).to_csv(output_root / "diff_ids.csv", index=False)
    combined[
        ["normalized_pr_id", "entry_type", "model", "variation", "Context_Name"]
    ].to_csv(output_root / "row_metadata.csv", index=False)

    all_group_summaries: list[pd.DataFrame] = []
    all_calibration_tests: list[pd.DataFrame] = []
    all_pooled_comparisons: list[pd.DataFrame] = []
    all_context_comparisons: list[pd.DataFrame] = []
    encoder_metadata: dict[str, object] = {}

    for encoder, config in ENCODERS.items():
        calibrated, model_metadata = run_encoder(
            encoder, config, combined, diff_ids, diff_texts, output_root
        )
        all_group_summaries.append(build_group_summary(calibrated, encoder))
        all_calibration_tests.append(build_calibration_tests(calibrated, encoder))
        all_pooled_comparisons.append(build_pooled_comparisons(calibrated, encoder))
        all_context_comparisons.append(build_context_comparisons(calibrated, encoder))
        encoder_metadata[encoder] = model_metadata

    summary_dir = output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(all_group_summaries, ignore_index=True).to_csv(
        summary_dir / "group_summary.csv", index=False
    )
    pd.concat(all_calibration_tests, ignore_index=True).to_csv(
        summary_dir / "calibration_tests.csv", index=False
    )
    pd.concat(all_pooled_comparisons, ignore_index=True).to_csv(
        summary_dir / "pooled_human_ai_comparisons.csv", index=False
    )
    pd.concat(all_context_comparisons, ignore_index=True).to_csv(
        summary_dir / "context_human_ai_margin_comparisons.csv", index=False
    )

    chance_mrr = float(sum(1.0 / rank for rank in range(1, len(diff_ids) + 1)) / len(diff_ids))
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
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": package_version("tokenizers"),
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
        },
        "inputs": {
            "corpus": str(args.corpus.resolve()),
            "corpus_sha256": CORPUS_SHA256,
            "diffs": str(args.diffs.resolve()),
            "diffs_sha256": DIFFS_SHA256,
            "expected_codebert": str(args.expected_codebert.resolve()),
            "expected_codebert_sha256": EXPECTED_CODEBERT_SHA256,
            "expected_graphcodebert": str(args.expected_graphcodebert.resolve()),
            "expected_graphcodebert_sha256": EXPECTED_GRAPHCODEBERT_SHA256,
        },
        "population": {
            "descriptions": len(combined),
            "human_descriptions": int((combined["entry_type"] == "human").sum()),
            "ai_descriptions": int((combined["entry_type"] == "ai").sum()),
            "pr_diffs": len(diff_ids),
            "nonmatching_diffs_per_description": len(diff_ids) - 1,
        },
        "random_retrieval_reference": {
            "recall_at_1": 1.0 / len(diff_ids),
            "recall_at_5": 5.0 / len(diff_ids),
            "recall_at_10": 10.0 / len(diff_ids),
            "expected_mrr": chance_mrr,
            "expected_mean_rank": (len(diff_ids) + 1) / 2,
        },
        "encoders": encoder_metadata,
        "analysis": {
            "primary_margin": "matched cosine minus mean cosine over all 307 nonmatching diffs",
            "nonmatching_reference": "all incorrect PR diffs; deterministic; no random shuffle",
            "calibration_test": "one-sided Wilcoxon margin > 0; BH within 19 source cells per encoder",
            "human_ai_test": "two-sided paired Wilcoxon by PR; matched-pairs rank-biserial; BH",
            "postprocessing": "none",
        },
    }
    (output_root / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\nCalibration analysis complete; independent verification is still required.")


if __name__ == "__main__":
    main()
