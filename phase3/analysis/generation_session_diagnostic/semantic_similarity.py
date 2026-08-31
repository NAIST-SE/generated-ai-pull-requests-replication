#!/usr/bin/env python3
"""Offline pinned BERTScore comparison for the generation-session diagnostic."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "generation-session-diagnostic-bertscore-v1"
DEBERTA_REVISION = "5b07a9086c1dbb79981ff7b05b4d1ad83b3af51c"
MODEL_NAME = "microsoft/deberta-xlarge-mnli"
SAFE_MAX_LENGTH = 512
BATCH_SIZE = 8
CHUNK_SIZE = 64


def main() -> None:
    hostname = socket.gethostname().split(".")[0].lower()
    if hostname != "nokogiri":
        raise SystemExit(f"Refusing to run on {hostname!r}; expected nokogiri")

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    sys.path.insert(0, str(project_root))
    import generation_session_diagnostic as diagnostic

    output_dir = (
        project_root
        / "phase3"
        / "analysis"
        / "outputs"
        / "tier_b"
        / diagnostic.PROTOCOL_VERSION
    )
    verification_path = output_dir / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") not in {
        "complete_and_verified",
        "partial_generation_with_verified_failures",
    }:
        raise RuntimeError("Diagnostic outputs are not structurally verified")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from bert_score import BERTScorer

    if not torch.cuda.is_available():
        raise RuntimeError("Pinned diagnostic BERTScore requires CUDA")
    model_snapshot = Path(
        "/home/ayo-b/.cache/huggingface/hub/"
        f"models--microsoft--deberta-xlarge-mnli/snapshots/{DEBERTA_REVISION}"
    )
    baseline_path = Path(
        "/home/ayo-b/.pyenv/versions/3.12.9/lib/python3.12/site-packages/"
        "bert_score/rescale_baseline/en/microsoft/deberta-xlarge-mnli.tsv"
    )
    if not (model_snapshot / "config.json").is_file():
        raise RuntimeError(f"Pinned model snapshot is absent: {model_snapshot}")
    if not baseline_path.is_file():
        raise RuntimeError(f"Pinned rescaling baseline is absent: {baseline_path}")

    metadata_rows: list[dict[str, Any]] = []
    current_texts: list[str] = []
    historical_texts: list[str] = []
    for label in ("qwen", "gpt"):
        old = diagnostic.historical_index(project_root, label)
        current = diagnostic.read_generation(output_dir / f"{label}_current.csv")
        current = current.loc[diagnostic.successful_mask(current)].copy()
        for row in current.itertuples(index=False):
            historical = old.loc[(row.pr_id, row.variation)]
            old_text = diagnostic.normalized_text(str(historical["generated_text"]))
            new_text = diagnostic.normalized_text(str(row.generated_text))
            metadata_rows.append(
                {
                    "model_label": label,
                    "model": row.model,
                    "pr_id": row.pr_id,
                    "variation": row.variation,
                    "old_text_sha256": diagnostic.sha256_text(old_text),
                    "new_text_sha256": diagnostic.sha256_text(new_text),
                }
            )
            historical_texts.append(old_text)
            current_texts.append(new_text)
    if len(metadata_rows) != int(verification["successful_rows"]):
        raise RuntimeError("Semantic input count differs from verified successful rows")

    scorer = BERTScorer(
        model_type=str(model_snapshot),
        num_layers=40,
        rescale_with_baseline=True,
        baseline_path=str(baseline_path),
        lang="en",
        device="cuda",
        batch_size=BATCH_SIZE,
    )
    configured = getattr(scorer._tokenizer, "model_max_length", SAFE_MAX_LENGTH)
    if configured is None or int(configured) > 1_000_000:
        configured = SAFE_MAX_LENGTH
    scorer._tokenizer.model_max_length = min(int(configured), SAFE_MAX_LENGTH)

    precision_parts: list[float] = []
    recall_parts: list[float] = []
    f1_parts: list[float] = []
    with torch.no_grad():
        for start in range(0, len(metadata_rows), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(metadata_rows))
            precision, recall, f1 = scorer.score(
                current_texts[start:end], historical_texts[start:end]
            )
            precision_parts.extend(precision.detach().cpu().numpy().tolist())
            recall_parts.extend(recall.detach().cpu().numpy().tolist())
            f1_parts.extend(f1.detach().cpu().numpy().tolist())
            print(f"BERTScore progress: {end}/{len(metadata_rows)}", flush=True)

    tokenizer = scorer._tokenizer
    new_token_counts = [
        len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        for text in current_texts
    ]
    old_token_counts = [
        len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        for text in historical_texts
    ]
    pairs = pd.DataFrame(metadata_rows)
    pairs["bertscore_precision"] = precision_parts
    pairs["bertscore_recall"] = recall_parts
    pairs["bertscore_f1"] = f1_parts
    pairs["new_model_tokens"] = new_token_counts
    pairs["old_model_tokens"] = old_token_counts
    pairs["new_truncated"] = pairs["new_model_tokens"] > SAFE_MAX_LENGTH
    pairs["old_truncated"] = pairs["old_model_tokens"] > SAFE_MAX_LENGTH
    pairs["pair_truncated"] = pairs["new_truncated"] | pairs["old_truncated"]
    pairs = pairs.sort_values(["model_label", "variation", "pr_id"])

    lexical = pd.read_csv(
        output_dir / "paired_comparison.csv", dtype=str, keep_default_na=False
    )
    semantic_keys = set(
        pairs[["model_label", "pr_id", "variation"]].itertuples(index=False, name=None)
    )
    lexical_keys = set(
        lexical[["model_label", "pr_id", "variation"]].itertuples(index=False, name=None)
    )
    if semantic_keys != lexical_keys:
        raise RuntimeError("Semantic and lexical pair sets differ")

    summary_rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = []
    for (label, variation), group in pairs.groupby(
        ["model_label", "variation"], sort=True
    ):
        groups.append((label, variation, group))
    for label, group in pairs.groupby("model_label", sort=True):
        groups.append((label, "ALL_P1_P4", group))
    groups.append(("ALL_MODELS", "ALL_P1_P4", pairs))
    for index, (label, variation, group) in enumerate(groups):
        if label == "ALL_MODELS":
            expected_pairs = 400
        elif variation == "ALL_P1_P4":
            expected_pairs = 200
        else:
            expected_pairs = 50
        values = group["bertscore_f1"].to_numpy(float)
        interval = diagnostic.bootstrap_interval(
            values,
            "median",
            seed=diagnostic.SELECTION_SEED + 10_000 + index,
        )
        summary_rows.append(
            {
                "model_label": label,
                "variation": variation,
                "n_pairs": len(group),
                "expected_pairs": expected_pairs,
                "missing_pairs": expected_pairs - len(group),
                "bertscore_f1_median": float(np.median(values)),
                "bertscore_f1_median_ci_low": interval[0],
                "bertscore_f1_median_ci_high": interval[1],
                "bertscore_f1_mean": float(np.mean(values)),
                "pair_truncated_count": int(group["pair_truncated"].sum()),
                "pair_truncated_rate": float(group["pair_truncated"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    pair_path = output_dir / "bertscore_pairs.csv"
    summary_path = output_dir / "bertscore_summary.csv"
    pairs.to_csv(pair_path, index=False)
    summary.to_csv(summary_path, index=False)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": diagnostic.utc_now(),
        "generation_protocol_version": diagnostic.PROTOCOL_VERSION,
        "verification_sha256": diagnostic.sha256_file(verification_path),
        "model_name": MODEL_NAME,
        "model_revision": DEBERTA_REVISION,
        "model_config_sha256": diagnostic.sha256_file(model_snapshot / "config.json"),
        "baseline_sha256": diagnostic.sha256_file(baseline_path),
        "rescale_with_baseline": True,
        "idf": False,
        "tokenizer_max_length": SAFE_MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "chunk_size": CHUNK_SIZE,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "pairs": len(pairs),
        "expected_pairs": 400,
        "missing_pairs": 400 - len(pairs),
        "raw_text_in_outputs": False,
        "pair_csv_sha256": diagnostic.sha256_file(pair_path),
        "summary_csv_sha256": diagnostic.sha256_file(summary_path),
        "summary": summary.to_dict(orient="records"),
        "limitations": [
            "BERTScore is a semantic-similarity diagnostic, not an outcome-equivalence test.",
            "Texts exceeding 512 DeBERTa tokens are truncated and reported per pair.",
            "The 11 missing Qwen P4 pairs are non-random with respect to prompt length.",
        ],
    }
    diagnostic.write_json(output_dir / "bertscore_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
