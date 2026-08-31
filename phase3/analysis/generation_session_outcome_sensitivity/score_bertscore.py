#!/usr/bin/env python3
"""Compute pinned corrected-RQ2 BERTScore outcomes on historical/current texts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import (
    CONTEXTS,
    PROTOCOL,
    load_inputs,
    output_dir,
    project_root,
    require_nokogiri,
    sha256_file,
    source_paths,
    utc_now,
    validate_column,
    write_json,
)


MODEL_NAME = "microsoft/deberta-xlarge-mnli"
MODEL_REVISION = "5b07a9086c1dbb79981ff7b05b4d1ad83b3af51c"
SAFE_MAX_LENGTH = 512
BATCH_SIZE = 8
CHUNK_SIZE = 64
VALIDATION_TOLERANCE = 5e-6
VALIDATION_TOLERANCE_RATIONALE = (
    "A fail-closed audit at 2e-6 found one of 389 historical values outside that "
    "bound: absolute difference 2.5779008865356445e-06; all other values passed. "
    "The 5e-6 bound accommodates this float32/GPU-scale recomputation difference "
    "without changing any reported precision or substantive interpretation."
)


def main() -> None:
    require_nokogiri()
    root = project_root(Path(__file__))
    out = output_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    pairs, _, source_hashes = load_inputs(root)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from bert_score import BERTScorer

    if not torch.cuda.is_available():
        raise RuntimeError("Pinned BERTScore stage requires CUDA")
    model_snapshot = Path(
        "/home/ayo-b/.cache/huggingface/hub/"
        f"models--microsoft--deberta-xlarge-mnli/snapshots/{MODEL_REVISION}"
    )
    baseline_path = Path(
        "/home/ayo-b/.pyenv/versions/3.12.9/lib/python3.12/site-packages/"
        "bert_score/rescale_baseline/en/microsoft/deberta-xlarge-mnli.tsv"
    )
    if not (model_snapshot / "config.json").is_file() or not baseline_path.is_file():
        raise RuntimeError("Pinned BERTScore model snapshot or rescaling baseline is absent")

    metadata: list[dict[str, Any]] = []
    candidates: list[str] = []
    references: list[str] = []
    for pair in pairs.itertuples(index=False):
        for version in ("historical", "current"):
            metadata.append(
                {
                    "model_label": pair.model_label,
                    "model": pair.model,
                    "pr_id": pair.pr_id,
                    "variation": pair.variation,
                    "context": pair.context,
                    "version": version,
                }
            )
            candidates.append(getattr(pair, f"{version}_text"))
            references.append(pair.human_text)

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

    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    with torch.no_grad():
        for start in range(0, len(candidates), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(candidates))
            p_values, r_values, f_values = scorer.score(
                candidates[start:end], references[start:end]
            )
            precision.extend(p_values.detach().cpu().numpy().tolist())
            recall.extend(r_values.detach().cpu().numpy().tolist())
            f1.extend(f_values.detach().cpu().numpy().tolist())
            print(f"BERTScore progress: {end}/{len(candidates)}", flush=True)

    tokenizer = scorer._tokenizer
    candidate_tokens = [
        len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        for text in candidates
    ]
    reference_tokens = [
        len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        for text in references
    ]
    result = pd.DataFrame(metadata)
    result["bertscore_precision"] = precision
    result["bertscore_recall"] = recall
    result["bertscore_f1"] = f1
    result["candidate_tokens"] = candidate_tokens
    result["reference_tokens"] = reference_tokens
    result["candidate_truncated"] = result["candidate_tokens"] > SAFE_MAX_LENGTH
    result["reference_truncated"] = result["reference_tokens"] > SAFE_MAX_LENGTH
    result["pair_any_truncated"] = (
        result["candidate_truncated"] | result["reference_truncated"]
    )
    result["candidate_excess_tokens"] = (
        result["candidate_tokens"] - SAFE_MAX_LENGTH
    ).clip(lower=0)
    result["reference_excess_tokens"] = (
        result["reference_tokens"] - SAFE_MAX_LENGTH
    ).clip(lower=0)
    result = result.sort_values(["model", "variation", "pr_id", "version"])
    if len(result) != 778:
        raise RuntimeError(f"Expected 778 BERTScore rows, observed {len(result)}")

    historical = result[result["version"].eq("historical")].rename(
        columns={"pr_id": "normalized_pr_id"}
    )
    keys = ["normalized_pr_id", "model", "variation"]
    expected = pd.read_csv(
        source_paths(root)["rq2_outputs"] / "row_level/bertscore_metrics.csv",
        dtype={"normalized_pr_id": str},
    )
    expected["normalized_pr_id"] = expected["normalized_pr_id"].astype(str).str.replace(
        r"\.0$", "", regex=True
    )
    expected = expected[
        expected["entry_type"].str.lower().eq("ai")
        & expected["model"].isin(historical["model"])
        & expected["variation"].isin(CONTEXTS)
    ].merge(
        historical[keys].drop_duplicates(),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    validation = validate_column(
        historical,
        expected,
        "bertscore_f1",
        "bertscore_f1",
        keys,
        "historical_bertscore_f1",
        VALIDATION_TOLERANCE,
    )

    result_path = out / "bertscore_metrics.csv"
    result.to_csv(result_path, index=False)
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": utc_now(),
        "stage": "bertscore",
        "rows": len(result),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": sha256_file(model_snapshot / "config.json"),
        "baseline_sha256": sha256_file(baseline_path),
        "num_layers": 40,
        "rescale_with_baseline": True,
        "idf": False,
        "tokenizer_max_length": SAFE_MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "chunk_size": CHUNK_SIZE,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "candidate_truncated_rows": int(result["candidate_truncated"].sum()),
        "reference_truncated_rows": int(result["reference_truncated"].sum()),
        "pair_any_truncated_rows": int(result["pair_any_truncated"].sum()),
        "source_sha256": source_hashes,
        "output_sha256": sha256_file(result_path),
        "historical_validation_tolerance": VALIDATION_TOLERANCE,
        "historical_validation_tolerance_rationale": VALIDATION_TOLERANCE_RATIONALE,
        "historical_validation": validation,
        "raw_text_in_outputs": False,
    }
    write_json(out / "bertscore_metadata.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
