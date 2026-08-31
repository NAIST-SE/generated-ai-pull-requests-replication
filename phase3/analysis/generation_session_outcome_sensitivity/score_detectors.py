#!/usr/bin/env python3
"""Score current texts with the two pinned local RoBERTa detectors."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    PROTOCOL,
    load_inputs,
    normalize_pr_id,
    output_dir,
    project_root,
    require_nokogiri,
    sha256_file,
    sha256_text,
    source_paths,
    utc_now,
    write_json,
)


DETECTORS = ("roberta", "roberta-qa")
BATCH_SIZE = 32


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    require_nokogiri()
    root = project_root(Path(__file__))
    sys.path.insert(0, str(root))
    from phase3.detection.corrected_rerun.detectors import RobertaDetector
    from phase3.detection.corrected_rerun.spec import DETECTOR_CONFIGS
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Pinned local detector stage requires CUDA")
    out = output_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    pairs, _, source_hashes = load_inputs(root)
    corrected_path = source_paths(root)["corrected_detector_corpus"]
    corrected = pd.read_csv(corrected_path, dtype=str, keep_default_na=False)
    corrected["pr_id"] = corrected["pr_id"].map(normalize_pr_id)
    corrected_ai = corrected[corrected["entry_type"].str.lower().eq("ai")].set_index(
        ["model", "variation", "pr_id"], verify_integrity=True
    )
    corrected_human = corrected[
        corrected["entry_type"].str.lower().eq("human")
    ].set_index("pr_id", verify_integrity=True)

    current_entries: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        current_entries.append(
            {
                "model_label": pair.model_label,
                "model": pair.model,
                "pr_id": pair.pr_id,
                "variation": pair.variation,
                "context": pair.context,
                "text": pair.current_text,
                "text_sha256": sha256_text(pair.current_text),
                "qa_question": pair.title,
                "qa_question_sha256": sha256_text(pair.title),
            }
        )

    output_rows: list[dict[str, Any]] = []
    for detector_name in DETECTORS:
        config = DETECTOR_CONFIGS[detector_name]
        detector = RobertaDetector(
            detector_name, device="cuda", local_files_only=True
        )
        for start in range(0, len(current_entries), BATCH_SIZE):
            batch = current_entries[start : start + BATCH_SIZE]
            results = detector.score_batch(batch)
            for entry, result in zip(batch, results):
                output_rows.append(
                    {
                        "detector": detector_name,
                        "model_label": entry["model_label"],
                        "model": entry["model"],
                        "pr_id": entry["pr_id"],
                        "variation": entry["variation"],
                        "context": entry["context"],
                        "version": "current",
                        "ai_probability": result["ai_probability"],
                        "is_ai": result["is_ai_generated"],
                        "input_token_count": result["input_token_count"],
                        "used_token_count": result["used_token_count"],
                        "input_truncated": result["input_truncated"],
                        "question_token_count": result["question_token_count"],
                        "answer_token_count": result["answer_token_count"],
                        "raw_response_sha256": result["raw_response_sha256"],
                        "detector_model": result["detector_model"],
                        "detector_revision": result["detector_revision"],
                        "input_mode": result["input_mode"],
                    }
                )
            print(
                f"{detector_name} progress: {min(start + BATCH_SIZE, len(current_entries))}/"
                f"{len(current_entries)}",
                flush=True,
            )
        del detector
        gc.collect()
        torch.cuda.empty_cache()

        for pair in pairs.itertuples(index=False):
            source = corrected_ai.loc[(pair.model, pair.variation, pair.pr_id)]
            if not as_bool(source[f"{detector_name}_success"]):
                raise RuntimeError(
                    f"Historical {detector_name} score is unsuccessful for "
                    f"{pair.model} {pair.variation} {pair.pr_id}"
                )
            if str(source[f"{detector_name}_revision"]) != str(config["revision"]):
                raise RuntimeError(f"Historical {detector_name} revision differs")
            output_rows.append(
                {
                    "detector": detector_name,
                    "model_label": pair.model_label,
                    "model": pair.model,
                    "pr_id": pair.pr_id,
                    "variation": pair.variation,
                    "context": pair.context,
                    "version": "historical",
                    "ai_probability": float(source[f"{detector_name}_ai_probability"]),
                    "is_ai": as_bool(source[f"{detector_name}_is_ai"]),
                    "input_token_count": int(source[f"{detector_name}_input_token_count"]),
                    "used_token_count": int(source[f"{detector_name}_used_token_count"]),
                    "input_truncated": as_bool(source[f"{detector_name}_input_truncated"]),
                    "question_token_count": "",
                    "answer_token_count": "",
                    "raw_response_sha256": sha256_text(
                        str(source[f"{detector_name}_response"])
                    ),
                    "detector_model": str(source[f"{detector_name}_model"]),
                    "detector_revision": str(source[f"{detector_name}_revision"]),
                    "input_mode": str(source[f"{detector_name}_input_mode"]),
                }
            )

    scores = pd.DataFrame(output_rows).sort_values(
        ["detector", "model", "variation", "pr_id", "version"]
    )
    if len(scores) != 1556:
        raise RuntimeError(f"Expected 1,556 versioned detector rows, observed {len(scores)}")
    if scores.duplicated(
        ["detector", "model", "variation", "pr_id", "version"]
    ).any():
        raise RuntimeError("Duplicate detector score keys")

    sampled_ids = sorted(set(pairs["pr_id"]))
    human_rows: list[dict[str, Any]] = []
    for detector_name in DETECTORS:
        config = DETECTOR_CONFIGS[detector_name]
        for pr_id in sampled_ids:
            source = corrected_human.loc[pr_id]
            if not as_bool(source[f"{detector_name}_success"]):
                raise RuntimeError(f"Human {detector_name} score unsuccessful for {pr_id}")
            if str(source[f"{detector_name}_revision"]) != str(config["revision"]):
                raise RuntimeError(f"Human {detector_name} revision differs")
            human_rows.append(
                {
                    "detector": detector_name,
                    "pr_id": pr_id,
                    "ai_probability": float(source[f"{detector_name}_ai_probability"]),
                    "is_ai": as_bool(source[f"{detector_name}_is_ai"]),
                    "input_token_count": int(source[f"{detector_name}_input_token_count"]),
                    "used_token_count": int(source[f"{detector_name}_used_token_count"]),
                    "input_truncated": as_bool(source[f"{detector_name}_input_truncated"]),
                    "detector_model": str(source[f"{detector_name}_model"]),
                    "detector_revision": str(source[f"{detector_name}_revision"]),
                    "input_mode": str(source[f"{detector_name}_input_mode"]),
                }
            )
    human_scores = pd.DataFrame(human_rows).sort_values(["detector", "pr_id"])
    if len(human_scores) != 100:
        raise RuntimeError(f"Expected 100 human-detector rows, observed {len(human_scores)}")

    scores_path = out / "detector_scores.csv"
    human_path = out / "human_detector_scores.csv"
    scores.to_csv(scores_path, index=False)
    human_scores.to_csv(human_path, index=False)
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": utc_now(),
        "stage": "local_detectors",
        "detectors": {
            name: {
                "model": DETECTOR_CONFIGS[name]["model"],
                "revision": DETECTOR_CONFIGS[name]["revision"],
                "input_mode": DETECTOR_CONFIGS[name]["input_mode"],
                "max_length": DETECTOR_CONFIGS[name]["max_length"],
                "threshold": 0.5,
            }
            for name in DETECTORS
        },
        "batch_size": BATCH_SIZE,
        "gpu": torch.cuda.get_device_name(0),
        "versioned_ai_rows": len(scores),
        "human_rows": len(human_scores),
        "current_truncated_rows": int(
            scores[scores["version"].eq("current")]["input_truncated"].sum()
        ),
        "historical_truncated_rows": int(
            scores[scores["version"].eq("historical")]["input_truncated"].sum()
        ),
        "source_sha256": source_hashes,
        "scores_sha256": sha256_file(scores_path),
        "human_scores_sha256": sha256_file(human_path),
        "raw_text_in_outputs": False,
        "zerogpt_included": False,
    }
    write_json(out / "detector_metadata.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
