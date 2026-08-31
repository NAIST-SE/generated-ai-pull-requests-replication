"""Build the immutable, provenance-rich corpus for the corrected detector rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from .spec import (
    DEFAULT_OUTPUT_DIR,
    EXPECTED_AI_COUNTS,
    EXPECTED_AI_ROWS,
    EXPECTED_HUMAN_ROWS,
    EXPECTED_TOTAL_ROWS,
    MODELS,
    RUN_ID,
    RUN_SEED,
    SOURCE_FILES,
    normalise_pr_id,
    normalise_variation,
    sha256_file,
    sha256_text,
    source_path,
    stable_json_sha256,
    verify_frozen_sources,
)


GENERATION_REQUIRED_COLUMNS = {
    "experiment_id",
    "model",
    "pr_id",
    "variation",
    "prompt_hash",
    "full_prompt",
    "generated_text",
    "finish_reason",
    "provider",
    "precision",
}


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def _load_generation(path: Path, source_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, GENERATION_REQUIRED_COLUMNS, source_name)
    frame = frame.copy()
    frame["pr_id"] = frame["pr_id"].map(normalise_pr_id)
    frame["variation"] = frame["variation"].map(normalise_variation)
    frame["source_name"] = source_name
    frame["source_path"] = SOURCE_FILES[source_name]["path"]
    frame["source_file_sha256"] = SOURCE_FILES[source_name]["sha256"]
    frame["source_row_number"] = range(2, len(frame) + 2)
    return frame


def select_generation_rows(repo_root: Path) -> pd.DataFrame:
    old_qwen = _load_generation(source_path(repo_root, "old_qwen"), "old_qwen")
    old_gpt = _load_generation(source_path(repo_root, "old_gpt"), "old_gpt")
    corrected_llama = _load_generation(
        source_path(repo_root, "corrected_llama"), "corrected_llama"
    )
    corrected_qwen = _load_generation(
        source_path(repo_root, "corrected_qwen"), "corrected_qwen"
    )
    corrected_gpt = _load_generation(
        source_path(repo_root, "corrected_gpt"), "corrected_gpt"
    )

    reused = pd.concat([old_qwen, old_gpt], ignore_index=True)
    reused = reused[
        reused["variation"].isin(["P-1", "P-2", "P-3", "P-4"])
        & reused["finish_reason"].eq("stop")
    ].copy()
    reused["source_kind"] = "unchanged_run_3_control"

    corrected = pd.concat(
        [corrected_llama, corrected_qwen, corrected_gpt], ignore_index=True
    )
    corrected = corrected[corrected["finish_reason"].eq("stop")].copy()
    corrected["source_kind"] = "corrected_b4_v3_2"

    selected = pd.concat([reused, corrected], ignore_index=True)
    selected["generated_text"] = selected["generated_text"].fillna("").astype(str)
    selected["full_prompt"] = selected["full_prompt"].fillna("").astype(str)
    if selected["generated_text"].str.strip().eq("").any():
        raise ValueError("Selected generation rows contain empty generated_text values")
    if selected["generated_text"].str.contains("ERROR_GENERATION_FAILED", regex=False).any():
        raise ValueError("Selected generation rows contain serialized generation errors")
    if not selected["finish_reason"].eq("stop").all():
        raise ValueError("Selected generation rows include a non-stop finish reason")
    observed_prompt_md5 = selected["full_prompt"].map(
        lambda value: hashlib.md5(value.encode("utf-8")).hexdigest()
    )
    if observed_prompt_md5.ne(selected["prompt_hash"].astype(str)).any():
        raise ValueError("Selected generation rows contain a prompt MD5 mismatch")

    duplicate = selected.duplicated(["model", "pr_id", "variation"], keep=False)
    if duplicate.any():
        keys = selected.loc[duplicate, ["model", "pr_id", "variation"]].to_dict("records")
        raise ValueError(f"Duplicate generated model/PR/context keys: {keys[:10]}")

    actual_counts = selected.groupby(["model", "variation"]).size().to_dict()
    if actual_counts != EXPECTED_AI_COUNTS:
        raise ValueError(
            f"Generated-cell counts differ from the frozen specification: {actual_counts}"
        )
    if len(selected) != EXPECTED_AI_ROWS:
        raise ValueError(f"Expected {EXPECTED_AI_ROWS} AI rows, got {len(selected)}")

    return selected


def _entry_key(entry_type: str, pr_id: str, variation: str, model: str) -> str:
    if entry_type == "human":
        return f"{pr_id}_human"
    safe_model = model.replace(" ", "_").replace("/", "_")
    return f"{pr_id}_{variation}_{safe_model}"


def build_corpus_frame(repo_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    verified_sources = verify_frozen_sources(repo_root)
    completion = json.loads(source_path(repo_root, "corrected_completion").read_text())
    if completion.get("status") != "complete_and_verified":
        raise ValueError("Corrected generation completion manifest is not complete_and_verified")
    if completion.get("protocol_version") != "b4-corrected-generation-v3.2":
        raise ValueError("Unexpected corrected generation protocol version")
    if completion.get("total_rows") != 2870:
        raise ValueError("Unexpected corrected generation row count")

    prs = pd.read_csv(source_path(repo_root, "selected_prs"))
    _require_columns(prs, {"id", "title", "body"}, "selected_prs")
    prs = prs.copy()
    prs["pr_id"] = prs["id"].map(normalise_pr_id)
    if prs["pr_id"].duplicated().any():
        raise ValueError("Sampled PR source contains duplicate IDs")
    if len(prs) != EXPECTED_HUMAN_ROWS:
        raise ValueError(f"Expected {EXPECTED_HUMAN_ROWS} sampled PRs, got {len(prs)}")
    for column in ("title", "body"):
        if prs[column].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"Sampled PR source contains an empty {column}")

    titles = dict(zip(prs["pr_id"], prs["title"].astype(str)))
    generated = select_generation_rows(repo_root)
    missing_human = sorted(set(generated["pr_id"]) - set(titles))
    if missing_human:
        raise ValueError(f"Generated rows have no sampled PR record: {missing_human}")

    records: List[Dict[str, Any]] = []
    selected_pr_sha = SOURCE_FILES["selected_prs"]["sha256"]
    for source_index, row in prs.reset_index(drop=True).iterrows():
        pr_id = row["pr_id"]
        text = str(row["body"])
        question = str(row["title"])
        records.append(
            {
                "pr_id": pr_id,
                "variation": "original",
                "model": "human",
                "entry_type": "human",
                "type": "human",
                "entry_key": _entry_key("human", pr_id, "original", "human"),
                "text": text,
                "text_sha256": sha256_text(text),
                "qa_question": question,
                "qa_question_sha256": sha256_text(question),
                "source_kind": "human_baseline",
                "source_name": "selected_prs",
                "source_path": SOURCE_FILES["selected_prs"]["path"],
                "source_file_sha256": selected_pr_sha,
                "source_row_number": source_index + 2,
                "source_experiment_id": "",
                "source_prompt_hash": "",
                "source_prompt_hash_algorithm": "",
                "source_full_prompt_sha256": "",
                "generation_finish_reason": "human",
                "generation_provider": "human",
                "generation_precision": "",
            }
        )

    for _, row in generated.iterrows():
        pr_id = row["pr_id"]
        text = row["generated_text"]
        question = titles[pr_id]
        records.append(
            {
                "pr_id": pr_id,
                "variation": row["variation"],
                "model": row["model"],
                "entry_type": "ai",
                "type": "ai",
                "entry_key": _entry_key("ai", pr_id, row["variation"], row["model"]),
                "text": text,
                "text_sha256": sha256_text(text),
                "qa_question": question,
                "qa_question_sha256": sha256_text(question),
                "source_kind": row["source_kind"],
                "source_name": row["source_name"],
                "source_path": row["source_path"],
                "source_file_sha256": row["source_file_sha256"],
                "source_row_number": int(row["source_row_number"]),
                "source_experiment_id": row["experiment_id"],
                "source_prompt_hash": row["prompt_hash"],
                "source_prompt_hash_algorithm": "md5 (legacy generation field)",
                "source_full_prompt_sha256": sha256_text(row["full_prompt"]),
                "generation_finish_reason": row["finish_reason"],
                "generation_provider": row["provider"],
                "generation_precision": row["precision"],
            }
        )

    corpus = pd.DataFrame(records)
    if len(corpus) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_ROWS} total rows, got {len(corpus)}")
    if corpus["entry_key"].duplicated().any():
        raise ValueError("Corpus entry_key values are not unique")
    if corpus["text_sha256"].ne(corpus["text"].map(sha256_text)).any():
        raise ValueError("Corpus text hash invariant failed")
    if corpus["qa_question_sha256"].ne(corpus["qa_question"].map(sha256_text)).any():
        raise ValueError("Corpus QA-question hash invariant failed")

    corpus = corpus.sort_values(
        ["entry_type", "model", "pr_id", "variation", "entry_key"], kind="stable"
    ).reset_index(drop=True)
    scoring_order = list(range(1, len(corpus) + 1))
    random.Random(RUN_SEED).shuffle(scoring_order)
    corpus["scoring_order"] = scoring_order
    corpus.insert(0, "corpus_row", range(1, len(corpus) + 1))

    policy = {
        "run_id": RUN_ID,
        "human": "all 308 non-empty sampled PR bodies",
        "unchanged_controls": "Qwen and GPT P-1 through P-4 with finish_reason=stop from run_3",
        "corrected": "all verified rows from b4-corrected-v3: Llama P-1 through P-6; Qwen/GPT P-5 and P-6",
        "generation_failure_policy": "exclude non-stop generation rows; enforce frozen cell counts",
        "detector_text_policy": "score every non-empty complete description, including texts under 250 characters",
        "qa_pair_policy": "PR title as question and PR description as answer for every human and AI entry",
        "scoring_order": f"deterministic shuffle, seed={RUN_SEED}",
    }
    metadata = {
        "verified_sources": verified_sources,
        "policy": policy,
        "policy_sha256": stable_json_sha256(policy),
        "counts": {
            "total": len(corpus),
            "human": int(corpus["entry_type"].eq("human").sum()),
            "ai": int(corpus["entry_type"].eq("ai").sum()),
            "under_250_chars": int(corpus["text"].astype(str).str.len().lt(250).sum()),
            "over_15000_chars": int(corpus["text"].astype(str).str.len().gt(15000).sum()),
        },
    }
    return corpus, metadata


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        frame.to_csv(temp_path, index=False)
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_json(value: Dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        temp_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def write_corpus(repo_root: Path, output_dir: Path) -> Dict[str, Any]:
    corpus_path = output_dir / "corpus.csv"
    manifest_path = output_dir / "corpus_manifest.json"
    if corpus_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing versioned corpus in {output_dir}"
        )
    corpus, metadata = build_corpus_frame(repo_root)
    _atomic_write_csv(corpus, corpus_path)
    manifest = {
        "run_id": RUN_ID,
        "status": "corpus_complete_and_verified",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_git_commit": _git_commit(repo_root),
        "corpus_path": str(corpus_path.relative_to(repo_root)),
        "corpus_sha256": sha256_file(corpus_path),
        **metadata,
    }
    _atomic_write_json(manifest, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    manifest = write_corpus(repo_root, output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
