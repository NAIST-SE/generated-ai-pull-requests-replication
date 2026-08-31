#!/usr/bin/env python3
"""Server-only diagnostic for Qwen/GPT mixed-generation-session sensitivity."""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import json
import math
import os
import random
import re
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "generation-session-diagnostic-v1"
SELECTION_SEED = 20260831
GENERATION_SEED = 42
SAMPLE_SIZE = 50
VARIATIONS = ("P-1", "P-2", "P-3", "P-4")
QWEN_MODEL = "Qwen/Qwen2.5-72B-Instruct"
GPT_MODEL = "gpt-5-mini-2025-08-07"
FAILURE_MARKERS = ("ERROR_GENERATION_FAILED", "MOCKED DESCRIPTION")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id_hash(ids: Iterable[str]) -> str:
    return sha256_text("\n".join(sorted(str(value) for value in ids)) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_server(allow_non_server: bool) -> None:
    hostname = socket.gethostname().split(".")[0].lower()
    if hostname != "nokogiri" and not allow_non_server:
        raise SystemExit(
            f"Refusing to run on {hostname!r}; this diagnostic is server-only (expected nokogiri)."
        )


def successful_mask(frame: pd.DataFrame) -> pd.Series:
    generated = frame["generated_text"].fillna("").astype(str).str.strip()
    finish = frame["finish_reason"].fillna("").astype(str).str.strip().str.lower()
    failed = pd.Series(False, index=frame.index)
    upper = generated.str.upper()
    for marker in FAILURE_MARKERS:
        failed |= upper.str.contains(marker, regex=False)
    return generated.ne("") & ~failed & finish.eq("stop")


def read_generation(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "pr_id",
        "variation",
        "model",
        "prompt_hash",
        "full_prompt",
        "generated_text",
        "finish_reason",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{path}: missing columns {missing}")
    return frame


def old_complete_pairs(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    selected = frame.loc[
        frame["model"].eq(model) & frame["variation"].isin(VARIATIONS)
    ].copy()
    selected = selected.loc[successful_mask(selected)].copy()
    duplicates = selected.duplicated(["pr_id", "variation"], keep=False)
    if duplicates.any():
        count = int(duplicates.sum())
        raise RuntimeError(f"Historical {model} file has {count} duplicated successful rows")
    for row in selected.itertuples(index=False):
        observed = hashlib.md5(row.full_prompt.encode("utf-8")).hexdigest()
        if observed != row.prompt_hash:
            raise RuntimeError(
                f"Historical prompt hash mismatch for {model} {row.pr_id} {row.variation}"
            )
    return selected


def complete_pr_ids(frame: pd.DataFrame) -> set[str]:
    counts = frame.groupby("pr_id")["variation"].nunique()
    return set(counts[counts.eq(len(VARIATIONS))].index.astype(str))


def default_paths(project_root: Path) -> dict[str, Path]:
    base = project_root / "phase3" / "datasets" / "temperature_1" / "run_3"
    return {
        "issue_manifest": project_root
        / "phase3"
        / "analysis"
        / "outputs"
        / "tier_b"
        / "b4-issue-source-audit-v1"
        / "regeneration_sources.jsonl",
        "old_qwen": base / "Qwen_Qwen2.5-72B-Instruct_1771584467.csv",
        "old_gpt": base / "chatgpt_results_exp_chatgpt_1771584462.csv",
        "selected_prs": project_root / "phase3" / "datasets" / "selected_prs_sampled.csv",
        "pr_files": project_root
        / "phase3"
        / "datasets"
        / "selected_prs_with_diffs_and_file_contents.csv",
    }


def git_commit(project_root: Path) -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_issue_ids(project_root: Path, path: Path) -> set[str]:
    from phase3.generation.issue_sources import load_issue_source_manifest

    return set(load_issue_source_manifest(path))


def build_sample_manifest(project_root: Path, output_dir: Path) -> dict[str, Any]:
    paths = default_paths(project_root)
    issue_ids = load_issue_ids(project_root, paths["issue_manifest"])
    qwen = old_complete_pairs(read_generation(paths["old_qwen"]), QWEN_MODEL)
    gpt = old_complete_pairs(read_generation(paths["old_gpt"]), GPT_MODEL)
    population = sorted(issue_ids & complete_pr_ids(qwen) & complete_pr_ids(gpt))
    if len(population) < SAMPLE_SIZE:
        raise RuntimeError(
            f"Only {len(population)} complete eligible PRs; cannot sample {SAMPLE_SIZE}"
        )
    selected = sorted(random.Random(SELECTION_SEED).sample(population, SAMPLE_SIZE))
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": utc_now(),
        "git_commit": git_commit(project_root),
        "selection_seed": SELECTION_SEED,
        "sample_size": SAMPLE_SIZE,
        "variations": list(VARIATIONS),
        "models": [QWEN_MODEL, GPT_MODEL],
        "population_definition": (
            "Hash-pinned genuine-issue PRs with one successful historical Qwen and GPT "
            "description in every P1-P4 context"
        ),
        "audited_issue_pr_count": len(issue_ids),
        "complete_case_population_size": len(population),
        "complete_case_population_pr_ids_sha256": stable_id_hash(population),
        "selected_pr_ids": selected,
        "selected_pr_ids_sha256": stable_id_hash(selected),
        "sources": {
            label: {
                "path": str(path.relative_to(project_root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for label, path in paths.items()
        },
    }
    destination = output_dir / "sample_manifest.json"
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        immutable = (
            "protocol_version",
            "git_commit",
            "selection_seed",
            "sample_size",
            "complete_case_population_size",
            "complete_case_population_pr_ids_sha256",
            "selected_pr_ids",
            "selected_pr_ids_sha256",
            "sources",
        )
        differences = [key for key in immutable if previous.get(key) != payload.get(key)]
        if differences:
            raise RuntimeError(
                f"Existing sample manifest differs in immutable fields: {differences}"
            )
        return previous
    write_json(destination, payload)
    return payload


def load_sample(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "sample_manifest.json"
    if not path.is_file():
        raise RuntimeError("Run the select command first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Sample manifest protocol mismatch")
    selected = payload.get("selected_pr_ids")
    if not isinstance(selected, list) or len(selected) != SAMPLE_SIZE:
        raise RuntimeError("Sample manifest has the wrong number of PR IDs")
    if len(set(selected)) != SAMPLE_SIZE or stable_id_hash(selected) != payload.get(
        "selected_pr_ids_sha256"
    ):
        raise RuntimeError("Sample manifest ID set is duplicated or hash-invalid")
    return payload


def load_selected_data(project_root: Path, selected_ids: Sequence[str]) -> dict[str, Any]:
    from phase3.data_loader import DataLoader

    all_prs = DataLoader().get_comprehensive_data(require_issue_manifest=True)
    missing = sorted(set(selected_ids).difference(all_prs))
    if missing:
        raise RuntimeError(f"Selected PR IDs absent from current data loader: {missing}")
    return {pr_id: all_prs[pr_id] for pr_id in selected_ids}


def current_prompt(model_label: str, pr_data: Mapping[str, Any], variation: str) -> str:
    if model_label == "qwen":
        from phase3.generation.generate_openai import build_messages_for
        from phase3.generation.generate_qwen import PromptBuilder

        messages, truncated = build_messages_for(
            PromptBuilder,
            dict(pr_data),
            variation,
            truncate_p4=False,
            tokenizer=None,
            max_tokens=30_000,
        )
        if truncated is not None:
            raise RuntimeError("Unexpected Qwen prompt truncation")
        return "\n\n".join(message["content"] for message in messages)
    if model_label == "gpt":
        from phase3.generation.generate_chatgpt import PromptBuilder, VARIATIONS as GPT_VARIATIONS

        return PromptBuilder.build_prompt(dict(pr_data), GPT_VARIATIONS[variation])
    raise ValueError(model_label)


def historical_index(project_root: Path, model_label: str) -> pd.DataFrame:
    paths = default_paths(project_root)
    if model_label == "qwen":
        return old_complete_pairs(read_generation(paths["old_qwen"]), QWEN_MODEL).set_index(
            ["pr_id", "variation"], verify_integrity=True
        )
    return old_complete_pairs(read_generation(paths["old_gpt"]), GPT_MODEL).set_index(
        ["pr_id", "variation"], verify_integrity=True
    )


def preflight_prompts(project_root: Path, output_dir: Path) -> dict[str, Any]:
    sample = load_sample(output_dir)
    selected_ids = sample["selected_pr_ids"]
    all_prs = load_selected_data(project_root, selected_ids)
    rows: list[dict[str, Any]] = []
    mismatch_count = 0
    for label in ("qwen", "gpt"):
        old = historical_index(project_root, label)
        for pr_id in selected_ids:
            for variation in VARIATIONS:
                prompt = current_prompt(label, all_prs[pr_id], variation)
                historical = old.loc[(pr_id, variation)]
                old_prompt = str(historical["full_prompt"])
                exact = prompt == old_prompt
                mismatch_count += int(not exact)
                rows.append(
                    {
                        "model_label": label,
                        "pr_id": pr_id,
                        "variation": variation,
                        "exact_prompt_match": exact,
                        "historical_prompt_sha256": sha256_text(old_prompt),
                        "current_prompt_sha256": sha256_text(prompt),
                        "historical_prompt_md5": str(historical["prompt_hash"]),
                        "current_prompt_md5": hashlib.md5(prompt.encode("utf-8")).hexdigest(),
                    }
                )
    row_fingerprint = sha256_text(
        "\n".join(
            f"{row['model_label']}|{row['pr_id']}|{row['variation']}|"
            f"{row['current_prompt_sha256']}"
            for row in rows
        )
        + "\n"
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": utc_now(),
        "git_commit": git_commit(project_root),
        "sample_manifest_sha256": sha256_file(output_dir / "sample_manifest.json"),
        "expected_pairs": SAMPLE_SIZE * len(VARIATIONS) * 2,
        "checked_pairs": len(rows),
        "prompt_mismatch_count": mismatch_count,
        "all_prompts_exact": mismatch_count == 0,
        "prompt_set_sha256": row_fingerprint,
        "by_model_context": [],
    }
    check_frame = pd.DataFrame(rows)
    for (label, variation), group in check_frame.groupby(["model_label", "variation"]):
        payload["by_model_context"].append(
            {
                "model_label": label,
                "variation": variation,
                "pairs": len(group),
                "exact_matches": int(group["exact_prompt_match"].sum()),
            }
        )
    write_json(output_dir / "prompt_preflight.json", payload)
    if mismatch_count:
        mismatch_path = output_dir / "prompt_mismatches.csv"
        check_frame.loc[~check_frame["exact_prompt_match"]].to_csv(mismatch_path, index=False)
        raise RuntimeError(
            f"Prompt preflight failed: {mismatch_count} historical/current prompts differ"
        )
    return payload


def require_preflight(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "prompt_preflight.json"
    if not path.is_file():
        raise RuntimeError("Run preflight before any paid request")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("all_prompts_exact") or payload.get("checked_pairs") != 400:
        raise RuntimeError("Prompt preflight is absent, incomplete, or failed")
    if payload.get("sample_manifest_sha256") != sha256_file(
        output_dir / "sample_manifest.json"
    ):
        raise RuntimeError("Sample manifest changed after prompt preflight")
    return payload


def successful_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    frame = read_generation(path)
    return set(
        frame.loc[successful_mask(frame), ["pr_id", "variation"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )


def canonicalize_output(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        return
    frame = read_generation(path)
    frame["_success"] = successful_mask(frame).astype(int)
    frame["_order"] = np.arange(len(frame))
    frame = (
        frame.sort_values(["pr_id", "variation", "_success", "_order"])
        .drop_duplicates(["pr_id", "variation"], keep="last")
        .drop(columns=["_success", "_order"])
        .sort_values(["pr_id", "variation"])
    )
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def qwen_args() -> SimpleNamespace:
    return SimpleNamespace(
        truncate_p4=False,
        max_context_tokens=30_000,
        truncated_max_tokens=None,
        dry_run=False,
        provider="novita",
        quantization=["bf16"],
        model=QWEN_MODEL,
        temperature=1.0,
        max_tokens=None,
        seed=GENERATION_SEED,
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    )


def gpt_args() -> SimpleNamespace:
    return SimpleNamespace(
        truncate_p4=False,
        max_context_tokens=30_000,
        dry_run=False,
        mock=False,
        model=GPT_MODEL,
        temperature=None,
        seed=GENERATION_SEED,
    )


def run_qwen(tasks: list[tuple[str, str]], all_prs: dict[str, Any], out_path: Path) -> None:
    from phase3.generation.generate_openai import create_client, process_batch
    from phase3.generation.generate_qwen import PromptBuilder

    args = qwen_args()
    if not args.api_key:
        raise RuntimeError("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is required")
    client = create_client(args.base_url, api_key=args.api_key)
    experiment_id = f"{PROTOCOL_VERSION}-qwen-{int(time.time())}"
    for start in range(0, len(tasks), 5):
        batch = tasks[start : start + 5]
        process_batch(
            batch,
            all_prs,
            client,
            PromptBuilder,
            args,
            experiment_id,
            str(out_path),
            "huggingface",
            None,
        )
        print(f"qwen progress: {min(start + 5, len(tasks))}/{len(tasks)} attempted", flush=True)


def run_gpt(tasks: list[tuple[str, str]], all_prs: dict[str, Any], out_path: Path) -> None:
    from phase3.generation.generate_chatgpt import Generator, process_batch

    args = gpt_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")
    generator = Generator(
        model_name=GPT_MODEL,
        api_key=os.environ["OPENAI_API_KEY"],
        mock=False,
        temperature=None,
        seed=GENERATION_SEED,
    )
    if generator.client is None:
        raise RuntimeError("OpenAI client could not be initialized")
    experiment_id = f"{PROTOCOL_VERSION}-gpt-{int(time.time())}"
    for start in range(0, len(tasks), 5):
        batch = [
            (pr_id, __import__(
                "phase3.generation.generate_chatgpt", fromlist=["VARIATIONS"]
            ).VARIATIONS[variation])
            for pr_id, variation in tasks[start : start + 5]
        ]
        process_batch(batch, all_prs, generator, args, experiment_id, str(out_path))
        print(f"gpt progress: {min(start + 5, len(tasks))}/{len(tasks)} attempted", flush=True)


def expected_tasks(selected_ids: Sequence[str]) -> list[tuple[str, str]]:
    return [(pr_id, variation) for pr_id in selected_ids for variation in VARIATIONS]


def run_generation(
    project_root: Path,
    output_dir: Path,
    *,
    smoke: bool,
    models: Sequence[str],
    passes: int,
) -> dict[str, Any]:
    require_preflight(output_dir)
    sample = load_sample(output_dir)
    selected_ids = sample["selected_pr_ids"]
    all_prs = load_selected_data(project_root, selected_ids)
    if smoke:
        tasks = [(selected_ids[0], "P-2")]
        base_dir = output_dir / "smoke"
    else:
        tasks = expected_tasks(selected_ids)
        base_dir = output_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    outcomes: dict[str, Any] = {}
    for label in models:
        out_path = base_dir / f"{label}_current.csv"
        for pass_number in range(1, passes + 1):
            canonicalize_output(out_path)
            done = successful_pairs(out_path)
            pending = [task for task in tasks if task not in done]
            print(
                f"{label}: pass {pass_number}/{passes}; successful={len(done)}; pending={len(pending)}",
                flush=True,
            )
            if not pending:
                break
            if label == "qwen":
                run_qwen(pending, all_prs, out_path)
            elif label == "gpt":
                run_gpt(pending, all_prs, out_path)
            else:
                raise ValueError(label)
            canonicalize_output(out_path)
        final_done = successful_pairs(out_path)
        outcomes[label] = {
            "path": str(out_path.relative_to(project_root)),
            "expected": len(tasks),
            "successful": len(set(tasks) & final_done),
            "complete": set(tasks).issubset(final_done),
        }
        if smoke and not outcomes[label]["complete"]:
            raise RuntimeError(f"{label} smoke test failed; inspect {out_path}")
    return outcomes


def parse_decoding(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid decoding_params JSON: {value!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("decoding_params is not an object")
    return parsed


def validate_model_output(
    project_root: Path,
    output_dir: Path,
    label: str,
    expected_ids: set[str],
) -> dict[str, Any]:
    path = output_dir / f"{label}_current.csv"
    canonicalize_output(path)
    frame = read_generation(path)
    expected = {(pr_id, variation) for pr_id in expected_ids for variation in VARIATIONS}
    observed = set(frame[["pr_id", "variation"]].itertuples(index=False, name=None))
    if observed != expected:
        raise RuntimeError(
            f"{label}: expected {len(expected)} pairs, observed {len(observed)}; "
            f"missing={len(expected-observed)}, extra={len(observed-expected)}"
        )
    success = successful_mask(frame)
    if frame.duplicated(["pr_id", "variation"]).any():
        raise RuntimeError(f"{label}: duplicate pairs remain")
    old = historical_index(project_root, label)
    prompt_mismatches = 0
    for row in frame.itertuples(index=False):
        historical_prompt = str(old.loc[(row.pr_id, row.variation), "full_prompt"])
        prompt_mismatches += int(row.full_prompt != historical_prompt)
        if hashlib.md5(row.full_prompt.encode("utf-8")).hexdigest() != row.prompt_hash:
            raise RuntimeError(f"{label}: current prompt hash mismatch")
        decoding = parse_decoding(row.decoding_params)
        if label == "qwen":
            if decoding != {"temperature": 1.0, "seed": GENERATION_SEED}:
                raise RuntimeError(f"qwen: unexpected decoding parameters {decoding}")
        elif decoding != {"seed": GENERATION_SEED}:
            raise RuntimeError(f"gpt: unexpected decoding parameters {decoding}")
    if prompt_mismatches:
        raise RuntimeError(f"{label}: {prompt_mismatches} prompts differ from historical")
    expected_model = QWEN_MODEL if label == "qwen" else GPT_MODEL
    if set(frame["model"]) != {expected_model}:
        raise RuntimeError(f"{label}: model identity mismatch")
    if label == "qwen":
        if set(frame["provider"].str.lower()) != {"novita"}:
            raise RuntimeError("qwen: provider is not exactly Novita")
        if set(frame["precision"].str.lower()) != {"bf16"}:
            raise RuntimeError("qwen: precision is not exactly BF16")
    else:
        if set(frame["provider"].str.lower()) != {"openai"}:
            raise RuntimeError("gpt: provider is not exactly OpenAI")
        if set(frame["precision"].str.lower()) != {"auto"}:
            raise RuntimeError("gpt: precision is not provider-managed/auto")
    for numeric in ("prompt_token_count", "generated_token_count"):
        values = pd.to_numeric(frame[numeric], errors="coerce")
        if values.isna().any() or (values < 0).any():
            raise RuntimeError(f"{label}: {numeric} must be non-negative for every row")
        if (values.loc[success] <= 0).any():
            raise RuntimeError(f"{label}: successful rows require positive {numeric}")

    failure_records: list[dict[str, Any]] = []
    for row in frame.loc[~success].itertuples(index=False):
        historical = old.loc[(row.pr_id, row.variation)]
        text = str(row.generated_text).strip().upper()
        if "ERROR_GENERATION_FAILED" in text:
            category = "api_error_marker"
        elif not text:
            category = "empty_response"
        else:
            category = "non_stop_or_other"
        failure_records.append(
            {
                "model_label": label,
                "pr_id": row.pr_id,
                "variation": row.variation,
                "category": category,
                "finish_reason": row.finish_reason,
                "historical_prompt_token_count": int(
                    float(historical["prompt_token_count"])
                ),
                "prompt_sha256": sha256_text(row.full_prompt),
                "retry_passes_configured": 3,
                "persistent_after_all_passes": True,
            }
        )
    return {
        "rows": len(frame),
        "successful_rows": int(success.sum()),
        "failed_rows": int((~success).sum()),
        "pr_count": frame["pr_id"].nunique(),
        "variations": sorted(frame["variation"].unique()),
        "file_sha256": sha256_file(path),
        "system_fingerprints": sorted(
            value for value in frame.get("system_fingerprint", pd.Series(dtype=str)).unique() if value
        ),
        "failure_records": failure_records,
    }


def verify_outputs(project_root: Path, output_dir: Path) -> dict[str, Any]:
    require_preflight(output_dir)
    sample = load_sample(output_dir)
    ids = set(sample["selected_pr_ids"])
    models = {
        label: validate_model_output(project_root, output_dir, label, ids)
        for label in ("qwen", "gpt")
    }
    failure_records = [
        record
        for model_result in models.values()
        for record in model_result["failure_records"]
    ]
    failed_rows = len(failure_records)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": (
            "complete_and_verified"
            if failed_rows == 0
            else "partial_generation_with_verified_failures"
        ),
        "verified_at_utc": utc_now(),
        "git_commit": git_commit(project_root),
        "sample_manifest_sha256": sha256_file(output_dir / "sample_manifest.json"),
        "prompt_preflight_sha256": sha256_file(output_dir / "prompt_preflight.json"),
        "models": models,
        "total_rows": SAMPLE_SIZE * len(VARIATIONS) * 2,
        "successful_rows": SAMPLE_SIZE * len(VARIATIONS) * 2 - failed_rows,
        "failed_rows": failed_rows,
        "failure_interpretation": (
            "Failures remain part of the frozen sample; no PR replacement, prompt "
            "truncation, or provider substitution was performed."
        ),
    }
    if failure_records:
        failure_path = output_dir / "failure_audit.csv"
        pd.DataFrame(failure_records).sort_values(
            ["model_label", "variation", "historical_prompt_token_count", "pr_id"]
        ).to_csv(failure_path, index=False)
        payload["failure_audit_sha256"] = sha256_file(failure_path)
    write_json(output_dir / "verification.json", payload)
    return payload


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def normalized_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def word_tokens(value: str) -> list[str]:
    return TOKEN_RE.findall(normalized_text(value).lower())


def rouge_l_f1(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return float(left == right)
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(right)
    recall = lcs / len(left)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def comparison_rows(project_root: Path, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in ("qwen", "gpt"):
        old = historical_index(project_root, label)
        current_frame = read_generation(output_dir / f"{label}_current.csv")
        current_frame = current_frame.loc[successful_mask(current_frame)].copy()
        current = current_frame.set_index(
            ["pr_id", "variation"], verify_integrity=True
        )
        for key, new_row in current.iterrows():
            old_row = old.loc[key]
            old_text = normalized_text(str(old_row["generated_text"]))
            new_text = normalized_text(str(new_row["generated_text"]))
            old_words, new_words = word_tokens(old_text), word_tokens(new_text)
            rows.append(
                {
                    "model_label": label,
                    "model": str(new_row["model"]),
                    "pr_id": key[0],
                    "variation": key[1],
                    "old_text_sha256": sha256_text(old_text),
                    "new_text_sha256": sha256_text(new_text),
                    "normalized_exact_match": old_text == new_text,
                    "casefold_exact_match": old_text.casefold() == new_text.casefold(),
                    "sequence_similarity": difflib.SequenceMatcher(
                        None, old_text.casefold(), new_text.casefold(), autojunk=False
                    ).ratio(),
                    "rouge_l_f1": rouge_l_f1(old_words, new_words),
                    "word_set_jaccard": jaccard(old_words, new_words),
                    "old_word_count": len(old_words),
                    "new_word_count": len(new_words),
                    "word_count_difference": len(new_words) - len(old_words),
                    "word_count_ratio": (
                        len(new_words) / len(old_words) if old_words else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["model_label", "variation", "pr_id"])


def bootstrap_interval(
    values: Sequence[float], statistic: str, *, seed: int, draws: int = 10_000
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(array), size=(draws, len(array)))
    samples = array[indexes]
    estimates = samples.mean(axis=1) if statistic == "mean" else np.median(samples, axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def summarize_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = []
    for (label, variation), group in frame.groupby(["model_label", "variation"], sort=True):
        groups.append((label, variation, group))
    for label, group in frame.groupby("model_label", sort=True):
        groups.append((label, "ALL_P1_P4", group))
    groups.append(("ALL_MODELS", "ALL_P1_P4", frame))
    for index, (label, variation, group) in enumerate(groups):
        if label == "ALL_MODELS":
            expected_pairs = SAMPLE_SIZE * len(VARIATIONS) * 2
        elif variation == "ALL_P1_P4":
            expected_pairs = SAMPLE_SIZE * len(VARIATIONS)
        else:
            expected_pairs = SAMPLE_SIZE
        exact = group["normalized_exact_match"].astype(float).to_numpy()
        similarity = group["sequence_similarity"].to_numpy(float)
        rouge = group["rouge_l_f1"].to_numpy(float)
        ratio = group["word_count_ratio"].to_numpy(float)
        exact_ci = bootstrap_interval(exact, "mean", seed=SELECTION_SEED + index * 11)
        sim_ci = bootstrap_interval(similarity, "median", seed=SELECTION_SEED + index * 11 + 1)
        rouge_ci = bootstrap_interval(rouge, "median", seed=SELECTION_SEED + index * 11 + 2)
        ratio_ci = bootstrap_interval(ratio, "median", seed=SELECTION_SEED + index * 11 + 3)
        rows.append(
            {
                "model_label": label,
                "variation": variation,
                "n_pairs": len(group),
                "expected_pairs": expected_pairs,
                "missing_pairs": expected_pairs - len(group),
                "exact_match_rate": exact.mean(),
                "exact_match_rate_ci_low": exact_ci[0],
                "exact_match_rate_ci_high": exact_ci[1],
                "sequence_similarity_median": np.median(similarity),
                "sequence_similarity_median_ci_low": sim_ci[0],
                "sequence_similarity_median_ci_high": sim_ci[1],
                "rouge_l_f1_median": np.median(rouge),
                "rouge_l_f1_median_ci_low": rouge_ci[0],
                "rouge_l_f1_median_ci_high": rouge_ci[1],
                "word_count_ratio_median": np.nanmedian(ratio),
                "word_count_ratio_median_ci_low": ratio_ci[0],
                "word_count_ratio_median_ci_high": ratio_ci[1],
                "old_word_count_median": np.median(group["old_word_count"]),
                "new_word_count_median": np.median(group["new_word_count"]),
            }
        )
    return pd.DataFrame(rows)


def analyze_outputs(project_root: Path, output_dir: Path) -> dict[str, Any]:
    verification = output_dir / "verification.json"
    if not verification.is_file():
        raise RuntimeError("Run verify before analyze")
    verified = json.loads(verification.read_text(encoding="utf-8"))
    allowed_statuses = {
        "complete_and_verified",
        "partial_generation_with_verified_failures",
    }
    if verified.get("status") not in allowed_statuses:
        raise RuntimeError("Verification status is not analyzable")
    comparison = comparison_rows(project_root, output_dir)
    if len(comparison) != int(verified["successful_rows"]):
        raise RuntimeError(
            "Comparison-row count differs from the verified successful-row count: "
            f"{len(comparison)} != {verified['successful_rows']}"
        )
    summary = summarize_comparison(comparison)
    comparison_path = output_dir / "paired_comparison.csv"
    summary_path = output_dir / "summary.csv"
    comparison.to_csv(comparison_path, index=False)
    summary.to_csv(summary_path, index=False)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_type": "paired_descriptive_generation-protocol_sensitivity",
        "analyzed_at_utc": utc_now(),
        "git_commit": git_commit(project_root),
        "verification_sha256": sha256_file(verification),
        "paired_comparison_sha256": sha256_file(comparison_path),
        "summary_csv_sha256": sha256_file(summary_path),
        "pairs": len(comparison),
        "expected_pairs": 400,
        "missing_pairs": 400 - len(comparison),
        "verification_status": verified["status"],
        "raw_text_in_comparison_csv": False,
        "limitations": [
            "This diagnostic combines API date/backend changes with newly transmitted seeds.",
            "The sample can reveal material sensitivity but cannot establish equivalence.",
            "Similarity and length summaries do not directly reproduce RQ1-RQ3 outcomes.",
            "Qwen P4 failures are non-random with respect to prompt length; the observed "
            "Qwen P4 comparisons must not be generalized to the omitted long prompts.",
        ],
        "summary": summary.to_dict(orient="records"),
    }
    write_json(output_dir / "summary.json", payload)
    return payload


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("select", "preflight", "smoke", "run", "verify", "analyze")
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-non-server", action="store_true")
    parser.add_argument("--models", nargs="+", choices=("qwen", "gpt"), default=("qwen", "gpt"))
    parser.add_argument("--passes", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_server(args.allow_non_server)
    project_root = (args.project_root or project_root_from_script()).resolve()
    sys.path.insert(0, str(project_root))
    output_dir = (
        args.output_dir
        or project_root
        / "phase3"
        / "analysis"
        / "outputs"
        / "tier_b"
        / PROTOCOL_VERSION
    ).resolve()
    if args.passes < 1:
        raise SystemExit("--passes must be positive")
    if args.command == "select":
        result = build_sample_manifest(project_root, output_dir)
    elif args.command == "preflight":
        result = preflight_prompts(project_root, output_dir)
    elif args.command == "smoke":
        result = run_generation(
            project_root, output_dir, smoke=True, models=args.models, passes=args.passes
        )
    elif args.command == "run":
        result = run_generation(
            project_root, output_dir, smoke=False, models=args.models, passes=args.passes
        )
    elif args.command == "verify":
        result = verify_outputs(project_root, output_dir)
    else:
        result = analyze_outputs(project_root, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
