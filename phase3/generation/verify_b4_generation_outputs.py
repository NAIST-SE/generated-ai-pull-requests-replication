"""Fail-closed verification for the corrected B4 generation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS_DIR = PROJECT_ROOT / "phase3" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from deep_audit.server_guard import require_nokogiri
from phase3.generation.issue_sources import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PR_COUNT,
    load_issue_source_manifest,
    sha256_file,
    sha256_text,
)


PROTOCOL_VERSION = "b4-corrected-generation-v3.2"
EXPECTED_SAMPLED_PRS_SHA256 = (
    "d00daddbf1ed5d054bc3d126248673e9487f381f46c3ae64e68af80e1d139a8e"
)
EXPECTED_SAMPLED_PR_COUNT = 308
LLAMA_P4_FILE_TOKEN_BUDGET = 90_000
LLAMA_TRUNCATED_P4_MAX_TOKENS = 4_096
EXPECTED_FULL_RUN_LLAMA_P4_TRUNCATIONS = 7
ISSUE_VARIATIONS = frozenset({"P-5", "P-6"})
LLAMA_CONTROL_VARIATIONS = frozenset({"P-1", "P-2", "P-3", "P-4"})
REQUIRED_COLUMNS = {
    "experiment_id",
    "timestamp",
    "model",
    "pr_id",
    "variation",
    "seed",
    "decoding_params",
    "prompt_hash",
    "prompt_token_count",
    "generated_token_count",
    "full_prompt",
    "generated_text",
    "device_map",
    "finish_reason",
    "provider",
    "precision",
    "truncated_token_count",
    "system_fingerprint",
}
MODEL_SPECS = {
    "llama": {
        "model": "meta-llama/llama-3.1-70b-instruct",
        "provider": "coreweave",
        "precision": "bf16",
        "decoding": {"temperature": 1, "seed": 42},
        "variations": LLAMA_CONTROL_VARIATIONS | ISSUE_VARIATIONS,
    },
    "qwen": {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "provider": "novita",
        "precision": "bf16",
        "decoding": {"temperature": 1, "seed": 42},
        "variations": ISSUE_VARIATIONS,
    },
    "gpt": {
        "model": "gpt-5-mini-2025-08-07",
        "provider": "OpenAI",
        "precision": "auto",
        "decoding": {"seed": 42},
        "variations": ISSUE_VARIATIONS,
    },
}


class GenerationVerificationError(RuntimeError):
    """Raised when corrected generation output violates a frozen gate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GenerationVerificationError(message)


def load_sampled_pr_ids(path: Path) -> set[str]:
    require(path.is_file(), f"Sampled-PR dataset is missing: {path}")
    require(
        sha256_file(path) == EXPECTED_SAMPLED_PRS_SHA256,
        "Sampled-PR dataset hash differs",
    )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    require("id" in frame.columns, "Sampled-PR dataset lacks the id column")
    require(len(frame) == EXPECTED_SAMPLED_PR_COUNT, "Sampled-PR row count differs")
    ids = set(frame["id"])
    require(len(ids) == EXPECTED_SAMPLED_PR_COUNT, "Sampled-PR IDs are not unique")
    require(all(ids), "Sampled-PR dataset contains an empty ID")
    return ids


def extract_issue_block(prompt: str) -> str:
    require(prompt.count("<issues>") == 1, "Prompt lacks one exact issue opening tag")
    require(prompt.count("</issues>") == 1, "Prompt lacks one exact issue closing tag")
    _before, opening, remainder = prompt.partition("<issues>\n")
    require(bool(opening), "Prompt issue opening delimiter differs")
    block, closing, _after = remainder.partition("\n</issues>")
    require(bool(closing), "Prompt issue closing delimiter differs")
    return block


def parse_decoding(value: str, label: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise GenerationVerificationError(
            f"{label}: invalid decoding_params JSON"
        ) from exc
    require(isinstance(decoded, Mapping), f"{label}: decoding_params is not an object")
    return decoded


def require_complete_variations(
    frame: pd.DataFrame,
    variations: frozenset[str],
    expected_prs: int,
    label: str,
) -> set[str]:
    subset = frame[frame["variation"].isin(variations)]
    require(
        len(subset) == expected_prs * len(variations),
        f"{label}: row count differs",
    )
    require(set(subset["variation"]) == variations, f"{label}: context set differs")
    per_pr = subset.groupby("pr_id")["variation"].nunique()
    require(len(per_pr) == expected_prs, f"{label}: unique PR count differs")
    require(
        per_pr.eq(len(variations)).all(),
        f"{label}: a PR lacks a required context",
    )
    variation_sets = [
        set(subset.loc[subset["variation"] == variation, "pr_id"])
        for variation in sorted(variations)
    ]
    require(
        all(pr_ids == variation_sets[0] for pr_ids in variation_sets[1:]),
        f"{label}: context PR sets differ",
    )
    return variation_sets[0]


def validate_output(
    label: str,
    path: Path,
    manifest: Mapping[str, Mapping[str, Any]],
    sampled_pr_ids: set[str],
    *,
    expected_issue_prs: int,
    expected_llama_control_prs: int,
) -> dict[str, Any]:
    require(label in MODEL_SPECS, f"Unknown model label: {label}")
    require(path.is_file(), f"{label}: output is missing: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    require(not missing, f"{label}: missing columns {missing}")

    duplicate = frame.duplicated(["pr_id", "variation"], keep=False)
    require(not duplicate.any(), f"{label}: duplicate PR/context rows")
    spec = MODEL_SPECS[label]
    require(set(frame["variation"]) == spec["variations"], f"{label}: context set differs")

    issue_pr_ids = require_complete_variations(
        frame,
        ISSUE_VARIATIONS,
        expected_issue_prs,
        f"{label} P5/P6",
    )
    require(issue_pr_ids.issubset(manifest), f"{label}: output contains an ineligible issue PR")
    if expected_issue_prs == EXPECTED_PR_COUNT:
        require(issue_pr_ids == set(manifest), f"{label}: full issue PR set differs")

    control_pr_ids: set[str] = set()
    if label == "llama":
        control_pr_ids = require_complete_variations(
            frame,
            LLAMA_CONTROL_VARIATIONS,
            expected_llama_control_prs,
            "llama P1-P4",
        )
        require(
            control_pr_ids.issubset(sampled_pr_ids),
            "llama: output contains an unknown control PR",
        )
        if expected_llama_control_prs == EXPECTED_SAMPLED_PR_COUNT:
            require(
                control_pr_ids == sampled_pr_ids,
                "llama: full control PR set differs",
            )
        else:
            require(
                control_pr_ids == issue_pr_ids,
                "llama: partial control and issue PR sets differ",
            )

    expected_rows = expected_issue_prs * 2
    if label == "llama":
        expected_rows += expected_llama_control_prs * 4
    require(len(frame) == expected_rows, f"{label}: total row count differs")

    for column in ("model", "provider", "precision"):
        require(
            frame[column].eq(str(spec[column])).all(),
            f"{label}: {column} differs",
        )
    require(frame["device_map"].eq("api").all(), f"{label}: non-API row found")
    require(frame["seed"].eq("42").all(), f"{label}: seed differs")
    require(frame["finish_reason"].eq("stop").all(), f"{label}: non-stop finish")

    decoded = [parse_decoding(value, label) for value in frame["decoding_params"]]
    for row, observed_decoding in zip(frame.itertuples(index=False), decoded):
        expected_decoding = dict(spec["decoding"])
        if (
            label == "llama"
            and row.variation == "P-4"
            and str(row.truncated_token_count).strip()
        ):
            expected_decoding["max_tokens"] = LLAMA_TRUNCATED_P4_MAX_TOKENS
        require(
            dict(observed_decoding) == expected_decoding,
            f"{label}: decoding parameters differ",
        )

    generated = frame["generated_text"].str.strip()
    require(generated.ne("").all(), f"{label}: empty generated text")
    upper_generated = generated.str.upper()
    require(
        not upper_generated.str.contains("ERROR_GENERATION_FAILED", regex=False).any(),
        f"{label}: API failure marker found",
    )
    require(
        not upper_generated.str.contains("MOCKED DESCRIPTION", regex=False).any(),
        f"{label}: mock output found",
    )
    prompt_tokens = pd.to_numeric(frame["prompt_token_count"], errors="coerce")
    generated_tokens = pd.to_numeric(frame["generated_token_count"], errors="coerce")
    require(
        prompt_tokens.notna().all() and prompt_tokens.gt(0).all(),
        f"{label}: prompt token count invalid",
    )
    require(
        generated_tokens.notna().all() and generated_tokens.gt(0).all(),
        f"{label}: generated token count invalid",
    )

    truncation_raw = frame["truncated_token_count"].str.strip()
    truncation_present = truncation_raw.ne("")
    truncated_tokens = pd.to_numeric(truncation_raw, errors="coerce")
    if label == "llama":
        require(
            frame.loc[truncation_present, "variation"].eq("P-4").all(),
            "llama: truncation metadata appears outside P4",
        )
        require(
            truncated_tokens[truncation_present].notna().all()
            and truncated_tokens[truncation_present].gt(0).all()
            and truncated_tokens[truncation_present]
            .le(LLAMA_P4_FILE_TOKEN_BUDGET)
            .all(),
            "llama: P4 truncation exceeds the approved file-token budget",
        )
        if expected_llama_control_prs == EXPECTED_SAMPLED_PR_COUNT:
            require(
                int(truncation_present.sum())
                == EXPECTED_FULL_RUN_LLAMA_P4_TRUNCATIONS,
                "llama: full-run P4 truncation count differs",
            )
    else:
        require(
            not truncation_present.any(),
            f"{label}: unexpected truncation metadata",
        )

    issue_hashes: dict[tuple[str, str], str] = {}
    for row in frame.itertuples(index=False):
        full_prompt = row.full_prompt
        observed_md5 = hashlib.md5(full_prompt.encode("utf-8")).hexdigest()
        require(observed_md5 == row.prompt_hash, f"{label}/{row.pr_id}: prompt MD5 differs")
        if row.variation in ISSUE_VARIATIONS:
            observed_block = extract_issue_block(full_prompt)
            source = manifest[row.pr_id]
            require(
                observed_block == source["prompt_issue_block"],
                f"{label}/{row.pr_id}: issue block differs from manifest",
            )
            require(
                sha256_text(observed_block) == source["prompt_issue_block_sha256"],
                f"{label}/{row.pr_id}: issue block hash differs",
            )
            issue_hashes[(row.pr_id, row.variation)] = sha256_text(observed_block)
        elif row.pr_id in manifest:
            require(
                manifest[row.pr_id]["prompt_issue_block"] not in full_prompt,
                f"{label}/{row.pr_id}: control prompt contains the audited issue block",
            )

    for pr_id in issue_pr_ids:
        require(
            issue_hashes[(pr_id, "P-5")] == issue_hashes[(pr_id, "P-6")],
            f"{label}/{pr_id}: P5/P6 issue blocks differ",
        )

    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "issue_prs": len(issue_pr_ids),
        "control_prs": len(control_pr_ids),
        "variations": sorted(set(frame["variation"])),
        "model": spec["model"],
        "provider": spec["provider"],
        "precision": spec["precision"],
        "p4_truncated_rows": int(truncation_present.sum()),
        "p4_file_token_budget": (
            LLAMA_P4_FILE_TOKEN_BUDGET if label == "llama" else None
        ),
        "truncated_p4_completion_token_cap": (
            LLAMA_TRUNCATED_P4_MAX_TOKENS if label == "llama" else None
        ),
        "system_fingerprints": sorted(
            value for value in set(frame["system_fingerprint"]) if value
        ),
        "finish_reasons": frame["finish_reason"].value_counts().to_dict(),
    }


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), *args], text=True
    ).strip()


def main() -> None:
    require_nokogiri()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sampled-prs", type=Path, required=True)
    parser.add_argument("--expected-issue-prs", type=int, default=EXPECTED_PR_COUNT)
    parser.add_argument(
        "--expected-llama-control-prs",
        type=int,
        default=EXPECTED_SAMPLED_PR_COUNT,
    )
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    require(1 <= args.expected_issue_prs <= EXPECTED_PR_COUNT, "Invalid expected issue PR count")
    require(
        1 <= args.expected_llama_control_prs <= EXPECTED_SAMPLED_PR_COUNT,
        "Invalid expected Llama control PR count",
    )
    require(
        args.expected_llama_control_prs == EXPECTED_SAMPLED_PR_COUNT
        or args.expected_llama_control_prs == args.expected_issue_prs,
        "Partial Llama controls must match the issue PR count",
    )
    report_path = args.report.resolve()
    require(not report_path.exists(), f"Refusing to overwrite report: {report_path}")
    manifest = load_issue_source_manifest(args.manifest.resolve())
    sampled_pr_ids = load_sampled_pr_ids(args.sampled_prs.resolve())
    require(set(manifest).issubset(sampled_pr_ids), "Issue manifest contains an unsampled PR")

    paths = {"llama": args.llama, "qwen": args.qwen, "gpt": args.gpt}
    outputs = {
        label: validate_output(
            label,
            path.resolve(),
            manifest,
            sampled_pr_ids,
            expected_issue_prs=args.expected_issue_prs,
            expected_llama_control_prs=args.expected_llama_control_prs,
        )
        for label, path in paths.items()
    }
    issue_pr_sets = {
        label: set(
            pd.read_csv(path, dtype=str, keep_default_na=False)
            .loc[lambda frame: frame["variation"].isin(ISSUE_VARIATIONS), "pr_id"]
        )
        for label, path in paths.items()
    }
    require(
        issue_pr_sets["llama"] == issue_pr_sets["qwen"] == issue_pr_sets["gpt"],
        "Generator issue PR sets differ",
    )

    total_calls = (
        args.expected_llama_control_prs * len(LLAMA_CONTROL_VARIATIONS)
        + args.expected_issue_prs * len(ISSUE_VARIATIONS) * len(MODEL_SPECS)
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete_and_verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "sampled_prs_path": str(args.sampled_prs.resolve()),
        "sampled_prs_sha256": sha256_file(args.sampled_prs.resolve()),
        "expected_sampled_prs_sha256": EXPECTED_SAMPLED_PRS_SHA256,
        "expected_issue_prs_per_model": args.expected_issue_prs,
        "expected_llama_control_prs": args.expected_llama_control_prs,
        "model_api_calls_expected": total_calls,
        "total_rows": sum(output["rows"] for output in outputs.values()),
        "full_273_issue_308_control_run": (
            args.expected_issue_prs == EXPECTED_PR_COUNT
            and args.expected_llama_control_prs == EXPECTED_SAMPLED_PR_COUNT
        ),
        "integrity_gates": {
            "sampled_and_issue_sources_hash_pinned": True,
            "eligible_issue_pr_set_matches_requested_limit": True,
            "llama_control_pr_set_matches_requested_limit": True,
            "required_contexts_complete": True,
            "unique_pr_context_pairs": True,
            "provider_model_precision_match": True,
            "decoding_and_request_seed_match": True,
            "all_finish_reasons_stop": True,
            "no_empty_error_or_mock_outputs": True,
            "positive_token_counts": True,
            "p4_truncation_within_approved_budget": True,
            "truncated_p4_completion_cap_match": True,
            "prompt_hashes_match": True,
            "issue_blocks_match_frozen_manifest": True,
            "control_prompts_exclude_audited_issue_blocks": True,
            "p5_p6_issue_blocks_identical": True,
            "generator_issue_pr_sets_identical": True,
        },
        "outputs": outputs,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temp_report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temp_report.replace(report_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
