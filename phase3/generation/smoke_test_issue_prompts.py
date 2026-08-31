"""Exhaustive, model-free P5/P6 prompt smoke test. SERVER ONLY."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from deep_audit.server_guard import require_nokogiri

from phase3.data_loader import DataLoader
from phase3.generation.generate_chatgpt import (
    PromptBuilder as ChatGPTPromptBuilder,
    VARIATIONS as CHATGPT_VARIATIONS,
)
from phase3.generation.generate_llama import PromptBuilder as LlamaPromptBuilder
from phase3.generation.generate_qwen import PromptBuilder as QwenPromptBuilder
from phase3.generation.issue_sources import (
    EXPECTED_ISSUE_COUNT,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PR_COUNT,
    IssueSourceError,
    has_audited_issue_source,
    sha256_file,
    sha256_text,
)


PROTOCOL_VERSION = "b4-prompt-smoke-v1"
EXPECTED_ALL_PRS = 308
BUILDERS = ("chatgpt", "llama", "qwen")
VARIATIONS = ("P-5", "P-6")
EXPECTED_POSITIVE_CHECKS = EXPECTED_PR_COUNT * len(BUILDERS) * len(VARIATIONS)
EXPECTED_NEGATIVE_CHECKS = (
    (EXPECTED_ALL_PRS - EXPECTED_PR_COUNT) * len(BUILDERS) * len(VARIATIONS)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify corrected P5/P6 prompt construction without model calls. SERVER ONLY."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase3/analysis/outputs/tier_b/b4-prompt-smoke-v1"),
    )
    return parser.parse_args()


def build_chatgpt(pr_data: dict[str, Any], variation: str) -> str:
    return ChatGPTPromptBuilder.build_prompt(pr_data, CHATGPT_VARIATIONS[variation])


def build_llama(pr_data: dict[str, Any], variation: str) -> str:
    return LlamaPromptBuilder.construct_context_block(pr_data, variation)


def build_qwen(pr_data: dict[str, Any], variation: str) -> str:
    return QwenPromptBuilder.construct_context_block(pr_data, variation)


BUILDER_FUNCTIONS: dict[str, Callable[[dict[str, Any], str], str]] = {
    "chatgpt": build_chatgpt,
    "llama": build_llama,
    "qwen": build_qwen,
}


def extract_issue_block(prompt: str) -> str:
    if prompt.count("<issues>") != 1 or prompt.count("</issues>") != 1:
        raise IssueSourceError("Prompt does not contain exactly one issue section")
    before, marker, remainder = prompt.partition("<issues>\n")
    if not marker:
        raise IssueSourceError("Issue section lacks the exact opening delimiter")
    block, closing, _after = remainder.partition("\n</issues>")
    if not closing:
        raise IssueSourceError("Issue section lacks the exact closing delimiter")
    if "<issues>" in before:
        raise IssueSourceError("Unexpected nested issue opening tag")
    return block


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def main() -> None:
    require_nokogiri()
    args = parse_args()
    root = args.repo_root.resolve()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    ).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite prompt smoke output: {output_dir}")

    loader = DataLoader(base_path=str(root / "phase3/datasets"))
    all_pr_data = loader.get_comprehensive_data(require_issue_manifest=True)
    if len(all_pr_data) != EXPECTED_ALL_PRS:
        raise RuntimeError(f"Expected {EXPECTED_ALL_PRS} PRs, found {len(all_pr_data)}")

    eligible = {
        pr_id: pr_data
        for pr_id, pr_data in all_pr_data.items()
        if has_audited_issue_source(pr_data)
    }
    ineligible = {
        pr_id: pr_data
        for pr_id, pr_data in all_pr_data.items()
        if not has_audited_issue_source(pr_data)
    }
    issue_count = sum(len(pr_data["issues"]) for pr_data in eligible.values())
    if len(eligible) != EXPECTED_PR_COUNT or issue_count != EXPECTED_ISSUE_COUNT:
        raise RuntimeError(
            f"Frozen source counts changed: PRs={len(eligible)}, issues={issue_count}"
        )

    rows: list[dict[str, Any]] = []
    for pr_id, pr_data in eligible.items():
        expected_block = pr_data["issue_source"]["prompt_issue_block"]
        expected_hash = pr_data["issue_source"]["prompt_issue_block_sha256"]
        if sha256_text(expected_block) != expected_hash:
            raise RuntimeError(f"PR {pr_id}: attached source hash mismatch")
        for builder_name, builder in BUILDER_FUNCTIONS.items():
            observed_by_variation = {}
            for variation in VARIATIONS:
                prompt = builder(pr_data, variation)
                observed_block = extract_issue_block(prompt)
                observed_hash = sha256_text(observed_block)
                if observed_block != expected_block or observed_hash != expected_hash:
                    raise RuntimeError(
                        f"PR {pr_id}: {builder_name}/{variation} differs from source"
                    )
                if variation == "P-5" and any(
                    tag in prompt
                    for tag in (
                        "<repository>",
                        "<affected_files>",
                        "<title>",
                        "<diffs>",
                        "<files>",
                    )
                ):
                    raise RuntimeError(
                        f"PR {pr_id}: {builder_name}/P-5 contains non-issue context"
                    )
                observed_by_variation[variation] = observed_hash
                rows.append(
                    {
                        "pr_id": pr_id,
                        "pull_number": pr_data["info"].get("pull_number", ""),
                        "builder": builder_name,
                        "variation": variation,
                        "issue_count": len(pr_data["issues"]),
                        "expected_issue_block_sha256": expected_hash,
                        "observed_issue_block_sha256": observed_hash,
                        "full_prompt_sha256": sha256_text(prompt),
                        "full_prompt_chars": len(prompt),
                        "issue_block_chars": len(observed_block),
                        "passed": True,
                    }
                )
            if observed_by_variation["P-5"] != observed_by_variation["P-6"]:
                raise RuntimeError(
                    f"PR {pr_id}: {builder_name} P5/P6 issue blocks differ"
                )

    negative_checks = 0
    for pr_id, pr_data in ineligible.items():
        for builder_name, builder in BUILDER_FUNCTIONS.items():
            for variation in VARIATIONS:
                try:
                    builder(pr_data, variation)
                except IssueSourceError:
                    negative_checks += 1
                else:
                    raise RuntimeError(
                        f"PR {pr_id}: {builder_name}/{variation} accepted no issue source"
                    )

    checks = pd.DataFrame(rows).sort_values(
        ["pr_id", "builder", "variation"]
    ).reset_index(drop=True)
    if len(checks) != EXPECTED_POSITIVE_CHECKS:
        raise RuntimeError(f"Expected {EXPECTED_POSITIVE_CHECKS} positive checks")
    if negative_checks != EXPECTED_NEGATIVE_CHECKS:
        raise RuntimeError(f"Expected {EXPECTED_NEGATIVE_CHECKS} negative checks")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        checks.to_csv(temp_dir / "prompt_checks.csv", index=False)
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "status": "complete",
            "model_calls": 0,
            "all_prs": len(all_pr_data),
            "eligible_prs": len(eligible),
            "ineligible_prs": len(ineligible),
            "genuine_issues": issue_count,
            "builders": list(BUILDERS),
            "variations": list(VARIATIONS),
            "positive_prompt_checks": len(checks),
            "negative_fail_closed_checks": negative_checks,
            "all_issue_blocks_match_frozen_source": True,
            "p5_contains_only_issue_context": True,
        }
        (temp_dir / "audit_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "python": sys.version,
            "pandas": pd.__version__,
            "repo_root": str(root),
            "git_branch": git_value(root, "branch", "--show-current"),
            "git_commit": git_value(root, "rev-parse", "HEAD"),
            "issue_manifest_path": str(loader.issue_manifest_path),
            "issue_manifest_sha256": sha256_file(loader.issue_manifest_path),
            "expected_issue_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        }
        (temp_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_hashes = {
            path.name: sha256_file(path)
            for path in sorted(temp_dir.iterdir())
            if path.is_file()
        }
        completion = {
            "protocol_version": PROTOCOL_VERSION,
            "status": "complete",
            "model_calls": 0,
            "counts": {
                "all_prs": len(all_pr_data),
                "eligible_prs": len(eligible),
                "ineligible_prs": len(ineligible),
                "genuine_issues": issue_count,
                "positive_prompt_checks": len(checks),
                "negative_fail_closed_checks": negative_checks,
            },
            "integrity_gates": {
                "frozen_manifest_hash_match": True,
                "all_builders_use_exact_frozen_issue_block": True,
                "p5_p6_issue_blocks_identical": True,
                "p5_contains_no_nonissue_context_tags": True,
                "ineligible_prs_fail_closed": True,
                "model_calls_zero": True,
            },
            "output_sha256": output_hashes,
        }
        (temp_dir / "completion.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Prompt smoke outputs written to {output_dir}")


if __name__ == "__main__":
    main()
