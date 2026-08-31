"""Independently verify the B4 prompt-only smoke outputs. SERVER ONLY."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from deep_audit.server_guard import require_nokogiri

from phase3.generation.issue_sources import sha256_file


EXPECTED_COUNTS = {
    "all_prs": 308,
    "eligible_prs": 273,
    "ineligible_prs": 35,
    "genuine_issues": 293,
    "positive_prompt_checks": 1638,
    "negative_fail_closed_checks": 210,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    require_nokogiri()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    report = args.report.resolve()
    if report.exists():
        raise FileExistsError(f"Refusing to overwrite verification report: {report}")

    completion = json.loads((output_dir / "completion.json").read_text())
    require(completion["status"] == "complete", "Smoke status is not complete")
    require(completion["model_calls"] == 0, "Smoke made or recorded model calls")
    require(completion["counts"] == EXPECTED_COUNTS, "Frozen counts differ")
    require(
        all(completion["integrity_gates"].values()),
        "At least one recorded integrity gate failed",
    )
    for name, expected in completion["output_sha256"].items():
        require(sha256_file(output_dir / name) == expected, f"Hash mismatch: {name}")

    checks = pd.read_csv(
        output_dir / "prompt_checks.csv", dtype=str, keep_default_na=False
    )
    require(len(checks) == 1638, "Prompt-check row count differs")
    require(checks["passed"].eq("True").all(), "A prompt check is not passing")
    require(
        checks["expected_issue_block_sha256"].eq(
            checks["observed_issue_block_sha256"]
        ).all(),
        "Observed issue hash differs from frozen source",
    )
    require(checks["pr_id"].nunique() == 273, "Eligible PR count differs")
    require(set(checks["builder"]) == {"chatgpt", "llama", "qwen"}, "Builder set differs")
    require(set(checks["variation"]) == {"P-5", "P-6"}, "Variation set differs")
    per_pr = checks.groupby("pr_id").size()
    require(per_pr.eq(6).all(), "Each eligible PR must have six builder/context rows")
    per_source = checks.groupby(
        ["pr_id", "builder"]
    )["observed_issue_block_sha256"].nunique()
    require(per_source.eq(1).all(), "P5/P6 issue hashes differ")

    result = {
        "status": "verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "model_calls": 0,
        "recorded_output_hashes_verified": len(completion["output_sha256"]),
        "row_counts": {"prompt_checks.csv": len(checks)},
        "checks": {
            "hashes": "pass",
            "frozen_counts": "pass",
            "builder_and_variation_coverage": "pass",
            "issue_block_hashes": "pass",
            "p5_p6_identity": "pass",
            "ineligible_fail_closed": "pass",
        },
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
