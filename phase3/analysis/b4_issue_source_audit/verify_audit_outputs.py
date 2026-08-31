from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from audit_utils import prompt_block, sha256_file, sha256_text


EXPECTED_COUNTS = {
    "source_rows": 308,
    "artifacts": 352,
    "issues": 293,
    "pull_requests": 59,
    "eligible_prs": 273,
    "excluded_prs": 35,
    "mixed_prs": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently verify B4 source-audit outputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    report = args.report.resolve()
    if report.exists():
        raise FileExistsError(f"Refusing to overwrite verification report: {report}")

    completion = json.loads((output_dir / "completion.json").read_text())
    require(completion["status"] == "complete", "Audit completion status is not complete")
    require(completion["counts"] == EXPECTED_COUNTS, "Completion counts differ from frozen counts")
    recorded = completion["output_sha256"]
    for name, expected in recorded.items():
        require(sha256_file(output_dir / name) == expected, f"Hash mismatch: {name}")

    manifest = pd.read_csv(output_dir / "artifact_manifest.csv", dtype=str, keep_default_na=False)
    genuine = pd.read_csv(output_dir / "genuine_issue_manifest.csv", dtype=str, keep_default_na=False)
    eligibility = pd.read_csv(output_dir / "pr_eligibility.csv", dtype=str, keep_default_na=False)
    excluded = pd.read_csv(output_dir / "excluded_prs.csv", dtype=str, keep_default_na=False)
    anomalies = pd.read_csv(output_dir / "audit_anomalies.csv", dtype=str, keep_default_na=False)

    require(len(manifest) == 352, "Artifact manifest must contain 352 rows")
    require(len(genuine) == 293, "Genuine issue manifest must contain 293 rows")
    require(len(eligibility) == 308, "Eligibility table must contain 308 rows")
    require(len(excluded) == 35, "Excluded PR table must contain 35 rows")
    require(anomalies.empty, "Audit anomalies table is not empty")
    require(manifest["reference_found"].eq("True").all(), "Missing PR-body reference evidence")
    require(set(manifest["stored_artifact_type"]) == {"issue", "pull_request"}, "Unexpected artifact type")
    require(genuine["stored_artifact_type"].eq("issue").all(), "Non-issue in genuine manifest")
    require(~genuine["stored_url"].str.contains("/pull/").any(), "Pull URL in genuine manifest")
    require(
        genuine["stored_title_empty"].eq("False").all(),
        "Empty title in genuine issue manifest",
    )
    require(
        genuine["stored_body_empty"].eq("False").all(),
        "Empty body in genuine issue manifest",
    )
    require(eligibility["eligible"].eq("True").sum() == 273, "Eligible PR count mismatch")

    records = [json.loads(line) for line in (output_dir / "regeneration_sources.jsonl").read_text().splitlines() if line]
    require(len(records) == 273, "Regeneration source count mismatch")
    manifest_keys = {
        (row["pr_id"], int(row["issue_number"]), row["stored_title_sha256"], row["stored_body_sha256"])
        for _, row in genuine.iterrows()
    }
    record_keys = set()
    for record in records:
        issues_for_block = []
        for issue in record["issues"]:
            require(sha256_text(issue["title"]) == issue["title_sha256"], "Regeneration title hash mismatch")
            require(sha256_text(issue["body"]) == issue["body_sha256"], "Regeneration body hash mismatch")
            require("/issues/" in issue["url"], "Regeneration record contains non-issue URL")
            record_keys.add((record["pr_id"], int(issue["issue_number"]), issue["title_sha256"], issue["body_sha256"]))
            issues_for_block.append({"stored_title": issue["title"], "stored_body": issue["body"]})
        block = prompt_block(issues_for_block)
        require(block == record["prompt_issue_block"], "Prompt block reconstruction mismatch")
        require(sha256_text(block) == record["prompt_issue_block_sha256"], "Prompt block hash mismatch")
    require(record_keys == manifest_keys, "Regeneration records differ from genuine manifest")

    api_path = output_dir / "api_verification.csv"
    api_verified = completion["api_verified"]
    if api_verified:
        api = pd.read_csv(api_path, dtype=str, keep_default_na=False)
        require(len(api) == 352, "API verification must contain 352 rows")
        require(api["api_ok"].eq("True").all(), "At least one API request failed")
        require(api["stored_type_matches_api"].eq("True").all(), "API type mismatch")
    else:
        require(not api_path.exists(), "API file exists although audit says API was not run")

    result = {
        "status": "verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "api_verified": api_verified,
        "recorded_output_hashes_verified": len(recorded),
        "row_counts": {
            "artifact_manifest.csv": len(manifest),
            "genuine_issue_manifest.csv": len(genuine),
            "pr_eligibility.csv": len(eligibility),
            "excluded_prs.csv": len(excluded),
            "regeneration_sources.jsonl": len(records),
        },
        "checks": {
            "hashes": "pass",
            "artifact_types": "pass",
            "reference_evidence": "pass",
            "genuine_issue_filter": "pass",
            "genuine_issue_title_body_nonempty": "pass",
            "regeneration_hashes_and_prompt_blocks": "pass",
            "api_identity_and_type": "pass" if api_verified else "not_run",
        },
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
