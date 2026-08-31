from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from deep_audit.server_guard import require_nokogiri

from audit_utils import (
    artifact_type_from_url,
    parse_aggregate,
    parse_issue_numbers,
    prompt_block,
    reference_evidence,
    sha256_file,
    sha256_text,
)


PROTOCOL_VERSION = "b4-issue-source-audit-v1"
EXPECTED_HASHES = {
    "source": "c965da8ea751d7c0e9114a8bc6aa6fb6ae925f8a36a8c4cfcb4a86415e4120f2",
    "sampled": "d00daddbf1ed5d054bc3d126248673e9487f381f46c3ae64e68af80e1d139a8e",
}
EXPECTED_COUNTS = {
    "source_rows": 308,
    "artifacts": 352,
    "issues": 293,
    "pull_requests": 59,
    "eligible_prs": 273,
    "excluded_prs": 35,
    "mixed_prs": 20,
}

AGGREGATE_FIELDS = {
    "issue_ids": "stored_id",
    "issue_titles": "stored_title",
    "issue_bodies": "stored_body",
    "issue_states": "stored_state",
    "issue_created_ats": "stored_created_at",
    "issue_updated_ats": "stored_updated_at",
    "issue_closed_ats": "stored_closed_at",
    "issue_authors": "stored_author",
    "issue_urls": "stored_url",
}

REFERENCE_REPOSITORY_ALIASES = {
    # The authoritative sampling notebook explicitly accepted both locations.
    "jabref/jabref": ("koppor/jabref",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PR-body issue references and build corrected issue sources. SERVER ONLY."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("phase3/datasets/selected_prs_with_related_issues.csv"),
    )
    parser.add_argument(
        "--sampled",
        type=Path,
        default=Path("phase3/datasets/selected_prs_sampled.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase3/analysis/outputs/tier_b/b4-issue-source-audit-v1"),
    )
    parser.add_argument("--with-api", action="store_true")
    parser.add_argument("--tokens-file", type=Path)
    parser.add_argument("--api-workers", type=int, default=6)
    return parser.parse_args()


def resolve_under(root: Path, value: Path) -> Path:
    path = value if value.is_absolute() else root / value
    return path.resolve()


def load_sources(source_path: Path, sampled_path: Path) -> tuple[pd.DataFrame, dict]:
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    sampled = pd.read_csv(sampled_path, dtype=str, keep_default_na=False)
    if len(source) != EXPECTED_COUNTS["source_rows"] or source["id"].nunique() != len(source):
        raise RuntimeError("Expected 308 unique source PR rows")
    if len(sampled) != len(source) or sampled["id"].nunique() != len(sampled):
        raise RuntimeError("Sampled PR source must contain the same 308 unique PRs")
    comparison = source[["id", "title", "body"]].merge(
        sampled[["id", "title", "body"]],
        on="id",
        suffixes=("_merged", "_sampled"),
        validate="one_to_one",
    )
    title_matches = comparison["title_merged"].eq(comparison["title_sampled"])
    body_matches = comparison["body_merged"].eq(comparison["body_sampled"])
    if not title_matches.all() or not body_matches.all():
        raise RuntimeError("Copied PR title/body fields differ from authoritative sampled PRs")
    return source, {
        "source_rows": len(source),
        "copied_pr_titles_verified": int(title_matches.sum()),
        "copied_pr_bodies_verified": int(body_matches.sum()),
    }


def build_manifest(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    anomalies: list[dict[str, object]] = []
    for _, pr in source.sort_values("id").iterrows():
        try:
            numbers = parse_issue_numbers(pr["issue_numbers"])
            referenced = parse_issue_numbers(pr["issue_references"])
            if set(numbers) != set(referenced):
                raise ValueError(
                    f"issue_numbers {numbers} != issue_references {referenced}"
                )
            parsed_fields = {
                output_name: parse_aggregate(pr[input_name], numbers, input_name)
                for input_name, output_name in AGGREGATE_FIELDS.items()
            }
        except Exception as exc:
            anomalies.append(
                {"pr_id": pr["id"], "stage": "parse", "detail": str(exc)}
            )
            continue

        repository = pr["repository_name_with_owner"]
        for order, number in enumerate(numbers, start=1):
            artifact = {
                "pr_id": pr["id"],
                "pull_number": pr["pull_number"],
                "repository": repository,
                "pr_title": pr["title"],
                "pr_body_sha256": sha256_text(pr["body"]),
                "pr_created_at": pr["created_at"],
                "pr_updated_at": pr["updated_at"],
                "pr_merged_at": pr["merged_at"],
                "reference_order": order,
                "issue_number": number,
            }
            for output_name, values in parsed_fields.items():
                artifact[output_name] = values[order - 1]
            artifact_type = artifact_type_from_url(
                str(artifact["stored_url"]), repository, number
            )
            evidence = reference_evidence(
                pr["body"],
                repository,
                number,
                REFERENCE_REPOSITORY_ALIASES.get(repository.lower(), ()),
            )
            artifact.update(evidence)
            artifact["stored_artifact_type"] = artifact_type
            artifact["stored_title_sha256"] = sha256_text(
                str(artifact["stored_title"])
            )
            artifact["stored_body_sha256"] = sha256_text(
                str(artifact["stored_body"])
            )
            artifact["stored_title_empty"] = not bool(
                str(artifact["stored_title"]).strip()
            )
            artifact["stored_body_empty"] = not bool(
                str(artifact["stored_body"]).strip()
            )
            artifact["eligible_issue_source"] = artifact_type == "issue"
            artifact["pr_target_collision"] = (
                str(artifact["stored_title"]) == pr["title"]
                and str(artifact["stored_body"]) == pr["body"]
            )
            rows.append(artifact)

    manifest = pd.DataFrame(rows).sort_values(
        ["pr_id", "reference_order"]
    ).reset_index(drop=True)
    anomaly_frame = pd.DataFrame(anomalies, columns=["pr_id", "stage", "detail"])
    return manifest, anomaly_frame


def build_pr_eligibility(manifest: pd.DataFrame) -> pd.DataFrame:
    records = []
    for pr_id, group in manifest.groupby("pr_id", sort=True):
        group = group.sort_values("reference_order")
        issues = group[group["stored_artifact_type"].eq("issue")]
        pulls = group[group["stored_artifact_type"].eq("pull_request")]
        records.append(
            {
                "pr_id": pr_id,
                "pull_number": group["pull_number"].iloc[0],
                "repository": group["repository"].iloc[0],
                "referenced_artifacts": len(group),
                "genuine_issue_count": len(issues),
                "pull_request_count": len(pulls),
                "unknown_artifact_count": int(
                    group["stored_artifact_type"].eq("unknown").sum()
                ),
                "eligible": len(issues) > 0,
                "mixed_issue_and_pull": len(issues) > 0 and len(pulls) > 0,
                "genuine_issue_numbers": ",".join(
                    str(value) for value in issues["issue_number"]
                ),
                "excluded_pull_numbers": ",".join(
                    str(value) for value in pulls["issue_number"]
                ),
            }
        )
    return pd.DataFrame(records).sort_values("pr_id").reset_index(drop=True)


def build_regeneration_records(manifest: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    genuine = manifest[manifest["stored_artifact_type"].eq("issue")]
    for pr_id, group in genuine.groupby("pr_id", sort=True):
        group = group.sort_values("reference_order")
        issues = [
            {
                "reference_order": int(row["reference_order"]),
                "issue_number": int(row["issue_number"]),
                "url": row["stored_url"],
                "title": row["stored_title"],
                "body": row["stored_body"],
                "title_sha256": row["stored_title_sha256"],
                "body_sha256": row["stored_body_sha256"],
                "created_at": row["stored_created_at"],
                "updated_at": row["stored_updated_at"],
            }
            for _, row in group.iterrows()
        ]
        block = prompt_block(
            [
                {"stored_title": item["title"], "stored_body": item["body"]}
                for item in issues
            ]
        )
        records.append(
            {
                "pr_id": pr_id,
                "pull_number": group["pull_number"].iloc[0],
                "repository": group["repository"].iloc[0],
                "issues": issues,
                "prompt_issue_block": block,
                "prompt_issue_block_sha256": sha256_text(block),
            }
        )
    return records


def load_valid_tokens(path: Path) -> list[str]:
    tokens = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not tokens:
        raise RuntimeError(f"No non-empty tokens in {path}")
    valid = []
    for token in tokens:
        response = requests.get(
            "https://api.github.com/rate_limit",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "llm-in-prs-b4-source-audit",
            },
            timeout=30,
        )
        if response.status_code == 200:
            valid.append(token)
    if not valid:
        raise RuntimeError("No token in the supplied file passed GitHub authentication")
    return valid


def verify_one_api(row: dict[str, object], tokens: list[str], index: int) -> dict[str, object]:
    url = (
        f"https://api.github.com/repos/{row['repository']}/issues/"
        f"{int(row['issue_number'])}"
    )
    status = None
    error = ""
    payload = None
    for offset in range(len(tokens)):
        token = tokens[(index + offset) % len(tokens)]
        try:
            response = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": "llm-in-prs-b4-source-audit",
                },
                timeout=30,
            )
            status = response.status_code
            if status == 200:
                payload = response.json()
                break
            if status not in {401, 403, 429}:
                error = f"HTTP {status}"
                break
        except requests.RequestException as exc:
            error = type(exc).__name__
    result = {
        "pr_id": row["pr_id"],
        "repository": row["repository"],
        "issue_number": int(row["issue_number"]),
        "stored_artifact_type": row["stored_artifact_type"],
        "api_status_code": status,
        "api_ok": payload is not None,
        "api_error": error,
    }
    if payload is None:
        return result
    api_type = "pull_request" if "pull_request" in payload else "issue"
    api_title = payload.get("title") or ""
    api_body = payload.get("body") or ""
    result.update(
        {
            "api_artifact_type": api_type,
            "api_html_url": payload.get("html_url") or "",
            "api_title_sha256": sha256_text(api_title),
            "api_body_sha256": sha256_text(api_body),
            "api_created_at": payload.get("created_at") or "",
            "api_updated_at": payload.get("updated_at") or "",
            "api_state": payload.get("state") or "",
            "stored_type_matches_api": api_type == row["stored_artifact_type"],
            "stored_title_matches_current": (
                sha256_text(api_title) == row["stored_title_sha256"]
            ),
            "stored_body_matches_current": (
                sha256_text(api_body) == row["stored_body_sha256"]
            ),
        }
    )
    return result


def verify_api(manifest: pd.DataFrame, token_path: Path, workers: int) -> pd.DataFrame:
    tokens = load_valid_tokens(token_path)
    rows = manifest.to_dict(orient="records")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(verify_one_api, row, tokens, index): index
            for index, row in enumerate(rows)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0 or completed == len(rows):
                print(f"GitHub verification: {completed}/{len(rows)}", flush=True)
    frame = pd.DataFrame(results).sort_values(
        ["pr_id", "issue_number"]
    ).reset_index(drop=True)
    if not frame["api_ok"].all():
        failed = int((~frame["api_ok"]).sum())
        raise RuntimeError(f"GitHub API verification failed for {failed} artifacts")
    if not frame["stored_type_matches_api"].all():
        mismatches = int((~frame["stored_type_matches_api"]).sum())
        raise RuntimeError(f"Stored/API artifact type mismatch for {mismatches} artifacts")
    return frame


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    require_nokogiri()
    args = parse_args()
    if args.api_workers < 1:
        raise ValueError("--api-workers must be positive")
    root = args.repo_root.resolve()
    source_path = resolve_under(root, args.source)
    sampled_path = resolve_under(root, args.sampled)
    output_dir = resolve_under(root, args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {output_dir}")

    input_hashes = {
        "source": sha256_file(source_path),
        "sampled": sha256_file(sampled_path),
    }
    if input_hashes != EXPECTED_HASHES:
        raise RuntimeError(f"Frozen input hash mismatch: {input_hashes}")

    source, source_checks = load_sources(source_path, sampled_path)
    manifest, anomalies = build_manifest(source)
    if not anomalies.empty:
        raise RuntimeError(
            f"Artifact decomposition failed for {len(anomalies)} PRs: "
            f"{anomalies.head(3).to_dict(orient='records')}"
        )
    if not manifest["reference_found"].all():
        raise RuntimeError("At least one stored artifact number is absent from its PR body")
    if manifest["stored_artifact_type"].eq("unknown").any():
        raise RuntimeError("At least one stored artifact URL has unknown type/repository/number")
    genuine_issue_mask = manifest["stored_artifact_type"].eq("issue")
    genuine_issues = manifest[genuine_issue_mask]
    if (
        genuine_issues["stored_title_empty"].any()
        or genuine_issues["stored_body_empty"].any()
    ):
        raise RuntimeError("At least one retained genuine issue has an empty title or body")
    if manifest["pr_target_collision"].any():
        raise RuntimeError("At least one fetched artifact exactly equals its target PR")

    eligibility = build_pr_eligibility(manifest)
    observed_counts = {
        "source_rows": len(source),
        "artifacts": len(manifest),
        "issues": int(manifest["stored_artifact_type"].eq("issue").sum()),
        "pull_requests": int(
            manifest["stored_artifact_type"].eq("pull_request").sum()
        ),
        "eligible_prs": int(eligibility["eligible"].sum()),
        "excluded_prs": int((~eligibility["eligible"]).sum()),
        "mixed_prs": int(eligibility["mixed_issue_and_pull"].sum()),
    }
    if observed_counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Frozen count mismatch: {observed_counts}")

    regeneration = build_regeneration_records(manifest)
    if len(regeneration) != EXPECTED_COUNTS["eligible_prs"]:
        raise RuntimeError("Regeneration record count does not match eligible PR count")

    api_frame = pd.DataFrame()
    token_path = None
    if args.with_api:
        if args.tokens_file is None:
            raise ValueError("--with-api requires --tokens-file")
        token_path = resolve_under(root, args.tokens_file)
        api_frame = verify_api(manifest, token_path, args.api_workers)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        manifest.to_csv(temp_dir / "artifact_manifest.csv", index=False)
        genuine = manifest[manifest["stored_artifact_type"].eq("issue")].copy()
        genuine.to_csv(temp_dir / "genuine_issue_manifest.csv", index=False)
        eligibility.to_csv(temp_dir / "pr_eligibility.csv", index=False)
        eligibility[~eligibility["eligible"]].to_csv(
            temp_dir / "excluded_prs.csv", index=False
        )
        anomalies.to_csv(temp_dir / "audit_anomalies.csv", index=False)
        write_jsonl(temp_dir / "regeneration_sources.jsonl", regeneration)
        if args.with_api:
            api_frame.to_csv(temp_dir / "api_verification.csv", index=False)

        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "status": "complete",
            "api_verified": args.with_api,
            **observed_counts,
            "issue_only_prs": int(
                (
                    eligibility["eligible"]
                    & eligibility["pull_request_count"].eq(0)
                ).sum()
            ),
            "pull_only_prs": int(
                (
                    ~eligibility["eligible"]
                    & eligibility["pull_request_count"].gt(0)
                ).sum()
            ),
            "multi_artifact_prs": int(
                eligibility["referenced_artifacts"].gt(1).sum()
            ),
            "multi_genuine_issue_prs": int(
                eligibility["genuine_issue_count"].gt(1).sum()
            ),
            "empty_field_rows": {
                "genuine_issue_titles": int(
                    genuine_issues["stored_title_empty"].sum()
                ),
                "genuine_issue_bodies": int(
                    genuine_issues["stored_body_empty"].sum()
                ),
                "excluded_pull_titles": int(
                    manifest.loc[~genuine_issue_mask, "stored_title_empty"].sum()
                ),
                "excluded_pull_bodies": int(
                    manifest.loc[~genuine_issue_mask, "stored_body_empty"].sum()
                ),
            },
            "reference_evidence_counts": {
                str(key): int(value)
                for key, value in manifest["reference_evidence_type"]
                .value_counts()
                .sort_index()
                .items()
            },
            "stored_title_current_matches": (
                int(api_frame["stored_title_matches_current"].sum())
                if args.with_api
                else None
            ),
            "stored_body_current_matches": (
                int(api_frame["stored_body_matches_current"].sum())
                if args.with_api
                else None
            ),
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
            "requests": requests.__version__,
            "input_paths": {
                "source": str(source_path),
                "sampled": str(sampled_path),
            },
            "input_sha256": input_hashes,
            "source_checks": source_checks,
            "api_verified": args.with_api,
            "token_path_recorded": str(token_path) if token_path else None,
            "token_values_recorded": False,
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
            "api_verified": args.with_api,
            "input_sha256": input_hashes,
            "counts": observed_counts,
            "integrity_gates": {
                "input_hashes_match": True,
                "source_pr_fields_match_sampled": True,
                "aggregate_fields_losslessly_decomposed": True,
                "all_numbers_found_in_pr_bodies": True,
                "artifact_urls_match_repository_number_type": True,
                "retained_genuine_issue_title_body_nonempty": True,
                "no_fetched_artifact_equals_target_pr": True,
                "api_identity_and_type_match": args.with_api,
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

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"Audit outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
