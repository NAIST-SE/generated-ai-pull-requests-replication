"""Fail-closed loading and rendering for the corrected P5/P6 issue context."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


PROTOCOL_VERSION = "b4-issue-source-audit-v1"
ISSUE_CONTEXT_VARIATIONS = frozenset({"P-5", "P-6"})
EXPECTED_MANIFEST_SHA256 = (
    "bdeb1d70b03c46d9f54e8775bb816d71183d486a5e55d9f76f992ab7938b49dc"
)
EXPECTED_PR_COUNT = 273
EXPECTED_ISSUE_COUNT = 293
DEFAULT_MANIFEST_RELATIVE_PATH = Path(
    "analysis/outputs/tier_b/b4-issue-source-audit-v1/regeneration_sources.jsonl"
)


class IssueSourceError(RuntimeError):
    """Raised when issue-source provenance or prompt integrity is not exact."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_issue_prompt_block(issues: Iterable[Mapping[str, Any]]) -> str:
    """Render the exact text placed between ``<issues>`` tags."""
    rendered = []
    for index, issue in enumerate(issues, start=1):
        title = issue.get("title")
        body = issue.get("body")
        if not isinstance(title, str) or not title.strip():
            raise IssueSourceError(f"Issue {index} has an empty or non-string title")
        if not isinstance(body, str) or not body.strip():
            raise IssueSourceError(f"Issue {index} has an empty or non-string body")
        rendered.append(f"Title: {title}\nBody: {body}")
    if not rendered:
        raise IssueSourceError("No genuine issue source is attached")
    return "\n---\n".join(rendered)


def render_issue_section(pr_data: Mapping[str, Any]) -> str:
    """Validate a PR's audited source and return its complete ``<issues>`` section."""
    issues = pr_data.get("issues")
    source = pr_data.get("issue_source")
    info = pr_data.get("info")
    if not isinstance(issues, list) or not isinstance(source, Mapping):
        raise IssueSourceError("P5/P6 requires an attached audited issue-source record")
    if not isinstance(info, Mapping):
        raise IssueSourceError("PR metadata is missing")

    block = render_issue_prompt_block(issues)
    expected_block = source.get("prompt_issue_block")
    expected_hash = source.get("prompt_issue_block_sha256")
    if block != expected_block:
        raise IssueSourceError("Rendered issue block differs from the frozen manifest")
    if sha256_text(block) != expected_hash:
        raise IssueSourceError("Rendered issue-block hash differs from the frozen manifest")

    target_title = info.get("title")
    target_body = info.get("body")
    if isinstance(target_title, str) and isinstance(target_body, str):
        target_block = f"Title: {target_title}\nBody: {target_body}"
        if block == target_block:
            raise IssueSourceError("Issue block equals the target PR title/body")

    return f"<issues>\n{block}\n</issues>"


def is_issue_context(variation: str) -> bool:
    return variation in ISSUE_CONTEXT_VARIATIONS


def has_audited_issue_source(pr_data: Mapping[str, Any]) -> bool:
    return bool(pr_data.get("issues")) and isinstance(
        pr_data.get("issue_source"), Mapping
    )


def _normalise_issue(issue: Mapping[str, Any], pr_id: str) -> dict[str, Any]:
    required = {
        "reference_order",
        "issue_number",
        "url",
        "title",
        "body",
        "title_sha256",
        "body_sha256",
    }
    missing = sorted(required.difference(issue))
    if missing:
        raise IssueSourceError(f"PR {pr_id}: issue record lacks {missing}")

    title = issue["title"]
    body = issue["body"]
    url = issue["url"]
    if not isinstance(title, str) or not title.strip():
        raise IssueSourceError(f"PR {pr_id}: retained issue title is empty")
    if not isinstance(body, str) or not body.strip():
        raise IssueSourceError(f"PR {pr_id}: retained issue body is empty")
    if not isinstance(url, str) or "/issues/" not in url or "/pull/" in url:
        raise IssueSourceError(f"PR {pr_id}: non-issue URL in source manifest")
    if sha256_text(title) != issue["title_sha256"]:
        raise IssueSourceError(f"PR {pr_id}: issue title hash mismatch")
    if sha256_text(body) != issue["body_sha256"]:
        raise IssueSourceError(f"PR {pr_id}: issue body hash mismatch")

    try:
        reference_order = int(issue["reference_order"])
        issue_number = int(issue["issue_number"])
    except (TypeError, ValueError) as exc:
        raise IssueSourceError(f"PR {pr_id}: invalid issue order/number") from exc
    if reference_order < 1 or issue_number < 1:
        raise IssueSourceError(f"PR {pr_id}: invalid issue order/number")

    return {
        **dict(issue),
        "reference_order": reference_order,
        "issue_number": issue_number,
    }


def validate_manifest_record(record: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "pr_id",
        "pull_number",
        "repository",
        "issues",
        "prompt_issue_block",
        "prompt_issue_block_sha256",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise IssueSourceError(f"Manifest record lacks {missing}")
    pr_id = record["pr_id"]
    if not isinstance(pr_id, str) or not pr_id:
        raise IssueSourceError("Manifest PR ID is empty or non-string")
    if not isinstance(record["issues"], list) or not record["issues"]:
        raise IssueSourceError(f"PR {pr_id}: issue list is empty")
    repository = record["repository"]
    if not isinstance(repository, str) or repository.count("/") != 1:
        raise IssueSourceError(f"PR {pr_id}: invalid repository identifier")

    issues = [_normalise_issue(issue, pr_id) for issue in record["issues"]]
    orders = [issue["reference_order"] for issue in issues]
    numbers = [issue["issue_number"] for issue in issues]
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        raise IssueSourceError(f"PR {pr_id}: issue reference order is not stable/unique")
    if len(numbers) != len(set(numbers)):
        raise IssueSourceError(f"PR {pr_id}: duplicate issue number")
    for issue in issues:
        parsed = urlparse(issue["url"])
        expected_path = f"/{repository}/issues/{issue['issue_number']}"
        if parsed.netloc.lower() != "github.com" or parsed.path.lower() != expected_path.lower():
            raise IssueSourceError(
                f"PR {pr_id}: issue URL differs from repository/number"
            )

    block = render_issue_prompt_block(issues)
    if block != record["prompt_issue_block"]:
        raise IssueSourceError(f"PR {pr_id}: prompt block reconstruction mismatch")
    if sha256_text(block) != record["prompt_issue_block_sha256"]:
        raise IssueSourceError(f"PR {pr_id}: prompt block hash mismatch")

    return {**dict(record), "issues": issues}


def load_issue_source_manifest(
    path: Path,
    *,
    expected_sha256: str = EXPECTED_MANIFEST_SHA256,
    expected_pr_count: int = EXPECTED_PR_COUNT,
    expected_issue_count: int = EXPECTED_ISSUE_COUNT,
) -> dict[str, dict[str, Any]]:
    """Load the frozen B4 source manifest and validate every record."""
    path = path.resolve()
    if not path.is_file():
        raise IssueSourceError(f"Audited issue-source manifest not found: {path}")
    observed_hash = sha256_file(path)
    if observed_hash != expected_sha256:
        raise IssueSourceError(
            f"Issue-source manifest hash mismatch: {observed_hash}"
        )

    records: dict[str, dict[str, Any]] = {}
    issue_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IssueSourceError(
                    f"Invalid JSON on manifest line {line_number}"
                ) from exc
            record = validate_manifest_record(raw)
            pr_id = record["pr_id"]
            if pr_id in records:
                raise IssueSourceError(f"Duplicate manifest PR ID: {pr_id}")
            records[pr_id] = record
            issue_count += len(record["issues"])

    if len(records) != expected_pr_count or issue_count != expected_issue_count:
        raise IssueSourceError(
            "Issue-source manifest count mismatch: "
            f"PRs={len(records)}, issues={issue_count}"
        )
    return records


def attach_issue_sources(
    pr_data: dict[str, dict[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
    *,
    manifest_path: Path,
    manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> None:
    """Attach typed issues to matching PRs and reject missing targets/collisions."""
    missing_pr_ids = sorted(set(manifest).difference(pr_data))
    if missing_pr_ids:
        raise IssueSourceError(
            f"Issue manifest contains {len(missing_pr_ids)} unknown PR IDs"
        )

    for record in pr_data.values():
        record["issues"] = []
        record["issue_source"] = None

    for pr_id, source in manifest.items():
        target = pr_data[pr_id]
        target["issues"] = [dict(issue) for issue in source["issues"]]
        target["issue_source"] = {
            "protocol_version": PROTOCOL_VERSION,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha256,
            "prompt_issue_block": source["prompt_issue_block"],
            "prompt_issue_block_sha256": source["prompt_issue_block_sha256"],
        }
        # Re-run the complete prompt-source gate after joining to target PR metadata.
        render_issue_section(target)
