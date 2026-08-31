from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse


JOIN_DELIMITER = " \n\n "


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_issue_numbers(value: str) -> list[int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid issue-number list: {value!r}")
    numbers = [int(part) for part in parts]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"Duplicate issue number: {value!r}")
    return numbers


def parse_aggregate(value: str, numbers: list[int], field: str) -> list[str]:
    """Reverse the fetcher's `JOIN_DELIMITER.join('issue #N: value')` format."""
    text = str(value)
    parsed: list[str] = []
    cursor = 0
    for index, number in enumerate(numbers):
        prefix = f"issue #{number}: "
        if not text.startswith(prefix, cursor):
            raise ValueError(
                f"{field}: expected {prefix!r} at offset {cursor}, "
                f"found {text[cursor:cursor + len(prefix)]!r}"
            )
        start = cursor + len(prefix)
        if index + 1 == len(numbers):
            parsed.append(text[start:])
            cursor = len(text)
            continue

        next_prefix = f"issue #{numbers[index + 1]}: "
        boundary = JOIN_DELIMITER + next_prefix
        end = text.find(boundary, start)
        if end < 0:
            raise ValueError(
                f"{field}: missing boundary before issue #{numbers[index + 1]}"
            )
        if text.find(boundary, end + 1) >= 0:
            raise ValueError(
                f"{field}: ambiguous repeated boundary before issue "
                f"#{numbers[index + 1]}"
            )
        parsed.append(text[start:end])
        cursor = end + len(JOIN_DELIMITER)

    reconstructed = JOIN_DELIMITER.join(
        f"issue #{number}: {item}" for number, item in zip(numbers, parsed)
    )
    if reconstructed != text:
        raise ValueError(f"{field}: lossless reconstruction failed")
    return parsed


def artifact_type_from_url(url: str, repository: str, number: int) -> str:
    parsed = urlparse(url.strip())
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc.lower() != "github.com" or len(parts) != 4:
        return "unknown"
    url_repository = "/".join(parts[:2])
    if url_repository.lower() != repository.lower() or parts[3] != str(number):
        return "unknown"
    if parts[2] == "issues":
        return "issue"
    if parts[2] == "pull":
        return "pull_request"
    return "unknown"


def reference_evidence(
    body: str,
    repository: str,
    number: int,
    repository_aliases: tuple[str, ...] = (),
) -> dict[str, object]:
    repositories = tuple(
        dict.fromkeys(
            candidate.lower()
            for candidate in (repository, *repository_aliases)
            if candidate.strip()
        )
    )
    repository_pattern = "(?:" + "|".join(
        re.escape(candidate) for candidate in repositories
    ) + ")"
    patterns = [
        (
            "closing_keyword",
            re.compile(
                rf"(?i)\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+"
                rf"(?:issue\s+)?#\s*{number}\b"
            ),
        ),
        (
            "explicit_issue_url",
            re.compile(
                rf"(?i)https://github\.com/{repository_pattern}/issues/{number}\b"
            ),
        ),
        (
            "explicit_pull_url",
            re.compile(
                rf"(?i)https://github\.com/{repository_pattern}/pull/{number}\b"
            ),
        ),
        (
            "number_reference",
            re.compile(rf"(?i)(?<![\w#])#\s*{number}\b"),
        ),
        (
            "repository_reference",
            re.compile(rf"(?i)(?:JabRef\s+issue\s+|jabref[/#]){number}\b"),
        ),
    ]
    all_matches = list(re.finditer(rf"(?i)#\s*{number}\b", body))
    for evidence_type, pattern in patterns:
        match = pattern.search(body)
        if match:
            start = max(0, match.start() - 70)
            end = min(len(body), match.end() + 70)
            excerpt = re.sub(r"\s+", " ", body[start:end]).strip()
            return {
                "reference_found": True,
                "reference_evidence_type": evidence_type,
                "reference_evidence_excerpt": excerpt,
                "reference_occurrences": max(1, len(all_matches)),
            }
    return {
        "reference_found": False,
        "reference_evidence_type": "missing",
        "reference_evidence_excerpt": "",
        "reference_occurrences": 0,
    }


def prompt_block(issues: list[dict[str, str]]) -> str:
    return "\n---\n".join(
        f"Title: {issue['stored_title']}\nBody: {issue['stored_body']}"
        for issue in issues
    )
