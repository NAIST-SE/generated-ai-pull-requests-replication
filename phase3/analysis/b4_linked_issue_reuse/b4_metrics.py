"""Pure text/provenance helpers for the B4 linked-issue reuse analysis."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass


ISSUE_OPEN = "<issues>\n"
ISSUE_CLOSE = "\n</issues>"
WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^\)]+\)")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_issue_block(full_prompt: str) -> str:
    """Return the exact prompt text inside the one <issues> block, or fail."""
    if full_prompt.count(ISSUE_OPEN) != 1 or full_prompt.count(ISSUE_CLOSE) != 1:
        raise ValueError("Prompt must contain exactly one newline-delimited <issues> block")
    start = full_prompt.index(ISSUE_OPEN) + len(ISSUE_OPEN)
    end = full_prompt.index(ISSUE_CLOSE, start)
    block = full_prompt[start:end]
    if not block.strip():
        raise ValueError("Prompt issue block is empty")
    return block


@dataclass(frozen=True)
class IssueComponents:
    combined: str
    title: str
    body: str
    document_count: int
    parse_mode: str


def parse_issue_components(block: str) -> IssueComponents:
    """Split the generation code's stable Title/Body rendering without changing it."""
    pieces = re.split(r"\n---\n(?=(?:Title|Titles): )", block)
    titles: list[str] = []
    bodies: list[str] = []
    parse_mode = "title_body"
    for piece in pieces:
        if piece.startswith("Title: ") and "\nBody: " in piece:
            title, body = piece[len("Title: "):].split("\nBody: ", 1)
        elif piece.startswith("Titles: ") and "\nBodies: " in piece:
            parse_mode = "legacy_titles_bodies"
            title, body = piece[len("Titles: "):].split("\nBodies: ", 1)
        else:
            raise ValueError("Issue block does not match the frozen Title/Body rendering")
        titles.append(title)
        bodies.append(body)
    return IssueComponents(
        combined=block,
        title="\n---\n".join(titles),
        body="\n---\n".join(bodies),
        document_count=len(pieces),
        parse_mode=parse_mode,
    )


def represent(text: str, representation: str) -> str:
    if representation == "raw_case_sensitive":
        return unicodedata.normalize("NFC", text)
    if representation == "conservative_normalized":
        value = unicodedata.normalize("NFKC", text)
        value = MARKDOWN_LINK_PATTERN.sub(r"\1", value)
        value = " ".join(value.split())
        return value.lower()
    raise ValueError(f"Unknown representation: {representation}")


def tokenize(text: str, representation: str = "conservative_normalized") -> list[str]:
    return WORD_PATTERN.findall(represent(text, representation))


def ngram_containment(source: list[str], generated: list[str], n: int) -> float:
    """Occurrence-level generated n-gram windows also present in the source."""
    if n < 1:
        raise ValueError("n must be positive")
    total = len(generated) - n + 1
    if total <= 0 or len(source) < n:
        return 0.0
    source_ngrams = {tuple(source[index:index + n]) for index in range(len(source) - n + 1)}
    matches = sum(
        tuple(generated[index:index + n]) in source_ngrams
        for index in range(total)
    )
    return matches / total


def matched_token_coverage(source: list[str], generated: list[str], n: int = 5) -> float:
    if not generated or len(generated) < n or len(source) < n:
        return 0.0
    source_ngrams = {tuple(source[index:index + n]) for index in range(len(source) - n + 1)}
    covered = set()
    for index in range(len(generated) - n + 1):
        if tuple(generated[index:index + n]) in source_ngrams:
            covered.update(range(index, index + n))
    return len(covered) / len(generated)


@dataclass(frozen=True)
class SharedSpan:
    length: int
    source_start: int
    generated_start: int
    tokens: tuple[str, ...]


def longest_shared_span(source: list[str], generated: list[str]) -> SharedSpan:
    """Find one longest common contiguous token span using monotonic binary search."""
    if not source or not generated:
        return SharedSpan(0, -1, -1, ())

    def find(length: int) -> tuple[int, int] | None:
        if length == 0:
            return (0, 0)
        source_positions: dict[tuple[str, ...], int] = {}
        for index in range(len(source) - length + 1):
            source_positions.setdefault(tuple(source[index:index + length]), index)
        for index in range(len(generated) - length + 1):
            key = tuple(generated[index:index + length])
            if key in source_positions:
                return source_positions[key], index
        return None

    low, high = 0, min(len(source), len(generated))
    best = (0, 0, 0)
    while low <= high:
        middle = (low + high) // 2
        match = find(middle)
        if match is not None:
            best = (middle, match[0], match[1])
            low = middle + 1
        else:
            high = middle - 1
    length, source_start, generated_start = best
    return SharedSpan(
        length,
        source_start if length else -1,
        generated_start if length else -1,
        tuple(generated[generated_start:generated_start + length]) if length else (),
    )


def lexical_metrics(source_text: str, generated_text: str, representation: str) -> dict:
    source = tokenize(source_text, representation)
    generated = tokenize(generated_text, representation)
    span = longest_shared_span(source, generated)
    return {
        "source_tokens": len(source),
        "generated_tokens": len(generated),
        "shorter_than_5_tokens": len(generated) < 5,
        "ngram_3_containment": ngram_containment(source, generated, 3),
        "ngram_5_containment": ngram_containment(source, generated, 5),
        "ngram_8_containment": ngram_containment(source, generated, 8),
        "matched_5gram_token_coverage": matched_token_coverage(source, generated, 5),
        "longest_span_tokens": span.length,
        "longest_span_generated_fraction": span.length / len(generated) if generated else 0.0,
        "longest_span_source_start": span.source_start,
        "longest_span_generated_start": span.generated_start,
        "longest_span_text": " ".join(span.tokens),
    }
