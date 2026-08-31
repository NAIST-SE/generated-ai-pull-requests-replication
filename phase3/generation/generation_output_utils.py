"""Shared completion policy for resumable generation outputs."""

from __future__ import annotations

from typing import Any, Mapping


FAILURE_MARKERS = ("ERROR_GENERATION_FAILED", "MOCKED DESCRIPTION")


def is_successful_generation_record(record: Mapping[str, Any]) -> bool:
    """Return whether a persisted row is safe to treat as complete on resume."""
    generated_text = str(record.get("generated_text", "")).strip()
    finish_reason = str(record.get("finish_reason", "")).strip().lower()
    if not generated_text or any(marker in generated_text.upper() for marker in FAILURE_MARKERS):
        return False
    return finish_reason == "stop"
