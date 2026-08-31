"""Fail-closed guard for result-bearing deep-audit programs."""

from __future__ import annotations

import socket


def require_nokogiri() -> None:
    """Refuse to process study data anywhere except the authorised server."""
    host = socket.gethostname().split(".", 1)[0]
    if host != "nokogiri":
        raise RuntimeError(
            f"Study analyses are server-only: expected nokogiri, current host is {host!r}"
        )
