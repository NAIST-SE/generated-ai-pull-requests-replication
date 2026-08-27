"""Execution guard retained from the audited study workflow.

The study team ran result-bearing analyses only on ``nokogiri``. External users of the
public replication package may opt in on another host by setting
``LLM_PRS_ALLOW_REPLICATION_RUN=1``. The opt-in is intentionally explicit so the study
team cannot accidentally overwrite verified outputs on a local workstation.
"""

from __future__ import annotations

import os
import socket


def require_nokogiri() -> None:
    """Require the study server or an explicit external-replication opt-in."""
    host = socket.gethostname().split(".", 1)[0]
    external_opt_in = os.environ.get("LLM_PRS_ALLOW_REPLICATION_RUN") == "1"
    if host != "nokogiri" and not external_opt_in:
        raise RuntimeError(
            "Study-team analyses are server-only. External reproducers must set "
            f"LLM_PRS_ALLOW_REPLICATION_RUN=1 explicitly (current host: {host!r})."
        )
