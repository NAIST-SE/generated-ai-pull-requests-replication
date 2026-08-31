#!/usr/bin/env python3
"""Fail-closed verifier and completion manifest for the bridge diagnostic."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import pandas as pd


PROTOCOL = "generation-session-six-context-bridge-v1"
EXPECTED_COMPLETE = {"gpt": 50, "qwen": 39}
EXPECTED_CONTEXTS = {f"P-{number}" for number in range(1, 7)}
EXPECTED_METRICS = {"rougeL_fmeasure", "bertscore_f1", "bm25_reciprocal_rank"}
EXPECTED_VERSIONS = {"historical_bridge", "current_bridge"}
FORBIDDEN_COLUMNS = {
    "text",
    "prompt",
    "full_prompt",
    "generated_text",
    "human_text",
    "title",
    "description",
    "body",
}


def project_root(script: Path) -> Path:
    for candidate in (script.resolve(), *script.resolve().parents):
        if (candidate / ".git").exists() and (candidate / "phase3").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    hostname = platform.node().lower()
    if "nokogiri" not in hostname:
        raise RuntimeError(f"Verification must run on Nokogiri, not {hostname!r}")
    root = project_root(Path(__file__))
    out = root / "phase3/analysis/outputs/tier_b/generation-session-six-context-bridge-v1"
    required = [
        "bridge_metric_rows.csv",
        "coverage.csv",
        "context_rankings.csv",
        "headline_stability.csv",
        "context_version_comparison.csv",
        "BRIDGE_REPORT.md",
        "analysis_metadata.json",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing bridge outputs: {missing}")

    metadata = json.loads((out / "analysis_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("protocol") != PROTOCOL:
        raise RuntimeError("Metadata protocol mismatch")
    guards = {
        "external_api_calls": False,
        "new_generation": False,
        "new_metric_scoring": False,
        "raw_text_in_outputs": False,
        "manuscript_modified": False,
    }
    for key, expected in guards.items():
        if metadata.get(key) is not expected:
            raise RuntimeError(f"Metadata guard {key!r} is not {expected!r}")

    bridge = pd.read_csv(out / "bridge_metric_rows.csv", dtype={"pr_id": str})
    rankings = pd.read_csv(out / "context_rankings.csv")
    headline = pd.read_csv(out / "headline_stability.csv")
    coverage = pd.read_csv(out / "coverage.csv")
    comparison = pd.read_csv(out / "context_version_comparison.csv")

    expected_rows = sum(EXPECTED_COMPLETE.values()) * 6 * 2 * 3
    if len(bridge) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} bridge rows, observed {len(bridge)}")
    if len(rankings) != 2 * 3 * 2 * 6:
        raise RuntimeError(f"Expected 72 ranking rows, observed {len(rankings)}")
    if len(headline) != 2 * 3:
        raise RuntimeError(f"Expected 6 headline rows, observed {len(headline)}")
    if len(comparison) != 2 * 3 * 6:
        raise RuntimeError(f"Expected 36 comparison rows, observed {len(comparison)}")
    if len(coverage) != 2:
        raise RuntimeError("Coverage must contain two model rows")

    if set(bridge["metric"]) != EXPECTED_METRICS:
        raise RuntimeError("Bridge metric set differs from protocol")
    if set(bridge["bridge_version"]) != EXPECTED_VERSIONS:
        raise RuntimeError("Bridge version set differs from protocol")
    if set(bridge["variation"]) != EXPECTED_CONTEXTS:
        raise RuntimeError("Bridge context set differs from P1--P6")
    if bridge.duplicated(["model_label", "pr_id", "variation", "bridge_version", "metric"]).any():
        raise RuntimeError("Bridge output contains duplicate keys")
    if bridge["value"].isna().any():
        raise RuntimeError("Bridge output contains missing metric values")

    observed_coverage = dict(zip(coverage["model_label"], coverage["complete_prs"].astype(int)))
    if observed_coverage != EXPECTED_COMPLETE:
        raise RuntimeError(f"Coverage differs: {observed_coverage}")
    for model_label, n in EXPECTED_COMPLETE.items():
        group = bridge[bridge["model_label"].eq(model_label)]
        counts = group.groupby(["metric", "bridge_version", "variation"])["pr_id"].nunique()
        if not counts.eq(n).all():
            raise RuntimeError(f"{model_label}: incomplete metric/context/version cells")

    p56 = bridge[bridge["variation"].isin(["P-5", "P-6"])]
    p56_wide = p56.pivot(
        index=["model_label", "metric", "pr_id", "variation"],
        columns="bridge_version",
        values="value",
    )
    if not (p56_wide["historical_bridge"] == p56_wide["current_bridge"]).all():
        raise RuntimeError("P5--P6 values are not fixed across bridge versions")

    for path in out.glob("*.csv"):
        columns = set(pd.read_csv(path, nrows=0).columns)
        leaked = FORBIDDEN_COLUMNS & columns
        if leaked:
            raise RuntimeError(f"{path.name} exposes forbidden raw columns: {sorted(leaked)}")

    current_top = bool(headline["current_p6_is_top"].all())
    if current_top != bool(metadata.get("p6_current_top_in_all_cells")):
        raise RuntimeError("Headline decision does not match metadata")
    expected_decision = (
        "does_not_trigger_complete_rerun"
        if current_top
        else "review_required_complete_rerun_triggered"
    )
    if metadata.get("decision_code") != expected_decision:
        raise RuntimeError("Decision code does not follow the predeclared rule")

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    stable_paths = [out / name for name in required]
    completion = {
        "protocol": PROTOCOL,
        "status": "complete_and_verified",
        "hostname": platform.node(),
        "git_commit": git_commit,
        "decision_code": expected_decision,
        "complete_prs": EXPECTED_COMPLETE,
        "bridge_metric_rows": len(bridge),
        "ranking_rows": len(rankings),
        "headline_cells": len(headline),
        "external_api_calls": False,
        "new_generation": False,
        "new_metric_scoring": False,
        "raw_text_in_outputs": False,
        "manuscript_modified": False,
        "output_sha256": {path.name: sha256_file(path) for path in stable_paths},
    }
    (out / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
