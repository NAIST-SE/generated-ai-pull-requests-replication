#!/usr/bin/env python3
"""Final integrity verification and completion manifest."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    EXPECTED,
    PROTOCOL,
    output_dir,
    project_root,
    require_nokogiri,
    sha256_file,
    utc_now,
    write_json,
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def check_counts(frame: pd.DataFrame, versions: bool = True) -> None:
    group_columns = ["model", "variation"] + (["version"] if versions else [])
    observed = frame.groupby(group_columns).size().to_dict()
    for (model, variation), expected in EXPECTED.items():
        if versions:
            for version in ("historical", "current"):
                value = int(observed.get((model, variation, version), 0))
                if value != expected:
                    raise RuntimeError(
                        f"{model} {variation} {version}: expected {expected}, observed {value}"
                    )
        else:
            value = int(observed.get((model, variation), 0))
            if value != expected:
                raise RuntimeError(
                    f"{model} {variation}: expected {expected}, observed {value}"
                )


def main() -> None:
    require_nokogiri()
    root = project_root(Path(__file__))
    out = output_dir(root)
    metadata_names = [
        "main_metrics_metadata.json",
        "py312_metrics_metadata.json",
        "bertscore_metadata.json",
        "detector_metadata.json",
        "analysis_metadata.json",
    ]
    metadata = {name: load_json(out / name) for name in metadata_names}
    for name, payload in metadata.items():
        if payload.get("protocol") != PROTOCOL:
            raise RuntimeError(f"Protocol mismatch in {name}")
        if payload.get("raw_text_in_outputs") is not False:
            raise RuntimeError(f"Raw-text output guard missing in {name}")

    main_metrics = pd.read_csv(out / "main_metrics.csv", dtype={"pr_id": str})
    py312 = pd.read_csv(out / "py312_metrics.csv", dtype={"pr_id": str})
    bertscore = pd.read_csv(out / "bertscore_metrics.csv", dtype={"pr_id": str})
    detector = pd.read_csv(out / "detector_scores.csv", dtype={"pr_id": str})
    human_detector = pd.read_csv(out / "human_detector_scores.csv", dtype={"pr_id": str})
    manifest = pd.read_csv(out / "pair_manifest.csv", dtype={"pr_id": str})

    for frame, name in [
        (main_metrics, "main_metrics"),
        (py312, "py312_metrics"),
        (bertscore, "bertscore_metrics"),
    ]:
        if len(frame) != 778:
            raise RuntimeError(f"{name}: expected 778 rows, observed {len(frame)}")
        check_counts(frame)
        if frame.duplicated(["model", "variation", "pr_id", "version"]).any():
            raise RuntimeError(f"{name}: duplicate keys")
    if len(manifest) != 389:
        raise RuntimeError(f"Pair manifest expected 389 rows, observed {len(manifest)}")
    check_counts(manifest, versions=False)
    if len(detector) != 1556 or detector["detector"].nunique() != 2:
        raise RuntimeError("Detector score shape differs from 389 pairs x 2 versions x 2 detectors")
    for _, group in detector.groupby("detector"):
        check_counts(group)
    if len(human_detector) != 100:
        raise RuntimeError("Human detector file expected 50 PRs x 2 detectors")

    forbidden_exact = {"text", "prompt", "full_prompt", "generated_text", "human_text", "title"}
    for path in out.glob("*.csv"):
        columns = set(pd.read_csv(path, nrows=0).columns)
        leaked = sorted(columns & forbidden_exact)
        if leaked:
            raise RuntimeError(f"{path.name} exposes forbidden raw columns: {leaked}")

    validation_payloads = [
        metadata["main_metrics_metadata.json"].get("historical_validations", []),
        metadata["py312_metrics_metadata.json"].get("historical_validations", []),
        [metadata["bertscore_metadata.json"].get("historical_validation", {})],
    ]
    validations = [item for group in validation_payloads for item in group]
    if not validations or any("max_abs_difference" not in item for item in validations):
        raise RuntimeError("Historical-output validations are incomplete")

    required_outputs = sorted(
        path
        for path in out.iterdir()
        if path.is_file()
        and path.name != "completion.json"
        and path.suffix not in {".log", ".exit"}
    )
    audit_sidecars = sorted(
        path.name
        for path in out.iterdir()
        if path.is_file() and path.suffix in {".log", ".exit"}
    )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    completion = {
        "protocol": PROTOCOL,
        "status": "complete_and_verified",
        "completed_at_utc": utc_now(),
        "hostname": platform.node(),
        "git_commit": git_commit,
        "pair_count": len(manifest),
        "versioned_metric_rows": len(main_metrics),
        "versioned_detector_rows": len(detector),
        "historical_validation_checks": len(validations),
        "maximum_historical_validation_difference": max(
            float(item["max_abs_difference"]) for item in validations
        ),
        "raw_text_in_outputs": False,
        "manuscript_modified": False,
        "audit_sidecars_excluded_from_artifact_hashes": audit_sidecars,
        "output_sha256": {
            path.name: sha256_file(path) for path in required_outputs
        },
    }
    write_json(out / "completion.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
