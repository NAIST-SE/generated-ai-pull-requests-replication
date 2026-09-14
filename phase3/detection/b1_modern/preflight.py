#!/usr/bin/env python3
"""Verify the approved host, pinned environment, repositories, and model snapshots."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .adapters import verify_official_interfaces
from .common import (
    B1ValidationError,
    atomic_write_json,
    model_snapshot_paths,
    require_approved_host,
    runtime_metadata,
    utc_now,
    verify_official_repositories,
)
from .config import RUN_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase3/analysis/outputs/tier_b/b1-modern-detectors-v1"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_approved_host(require_cuda=True)
    repo_root = args.repo_root.resolve()
    runtime_root = args.runtime_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    commits = verify_official_repositories(runtime_root)
    interfaces = verify_official_interfaces(runtime_root)
    snapshots = model_snapshot_paths(runtime_root, local_files_only=True)
    runtime = runtime_metadata(repo_root)
    devices = runtime.get("cuda_devices", [])
    if not devices or max(device["total_memory"] for device in devices) < 44 * 1024**3:
        raise B1ValidationError(
            "B1 requires at least one GPU with approximately 48 GB VRAM"
        )
    disk = shutil.disk_usage(runtime_root)
    report = {
        "run_id": RUN_ID,
        "status": "passed",
        "checked_at_utc": utc_now(),
        "runtime_root": str(runtime_root),
        "runtime_free_gib": disk.free / 1024**3,
        "official_repository_commits": commits,
        "official_interfaces": interfaces,
        "model_snapshot_paths": {name: str(path) for name, path in snapshots.items()},
        "runtime": runtime,
    }
    report_path = output_root / "environment" / "preflight.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
