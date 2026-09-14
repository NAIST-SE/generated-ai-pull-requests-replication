#!/usr/bin/env python3
"""Record the human-reviewed B1 execution gate without exposing study outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import B1ValidationError, atomic_write_json, sha256_file, utc_now
from .config import DETECTOR_NAMES, RUN_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=DETECTOR_NAMES, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase3/analysis/outputs/tier_b/b1-modern-detectors-v1"),
    )
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--infrastructure", required=True)
    parser.add_argument("--cost-ceiling", required=True)
    parser.add_argument("--configuration-approved", action="store_true")
    parser.add_argument("--licences-reviewed", action="store_true")
    parser.add_argument("--manual-reference-parity-confirmed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not all(
        (
            args.configuration_approved,
            args.licences_reviewed,
            args.manual_reference_parity_confirmed,
        )
    ):
        raise B1ValidationError(
            "Configuration, licences, and one-case reference parity must all be confirmed"
        )
    repo_root = args.repo_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    verification_path = output_root / "smoke" / args.detector / "verification.json"
    parity_path = output_root / "smoke" / args.detector / "reference_parity.json"
    if not verification_path.is_file():
        raise B1ValidationError("Smoke verification is missing")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "passed" or not verification.get("all_ten_scored"):
        raise B1ValidationError("Smoke verification has not passed")
    if not parity_path.is_file():
        raise B1ValidationError("Official-reference parity report is missing")
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if (
        parity.get("status") != "passed"
        or parity.get("detector") != args.detector
        or parity.get("detector_config_sha256")
        != verification.get("detector_config_sha256")
    ):
        raise B1ValidationError("Official-reference parity verification has not passed")
    gate = {
        "run_id": RUN_ID,
        "detector": args.detector,
        "status": "approved",
        "approved_at_utc": utc_now(),
        "approved_by": args.approved_by,
        "infrastructure": args.infrastructure,
        "cost_ceiling": args.cost_ceiling,
        "configuration_approved": True,
        "licences_reviewed": True,
        "manual_reference_parity_confirmed": True,
        "detector_config_sha256": verification["detector_config_sha256"],
        "smoke_verification_sha256": sha256_file(verification_path),
        "reference_parity_sha256": sha256_file(parity_path),
        "estimated_full_scoring_hours": verification["estimated_full_scoring_hours"],
    }
    gate_path = output_root / "execution_gates" / f"{args.detector}.json"
    if gate_path.exists():
        raise B1ValidationError(
            f"Refusing to overwrite existing execution gate: {gate_path}"
        )
    atomic_write_json(gate_path, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
