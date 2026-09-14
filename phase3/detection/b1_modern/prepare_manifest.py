#!/usr/bin/env python3
"""Create the metadata-only B1 input and deterministic ten-case smoke manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (
    B1ValidationError,
    atomic_write_csv,
    atomic_write_json,
    load_verified_corpus,
    manifest_row,
    select_smoke_rows,
    sha256_file,
    stable_json_sha256,
    utc_now,
)
from .config import CORPUS_SHA256, MANIFEST_COLUMNS, RUN_ID, SMOKE_COUNT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase3/analysis/outputs/tier_b/b1-modern-detectors-v1"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    input_path = output_root / "input_manifest.csv"
    smoke_path = output_root / "smoke_manifest.csv"
    metadata_path = output_root / "preparation.json"
    existing = [path.exists() for path in (input_path, smoke_path, metadata_path)]
    if any(existing):
        if not all(existing):
            raise B1ValidationError("Refusing to overwrite a partial B1 preparation")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("run_id") != RUN_ID
            or metadata.get("corpus_sha256") != CORPUS_SHA256
            or sha256_file(input_path) != metadata.get("input_manifest_sha256")
            or sha256_file(smoke_path) != metadata.get("smoke_manifest_sha256")
        ):
            raise B1ValidationError(
                "Existing B1 preparation does not match the frozen run"
            )
        print(f"Existing verified B1 preparation retained at {output_root}")
        return 0

    corpus_rows, corpus_metadata = load_verified_corpus(repo_root)
    manifest_rows = [manifest_row(row) for row in corpus_rows]
    manifest_rows.sort(key=lambda row: row["scoring_order"])
    smoke_rows = select_smoke_rows(corpus_rows)

    atomic_write_csv(input_path, manifest_rows, MANIFEST_COLUMNS)
    atomic_write_csv(
        smoke_path,
        smoke_rows,
        (*MANIFEST_COLUMNS, "smoke_role", "contains_markdown_or_code"),
    )
    metadata = {
        "run_id": RUN_ID,
        "created_at_utc": utc_now(),
        "corpus_sha256": CORPUS_SHA256,
        "corpus_manifest_status": corpus_metadata["status"],
        "input_manifest_path": str(input_path),
        "input_manifest_sha256": sha256_file(input_path),
        "smoke_manifest_path": str(smoke_path),
        "smoke_manifest_sha256": sha256_file(smoke_path),
        "counts": {
            "input": len(manifest_rows),
            "smoke": len(smoke_rows),
            "human": sum(row["entry_type"] == "human" for row in manifest_rows),
            "ai": sum(row["entry_type"] == "ai" for row in manifest_rows),
        },
        "smoke_selection": (
            "deterministic metadata/text-structure coverage only: shortest and longest human; "
            "one human Markdown/code case; shortest and longest output from each generator; "
            "one additional AI Markdown/code case; no detector outcome used"
        ),
        "smoke_description_ids_sha256": stable_json_sha256(
            [row["description_id"] for row in smoke_rows]
        ),
    }
    if metadata["counts"]["smoke"] != SMOKE_COUNT:
        raise RuntimeError("Unexpected smoke count")
    atomic_write_json(metadata_path, metadata)
    print(
        f"Prepared {len(manifest_rows)} B1 rows and {len(smoke_rows)} smoke rows "
        f"under {output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
