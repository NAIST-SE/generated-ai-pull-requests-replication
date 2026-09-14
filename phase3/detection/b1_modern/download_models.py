#!/usr/bin/env python3
"""Download only the files needed for the four frozen B1 model snapshots."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .common import atomic_write_json, sha256_file, utc_now
from .config import MODEL_SPECS, RUN_ID


ALLOW_PATTERNS = [
    "*.json",
    "*.py",
    "*.safetensors",
    "*.model",
    "*.txt",
    "tokenizer.*",
    "special_tokens_map.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_root = args.runtime_root.resolve()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is required after the responsible account has accepted the Meta Llama 3 licence"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Install the B1 requirements first") from error

    snapshots = {}
    for name, spec in MODEL_SPECS.items():
        print(
            f"Downloading frozen snapshot {spec['model_id']}@{spec['revision']}",
            flush=True,
        )
        path = Path(
            snapshot_download(
                repo_id=spec["model_id"],
                revision=spec["revision"],
                cache_dir=runtime_root / "hf-cache",
                token=os.environ["HF_TOKEN"],
                allow_patterns=ALLOW_PATTERNS,
            )
        )
        index_files = sorted(path.glob("*.safetensors.index.json"))
        snapshots[name] = {
            **spec,
            "snapshot_path": str(path),
            "config_sha256": sha256_file(path / "config.json"),
            "weight_index_sha256": sha256_file(index_files[0])
            if index_files
            else "single-shard",
        }
    atomic_write_json(
        runtime_root / "model_snapshots.json",
        {"run_id": RUN_ID, "downloaded_at_utc": utc_now(), "snapshots": snapshots},
    )
    print(
        f"All four frozen model snapshots are available under {runtime_root / 'hf-cache'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
