#!/usr/bin/env python3
"""Create an immutable, audited run-3 derivative for the deep-review analyses.

SERVER ONLY. This script reads the three frozen generation CSVs and the frozen
detection-results CSV. It never modifies those files. A generation is successful only
when `finish_reason == "stop"` and the generated payload is non-empty and does not carry
a known error marker. Detection rows are retained only when their generation key belongs
to that successful set; all human rows are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from server_guard import require_nokogiri


EXPECTED = {
    "planned_generation_rows": 5544,
    "successful_generation_rows": 5526,
    "excluded_generation_rows": 18,
    "frozen_detection_ai_rows": 5528,
    "excluded_rows_present_in_detection": 2,
    "cleaned_detection_ai_rows": 5526,
    "cleaned_detection_human_rows": 308,
}

KEY_COLUMNS = ["pr_id", "variation", "model"]
GENERATION_REQUIRED = {
    "pr_id",
    "variation",
    "model",
    "generated_text",
    "finish_reason",
}
DETECTION_REQUIRED = {"pr_id", "variation", "model", "entry_type", "text"}
ERROR_MARKERS = (
    "ERROR_GENERATION_FAILED",
    "ChatCompletion(id=None",
    "Requested input length",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in KEY_COLUMNS:
        result[column] = result[column].astype(str).str.strip()
    return result


def discover_generation_csvs(run_dir: Path, detection_csv: Path) -> list[Path]:
    selected: list[Path] = []
    for candidate in sorted(run_dir.glob("*.csv")):
        if candidate.resolve() == detection_csv.resolve():
            continue
        header = set(pd.read_csv(candidate, nrows=0).columns)
        if GENERATION_REQUIRED.issubset(header):
            selected.append(candidate)
    if len(selected) != 3:
        names = ", ".join(path.name for path in selected) or "none"
        raise RuntimeError(f"Expected three generation CSVs; discovered {len(selected)}: {names}")
    return selected


def failure_reason(row: pd.Series) -> str:
    text = "" if pd.isna(row["generated_text"]) else str(row["generated_text"])
    finish_reason = "" if pd.isna(row["finish_reason"]) else str(row["finish_reason"]).strip()
    reasons: list[str] = []
    if finish_reason != "stop":
        reasons.append(f"finish_reason={finish_reason or '<empty>'}")
    if not text.strip():
        reasons.append("empty_generated_text")
    for marker in ERROR_MARKERS:
        if marker in text:
            reasons.append(f"marker={marker}")
    return ";".join(reasons)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("phase3/datasets/temperature_1/run_3"),
    )
    parser.add_argument(
        "--detection-csv",
        type=Path,
        default=Path("phase3/datasets/temperature_1/run_3/detection_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase3/datasets/temperature_1/run_3_cleaned"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing cleaned derivative; raw run_3 files are never touched.",
    )
    return parser.parse_args()


def main() -> None:
    require_nokogiri()
    args = parse_args()
    run_dir = args.run_dir.resolve()
    detection_csv = args.detection_csv.resolve()
    output_dir = args.output_dir.resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not detection_csv.is_file():
        raise FileNotFoundError(detection_csv)
    if output_dir == run_dir or run_dir in output_dir.parents and output_dir.name == run_dir.name:
        raise RuntimeError("Output directory must be distinct from frozen run_3")
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; pass --overwrite explicitly")

    generation_csvs = discover_generation_csvs(run_dir, detection_csv)
    source_hashes_before = {str(path): sha256(path) for path in [*generation_csvs, detection_csv]}

    generation_parts = []
    for path in generation_csvs:
        part = pd.read_csv(path)
        part["source_file"] = path.name
        generation_parts.append(part)
    generations = normalized_keys(pd.concat(generation_parts, ignore_index=True))
    detections = normalized_keys(pd.read_csv(detection_csv))

    missing_detection = DETECTION_REQUIRED.difference(detections.columns)
    if missing_detection:
        raise RuntimeError(f"Detection CSV missing columns: {sorted(missing_detection)}")
    if generations.duplicated(KEY_COLUMNS).any():
        duplicate_count = int(generations.duplicated(KEY_COLUMNS, keep=False).sum())
        raise RuntimeError(f"Generation data contain {duplicate_count} rows with duplicate keys")

    generations["failure_reason"] = generations.apply(failure_reason, axis=1)
    generations["generation_success"] = generations["failure_reason"].eq("")
    successful = generations[generations["generation_success"]].copy()
    excluded = generations[~generations["generation_success"]].copy()

    successful_index = pd.MultiIndex.from_frame(successful[KEY_COLUMNS])
    ai_mask = detections["entry_type"].astype(str).str.lower().eq("ai")
    human_mask = detections["entry_type"].astype(str).str.lower().eq("human")
    detection_ai_index = pd.MultiIndex.from_frame(detections.loc[ai_mask, KEY_COLUMNS])
    keep_ai = detection_ai_index.isin(successful_index)

    cleaned_ai = detections.loc[ai_mask].loc[keep_ai].copy()
    cleaned_human = detections.loc[human_mask].copy()
    cleaned = pd.concat([cleaned_ai, cleaned_human], ignore_index=True)

    detection_ai_keys = set(map(tuple, detections.loc[ai_mask, KEY_COLUMNS].to_numpy()))
    excluded["included_in_frozen_detection"] = [
        tuple(values) in detection_ai_keys for values in excluded[KEY_COLUMNS].to_numpy()
    ]

    observed = {
        "planned_generation_rows": int(len(generations)),
        "successful_generation_rows": int(len(successful)),
        "excluded_generation_rows": int(len(excluded)),
        "frozen_detection_ai_rows": int(ai_mask.sum()),
        "excluded_rows_present_in_detection": int(excluded["included_in_frozen_detection"].sum()),
        "cleaned_detection_ai_rows": int(len(cleaned_ai)),
        "cleaned_detection_human_rows": int(len(cleaned_human)),
    }
    mismatches = {
        key: {"expected": EXPECTED[key], "observed": value}
        for key, value in observed.items()
        if value != EXPECTED[key]
    }
    if mismatches:
        raise RuntimeError(f"Integrity gate failed: {json.dumps(mismatches, indent=2)}")

    source_hashes_after = {str(path): sha256(path) for path in [*generation_csvs, detection_csv]}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("A frozen source file changed while the derivative was being prepared")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="run_3_cleaned-", dir=output_dir.parent))
    try:
        cleaned.to_csv(temp_dir / "detection_results.csv", index=False)
        excluded.to_csv(temp_dir / "generation_exclusions.csv", index=False)
        summary = {
            "expected": EXPECTED,
            "observed": observed,
            "key_columns": KEY_COLUMNS,
            "success_rule": (
                "finish_reason == 'stop', generated_text non-empty, and no known error marker"
            ),
            "source_sha256": source_hashes_after,
            "output_sha256": {
                "detection_results.csv": sha256(temp_dir / "detection_results.csv"),
                "generation_exclusions.csv": sha256(temp_dir / "generation_exclusions.csv"),
            },
        }
        (temp_dir / "integrity_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(json.dumps(observed, indent=2, sort_keys=True))
    print(f"Wrote cleaned derivative to {output_dir}")


if __name__ == "__main__":
    main()
