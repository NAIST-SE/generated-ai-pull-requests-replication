#!/usr/bin/env python3
"""Format frozen ROUGE-L/BERTScore medians as panel (b) of manuscript Table 4.

Uses only verified summary CSVs, never description text or metric computation.
Writes a TeX fragment and provenance into a separate presentation directory.
"""
import argparse
import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

SOURCES = {
    "canonical_rouge_summary.csv": "416f2eec5f39314a3fd86d96c0bbdf06498cd7f2accdb33ecf559eb1c82972fb",
    "canonical_bertscore_summary.csv": "bd7c6ec908926f5a710ebfb68b6316cc9468db470d2893346ed6168c9861e4f6",
}
MODELS = (
    ("Qwen/Qwen2.5-72B-Instruct", "Qwen2.5-72B"),
    ("gpt-5-mini-2025-08-07", "GPT-5-mini"),
    ("meta-llama/llama-3.1-70b-instruct", "Llama-3.1-70B"),
)
CONTEXTS = tuple(f"P-{i}" for i in range(1, 7))


def load(path):
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != SOURCES[path.name]:
        raise ValueError(f"Frozen source hash mismatch: {path.name}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(model, code) for model, _ in MODELS for code in CONTEXTS}
    result = {(row["model"], row["variation"]): row for row in rows}
    if len(rows) != 18 or set(result) != expected:
        raise ValueError(f"Expected 18 unique model/context cells: {path.name}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rouge = load(args.source_dir / "canonical_rouge_summary.csv")
    bert = load(args.source_dir / "canonical_bertscore_summary.csv")
    lines = [
        "% Presentation-only medians from hash-pinned verification CSVs.",
        r"\newcommand{\rqTwoSimilarityMedians}{%",
        r"\midrule",
        r"\multicolumn{7}{l}{\textbf{(b) Descriptive reference-similarity medians}} \\",
    ]
    cells = []
    for metric, title, data, column in [
        ("ROUGE-L F1", "ROUGE-L F1", rouge, "rougeL_median"),
        ("BERTScore F1", "BERTScore F1 (baseline-rescaled)", bert, "median"),
    ]:
        lines.append(r"\addlinespace[3pt]")
        lines.append(r"\multicolumn{7}{l}{\textit{" + title + r"}} \\")
        for model, display in MODELS:
            values = [Decimal(data[(model, code)][column]) for code in CONTEXTS]
            if not all(value.is_finite() for value in values):
                raise ValueError("Non-finite median")
            highest = max(values)
            formatted = []
            for code, value in zip(CONTEXTS, values):
                n = int(data[(model, code)]["n"])
                expected_n = (273 if code in ("P-5", "P-6") else
                              302 if code == "P-4" and model != MODELS[2][0] else 308)
                if n != expected_n or rouge[(model, code)]["n"] != bert[(model, code)]["n"]:
                    raise ValueError(f"Population mismatch: {model}, {code}")
                text = f"{value:.3f}"
                formatted.append(r"\textbf{" + text + "}" if value == highest else text)
                cells.append({"metric": metric, "model": model, "context_code": code,
                              "n": n, "source_median": str(value), "display": text,
                              "row_maximum": value == highest})
            lines.append(" & ".join([display] + formatted) + r" \\")
    lines.append("}")
    result = "\n".join(lines) + "\n"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "rq2_similarity_medians.tex"
    destination.write_text(result)
    provenance = {"source_sha256": SOURCES, "cells": cells,
                  "formatter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "tex_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                  "statistics_recomputed": False}
    (args.output_dir / "rq2_similarity_medians.provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
