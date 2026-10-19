#!/usr/bin/env python3
"""Combine verified TeX cells for publication; no metrics/statistics are computed.

Keep the original server reporting outputs immutable. This presentation-only
formatter copies FKGL from the structural table into the lexical table, retaining
every included H/A median, quartile, and complete-pair count verbatim.
"""
import argparse
import hashlib
import json
from pathlib import Path

INPUT_HASHES = {
    "rq2_structural_descriptives.tex": "a9264cd426a54340f583500f9bba949311e01b4f879826c38c2126db0f2e416b",
    "rq2_lexical_descriptives.tex": "0a9a21cea8f582345ec00f39d3c56c246db1b7c0194a7c2761ddddb5a04e4978",
}
OUTPUT = "rq2_feature_descriptives.tex"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source):
    for name, expected in INPUT_HASHES.items():
        assert sha(source / name) == expected, f"Changed source: {name}"
    structural = (source / "rq2_structural_descriptives.tex").read_text()
    lexical = (source / "rq2_lexical_descriptives.tex").read_text()
    structural_rows = [line for line in structural.splitlines() if r"\shortstack[l]{H:" in line]
    lexical_rows = [line for line in lexical.splitlines() if r"\shortstack[l]{H:" in line]
    assert len(structural_rows) == len(lexical_rows) == 18
    for structure, lexicon in zip(structural_rows, lexical_rows):
        a = structure.removesuffix(r" \\").split(" & ")
        b = lexicon.removesuffix(r" \\").split(" & ")
        assert len(a) == 6 and len(b) == 7
        assert a[:3] == b[:3], "Model/context/pair count mismatch"
        combined = " & ".join(b[:4] + [a[5]] + b[4:]) + r" \\"
        lexical = lexical.replace(lexicon, combined, 1)
    lines = lexical.splitlines()
    lines[0] = "% Presentation-only combination of verified TeX cells; do not hand-edit."
    for i, line in enumerate(lines):
        if line.startswith(r"\caption{"):
            lines[i] = (r"\caption{Readability and lexical values by model and context (RQ2): median [Q1, Q3]. "
                        r"H is the human baseline matched to that model--context cell; A is AI. "
                        r"The same $n$ applies to H and A; the main $n$ reflects eligibility or unavailable generations. "
                        r"FKGL is the Flesch--Kincaid grade level; TTR and lexical density are proportions. "
                        r"Bidirectional MTLD uses complete pairs only ($n_{\mathrm{MTLD}}$).}")
        elif line.startswith(r"\label{"):
            lines[i] = r"\label{tab:rq2_feature_descriptives}"
        elif line == r"\setlength{\tabcolsep}{4pt}":
            lines[i] = r"\setlength{\tabcolsep}{3pt}"
        elif line == r"\begin{tabular}{llrrlll}":
            lines[i] = r"\begin{tabular}{llrrllll}"
        elif line.startswith(r"\textbf{Model}"):
            lines[i] = line.replace(r" & \textbf{TTR}", r" & \textbf{FKGL} & \textbf{TTR}")
    result = "\n".join(lines) + "\n"
    assert result.count(r"\shortstack[l]{H:") == 72
    assert r"\textbf{Words}" not in result and r"\textbf{Sentences}" not in result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = render(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / OUTPUT).write_text(result)
    manifest = {
        "scope": "Presentation only; no new metrics or statistical calculations",
        "input_sha256": INPUT_HASHES,
        "formatter_sha256": sha(Path(__file__)),
        "output_sha256": {OUTPUT: sha(args.output_dir / OUTPUT)},
        "model_context_rows": 18,
        "unchanged_H_A_summary_cells": 72,
        "retained": ["FKGL", "TTR", "MTLD", "Lexical density"],
        "omitted_from_this_table_only": ["Word Count", "Sentence Count"],
        "paired_sample_sizes_copied_verbatim": True,
    }
    (args.output_dir / "presentation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
