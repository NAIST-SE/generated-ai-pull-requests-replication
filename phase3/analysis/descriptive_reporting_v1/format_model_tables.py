#!/usr/bin/env python3
"""Standardize model names/order in publication TeX; never compute statistics.

Read the current manuscript and baseline table, write a separate presentation
copy, and leave frozen analysis outputs untouched. Full model identifiers in
the manuscript prose are preserved.
"""
import argparse
import re
from pathlib import Path

ORDER = ("Qwen2.5-72B", "GPT-5-mini", "Llama-3.1-70B")
ALIASES = {
    r"\openSourceLLMTwo": ORDER[0],
    r"\proprietaryLLM": ORDER[1],
    r"\openSourceLLMOne": ORDER[2],
}
LABELS = ("tab:rq1_f1_range", "tab:rq2_feature_effects", "tab:rq3_retrieval")


def baseline_table(source):
    prefix, body = source.split("\\midrule\n", 1)
    body, suffix = body.split(r"\bottomrule", 1)
    groups = body.split("\\midrule\n")
    assert len(groups) == 3, "Expected three baseline model groups"
    by_model = {group.split(" & ", 1)[0]: group for group in groups}
    assert set(by_model) == set(ORDER)
    assert all(group.count(" & ") == 15 for group in groups)
    return prefix + "\\midrule\n" + "\\midrule\n".join(
        by_model[model] for model in ORDER
    ) + r"\bottomrule" + suffix


def manuscript(source):
    found = []

    def format_table(match):
        table = match.group()
        label = next((key for key in LABELS if r"\label{" + key + "}" in table), None)
        if label is None:
            return table
        found.append(label)
        for old, new in ALIASES.items():
            table = table.replace(old, new)
        lines = table.splitlines(keepends=True)
        if label == "tab:rq1_f1_range":
            for index, line in enumerate(lines):
                if line.startswith(r"\textbf{Detector} & "):
                    lines[index] = r"\textbf{Detector} & " + " & ".join(
                        r"\textbf{" + model + "}" for model in ORDER
                    ) + " \\\\\n"
        if label == "tab:rq3_retrieval":
            groups = 0
            for index, line in enumerate(lines):
                if line.startswith(r"\multirow{3}{*}"):
                    rows = lines[index + 1:index + 4]
                    by_model = {row.split("&")[1].strip(): row for row in rows}
                    assert set(by_model) == set(ORDER)
                    lines[index + 1:index + 4] = [by_model[model] for model in ORDER]
                    groups += 1
            assert groups == 2, "Expected two RQ3 encoders"
        return "".join(lines)

    result = re.sub(r"\\begin\{table\}(?:\[[^\]]*\])?.*?\\end\{table\}",
                    format_table, source, flags=re.S)
    assert sorted(found) == sorted(LABELS), "Expected all three main-paper tables"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.manuscript_dir.resolve()
    target = args.output_dir.resolve()
    assert source != target, "Write presentation copies to a separate directory"
    results = {
        "main.tex": manuscript((source / "main.tex").read_text()),
        "tables/prelim_detection_intervals.tex": baseline_table(
            (source / "tables/prelim_detection_intervals.tex").read_text()),
    }
    for relative, content in results.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)


if __name__ == "__main__":
    main()
