#!/usr/bin/env python3
"""Nokogiri-only layout refresh of Figure 7 from frozen BM25 scores.

Execute only the original plotting block, then rearrange its existing artists.
Do not compute retrieval scores, summaries, or significance tests. Refuse to
overwrite a previous run and verify that all data-space geometry is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INPUTS = {
    "phase3/analysis/deep_audit/analysis_rq2_bm25_retrieval.py":
        "e9e498ddb4b89f82a69dd1b4086cd7a10b99957de0d95025a4b9b20022628a1b",
    "phase3/analysis/outputs/corrected_rq23_v1/notebooks/outputs/bm25_retrieval/bm25_retrieval_scores.csv":
        "b3a2b2576741142032ebf04e637084c2f8dd3ce05f000cbb9a227d240f859c93",
}
CODES = [f"P-{i}" for i in range(1, 7)]
LEGACY = dict(zip(CODES, [
    "Repo Path", "PR Title", "File Diff", "File Contents", "Linked Issue",
    "Without File Contents",
]))
LABELS = ["Repo\nPath", "PR\nTitle", "Patch", "Full\nCode", "Linked\nIssue", "Multi-\nsource"]
EXPECTED_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct", "gpt-5-mini-2025-08-07",
    "meta-llama/llama-3.1-70b-instruct",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(axes):
    """Data-space coordinates only; exclude intended text/layout changes."""
    data = [{
        "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
        "xticks": np.asarray(ax.get_xticks()).tolist(),
        "yticks": np.asarray(ax.get_yticks()).tolist(),
        "lines": [np.asarray(line.get_xydata()).tolist() for line in ax.lines],
        "boxes": [patch.get_path().vertices.tolist() for patch in ax.patches],
        "collections": [np.asarray(c.get_offsets()).tolist() for c in ax.collections],
    } for ax in axes]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.study_root.resolve(), args.output_dir.resolve()
    sys.path.insert(0, str(root / "phase3/analysis/deep_audit"))
    from server_guard import require_nokogiri
    require_nokogiri()
    for relative, expected in INPUTS.items():
        if sha256(root / relative) != expected:
            raise RuntimeError(f"Frozen input hash mismatch: {relative}")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite previous output: {out}")

    source_key, scores_key = INPUTS
    scores = pd.read_csv(root / scores_key)
    if len(scores) != 5322 or sorted(scores["model"].unique()) != EXPECTED_MODELS:
        raise AssertionError("Unexpected score population or model order")
    if not scores["context"].equals(scores["variation"].map(LEGACY)):
        raise AssertionError("Unexpected frozen context mapping")
    if not scores["reciprocal_rank"].between(0, 1).all():
        raise AssertionError("Invalid reciprocal ranks")
    if scores.duplicated(["normalized_pr_id", "model", "variation"]).any():
        raise AssertionError("Duplicate frozen scores")
    counts = scores.groupby(["model", "variation"]).size()
    for model in EXPECTED_MODELS:
        for code in CODES:
            expected = 273 if code in {"P-5", "P-6"} else 308
            if code == "P-4" and model != EXPECTED_MODELS[-1]:
                expected = 302
            if int(counts.loc[model, code]) != expected:
                raise AssertionError(f"Unexpected count: {model}, {code}")

    # Only the frozen source's chart block; stop before layout/save or analysis IO.
    source = (root / source_key).read_text()
    start = source.index("    context_order = [VARIATION_TO_CONTEXT")
    end = source.index("    fig.tight_layout()", start)
    block = textwrap.dedent(source[start:end])
    plt.rcdefaults()
    env = {"plt": plt, "sns": sns, "scores": scores,
           "VARIATION_TO_CONTEXT": LEGACY, "CONTEXT_CODES": CODES}
    exec(compile(block, "frozen-bm25:plot-only", "exec"), env)
    fig, axes = env["fig"], list(env["axes"])
    fig.canvas.draw()
    reference = geometry(axes)

    # Reuse the same artists, scales, category order, and suppression of outliers.
    limits = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]
    ticks = [(ax.get_xticks(), ax.get_yticks()) for ax in axes]
    fig.set_size_inches(4.9, 7.6)
    grid = fig.add_gridspec(3, 1)
    for i, ax in enumerate(axes):
        ax.set_subplotspec(grid[i, 0])
        ax.set_xticks(ticks[i][0], LABELS)
        ax.set_yticks(ticks[i][1])
        ax.set_xlim(limits[i][0])
        ax.set_ylim(limits[i][1])
        ax.set_ylabel("Reciprocal rank", fontsize=14)
        ax.yaxis.label.set_visible(True)
        ax.set_xlabel("Context configuration" if i == 2 else "", fontsize=14)
        ax.tick_params(axis="x", labelrotation=0, labelsize=12, labelbottom=i == 2)
        ax.tick_params(axis="y", labelsize=12, labelleft=True)
    fig.tight_layout(h_pad=1.1)
    fig.canvas.draw()
    if any(not ax.yaxis.label.get_visible() or not ax.get_yticklabels() for ax in axes):
        raise AssertionError("Every stacked panel must show its y-axis labels")
    portrait = geometry(axes)
    if portrait != reference:
        raise AssertionError("Portrait layout changed plotted values or scales")

    out.mkdir(parents=True)
    pdf = out / "bm25_retrieval_portrait.pdf"
    png = out / "bm25_retrieval_portrait.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=180)
    if geometry(axes) != reference:
        raise AssertionError("Saving altered data-space geometry")
    plt.close(fig)
    for relative, expected in INPUTS.items():
        if sha256(root / relative) != expected:
            raise AssertionError(f"Input changed during plotting: {relative}")
    manifest = {
        "purpose": "Figure 7 layout only; no retrieval scoring or statistical tests",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": INPUTS, "script_sha256": sha256(__file__),
        "rows": len(scores), "models": EXPECTED_MODELS, "contexts": CODES,
        "cell_counts": [{"model": model, "variation": code, "n": int(n)}
                        for (model, code), n in counts.items()],
        "layout": {"panels": "3 rows, 1 column", "size_inches": [4.9, 7.6],
                   "axis_font_pt": 14, "tick_font_pt": 12, "showfliers": False},
        "reference_geometry_sha256": reference,
        "portrait_geometry_sha256": portrait,
        "outputs": {p.name: sha256(p) for p in (pdf, png)},
        "environment": {"python": platform.python_version(),
                        **{name: importlib.metadata.version(name)
                           for name in ("matplotlib", "numpy", "pandas", "seaborn")}},
    }
    (out / "presentation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "verified", "output": str(pdf),
                      "rows": len(scores), "geometry_sha256": portrait}))


if __name__ == "__main__":
    main()
