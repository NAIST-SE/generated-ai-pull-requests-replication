#!/usr/bin/env python3
"""Nokogiri-only, plot-only presentation refresh from hash-pinned cached results.

Never execute whole notebooks or recalculate scores/tests. Execute only the listed
plotting cells, with frames loaded from their frozen row-level exports. Preserve
legacy data keys and change tick labels only at save time. Historical outputs are
read-only; --output-dir must be new. See context_label_inputs.json for source hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
from pypdf import PdfWriter


DISPLAY = {
    "File Diff": "Patch", "File\nDiff": "Patch", "Diff": "Patch",
    "File Contents": "Full Code", "File\nContents": "Full\nCode",
    "File Cont.": "Full Code", "File": "Full Code",
    "Without File Contents": "Multi-source", "No File Cont.": "Multi-source",
    "No-FC": "Multi-source",
}
LEGACY = dict(zip([f"P-{i}" for i in range(1, 7)], [
    "Repo Path", "PR Title", "File Diff", "File Contents", "Linked Issue",
    "Without File Contents",
]))
NOTEBOOKS = "phase3/analysis/outputs/corrected_rq23_v1/notebooks"
PLOTS = [
    ("rq2_word_sentence_count", 17, "word_sentence", None),
    ("rq2_reading_ease", 15, "fkgl", "context_order = ["),
    ("rq2_ttr", 37, "ttr", None),
    ("rq2_mtld", 38, "mtld", None),
    ("rq2_lexical_density", 12, "lexical_density", "# Context comparison"),
    ("rq2_rouge", 15, "rouge", None),
    ("rq2_bertscore", 18, "bertscore", None),
    ("rq3_codebert", 15, "codebert", None),
    ("rq3_graphcodebert", 15, "graphcodebert", None),
]
PAGE_ORDER = [
    "verbosity_word_count.pdf", "fkgl.pdf", "ttr.pdf",
    "MTLD_Bidirectional_Distribution_Human_vs_Contexts.pdf", "lexical_density.pdf",
    "rouge.pdf", "bertscore.pdf", "bm25_retrieval.pdf", "codebert.pdf",
    "graphcodebert.pdf",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(fig):
    """Hash data coordinates, box geometry and scales, not layout/text positions."""
    axes = []
    for ax in fig.axes:
        axes.append({
            "limits": [list(ax.get_xlim()), list(ax.get_ylim())],
            "ticks": [np.asarray(ax.get_xticks()).tolist(), np.asarray(ax.get_yticks()).tolist()],
            "lines": [np.asarray(line.get_xydata()).tolist() for line in ax.lines],
            "boxes": [patch.get_path().vertices.tolist() for patch in ax.patches],
            "collections": [np.asarray(item.get_offsets()).tolist() for item in ax.collections],
        })
    return hashlib.sha256(json.dumps(axes, sort_keys=True).encode()).hexdigest()


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("context_label_inputs.json"))
    parser.add_argument("--rq3-plot-source", type=Path,
                        help="Optional exact exported RQ3 plot source (same pinned hash).")
    parser.add_argument("--retrieval-summary", type=Path,
                        help="Optional exact exported retrieval summary (same pinned hash).")
    args = parser.parse_args()
    root, out = args.study_root.resolve(), args.output_dir.resolve()
    sys.path[:0] = [str(root / "phase3/analysis/deep_audit"),
                   str(root / NOTEBOOKS)]
    from server_guard import require_nokogiri
    require_nokogiri()
    inputs = json.loads(args.manifest.read_text())["inputs"]
    rq3_key = "phase3/analysis/rq3_mismatched_calibration/plot_manuscript_rq3.py"
    retrieval_key = "phase3/analysis/outputs/rq3_retrieval_inference_v1/summaries/context_human_ai_reciprocal_rank_comparisons.csv"
    paths = {relative: root / relative for relative in inputs}
    if args.rq3_plot_source:
        paths[rq3_key] = args.rq3_plot_source.resolve()
    if args.retrieval_summary:
        paths[retrieval_key] = args.retrieval_summary.resolve()
    for relative, expected in inputs.items():
        if sha256(paths[relative]) != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {relative}")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite a previous run: {out}")
    out.mkdir(parents=True)
    os.chdir(out)
    audit = []
    save_original = Figure.savefig

    def save_with_labels(fig, filename, *pos, **kwargs):
        path = Path(filename).resolve()
        if not path.is_relative_to(out):
            raise ValueError(f"Refusing output outside new run directory: {path}")
        fig.canvas.draw()
        before = geometry(fig)
        changes = []
        is_rq1 = path.stem == "rq1_probability_boxplots_paired"
        for ax in fig.axes:
            labels = [item.get_text() for item in ax.get_xticklabels()]
            updated = [DISPLAY.get(label, label) for label in labels]
            if updated != labels:
                ax.set_xticks(ax.get_xticks(), updated)
                changes.append({"before": labels, "after": updated})
            if is_rq1 and any(labels):
                # Keep sample counts below the longer display labels.
                for annotation in ax.texts:
                    if annotation.get_text().startswith("n="):
                        annotation.set_position((0, -60))
        if is_rq1:
            fig.tight_layout(rect=[0.03, 0.045, 1, 1])
        elif not fig.get_constrained_layout():
            fig.tight_layout()
        fig.canvas.draw()
        after = geometry(fig)
        if before != after:
            raise AssertionError(f"Relabeling changed plotted values or axes: {path}")
        save_original(fig, filename, *pos, **kwargs)
        audit.append({"output": str(path.relative_to(out)), "label_changes": changes,
                      "geometry_before": before, "geometry_after": after})

    Figure.savefig = save_with_labels
    plt.show = lambda: plt.close("all")
    try:
        # RQ1 uses the original frozen pairwise results for bracket labels.
        rq1 = load_module(root / "phase3/analysis/deep_audit/plot_rq1_probability_boxplots.py")
        sys.argv = ["plot_rq1_probability_boxplots.py", "--input",
                    str(root / "phase3/datasets/temperature_1/run_3_corrected_detection_v1/detection_results.csv"),
                    "--pairwise", str(root / "phase3/analysis/outputs/corrected_detection_v1/rq1/rq1_model_context_pairwise.csv"),
                    "--output-dir", str(out / "rq1")]
        rq1.main()
        plt.close("all")

        # RQ2 (and historical bundle pages 9-10): cached metric columns only.
        for name, index, cache, start in PLOTS:
            plt.rcdefaults()
            sns.set_theme(style="whitegrid")
            notebook = json.loads((root / NOTEBOOKS / f"analysis_{name}.ipynb").read_text())
            cell = "".join(notebook["cells"][index]["source"])
            if start:
                cell = cell[cell.index(start):]
            frame = pd.read_csv(root / NOTEBOOKS / f"outputs/row_level/{cache}_metrics.csv")
            ai = frame[frame.entry_type.eq("ai")].copy()
            human = frame[frame.entry_type.eq("human")].copy()
            if len(ai) != 5322 or (len(human) != 308 and cache not in {"rouge", "bertscore"}):
                raise AssertionError(f"Unexpected cached row counts for {cache}")
            if not ai["Context_Name"].equals(ai["variation"].map(LEGACY)):
                raise AssertionError(f"Unexpected legacy label mapping for {cache}")
            env = {"plt": plt, "sns": sns, "pd": pd, "np": np, "Path": Path,
                   "ai_df": ai, "human_df": human, "CONTEXT_MAPPING": LEGACY}
            exec(compile(cell, f"{name}:plot-cell-{index}", "exec"), env)
            plt.close("all")

        # Extract only the existing BM25 plotting block, not its scoring/test code.
        plt.rcdefaults()
        source = (root / "phase3/analysis/deep_audit/analysis_rq2_bm25_retrieval.py").read_text()
        import textwrap
        block = source[source.index("    context_order = [VARIATION_TO_CONTEXT"):
                       source.index("    output_dir.mkdir", source.index("    context_order = [VARIATION_TO_CONTEXT"))]
        from types import SimpleNamespace
        scores = pd.read_csv(root / NOTEBOOKS / "outputs/bm25_retrieval/bm25_retrieval_scores.csv")
        if len(scores) != 5322:
            raise AssertionError("Unexpected BM25 cached row count")
        exec(compile(textwrap.dedent(block), "bm25:plot-block", "exec"), {
            "plt": plt, "sns": sns, "scores": scores,
            "VARIATION_TO_CONTEXT": LEGACY, "CONTEXT_CODES": list(LEGACY),
            "args": SimpleNamespace(figure_output=out / "outputs/figures_pdf/bm25_retrieval.pdf"),
        })

        writer = PdfWriter()
        for filename in PAGE_ORDER:
            writer.append(out / "outputs/figures_pdf" / filename)
        writer.write(out / "textual_features_boxplots.pdf")
        writer.close()

        plt.rcdefaults()
        rq3 = load_module(paths[rq3_key])
        sys.argv = ["plot_manuscript_rq3.py", "--retrieval-context",
                    str(paths[retrieval_key]),
                    "--calibration-groups",
                    str(root / "phase3/analysis/outputs/rq3_mismatched_calibration_v1/summaries/group_summary.csv"),
                    "--output-dir", str(out / "rq3")]
        rq3.main()
    finally:
        Figure.savefig = save_original

    for relative, expected in inputs.items():
        if sha256(paths[relative]) != expected:
            raise AssertionError(f"Input changed during plotting: {relative}")
    manifest = {
        "protocol": "context-display-labels-v1", "host": platform.node(),
        "operation": "plot-only; no inference or statistical tests",
        "script_sha256": sha256(__file__), "inputs": inputs,
        "source_overrides": {k: str(p) for k, p in paths.items() if p != root / k},
        "display_mapping": DISPLAY, "plot_geometry_checks": audit,
        "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                     "seaborn": sns.__version__, "pandas": pd.__version__},
        "bundle_page_order": PAGE_ORDER,
        "outputs": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()},
    }
    (out / "presentation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output_dir": str(out), "saved_figures": len(audit),
                      "input_hashes_unchanged": True, "geometry_checks_passed": True}))


if __name__ == "__main__":
    main()
