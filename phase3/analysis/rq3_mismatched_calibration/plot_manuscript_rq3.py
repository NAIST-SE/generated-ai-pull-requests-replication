#!/usr/bin/env python3
"""Plot the verified RQ3 retrieval and mismatched-pair calibration results.

The figure deliberately separates the established retrieval outcome (MRR) from
the corpus-specific background-adjusted cosine margin.  It reads only frozen,
verified summary files and records their hashes next to the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


EXPECTED_HASHES = {
    "retrieval_context": "e0665021847633b045ac48eaac3ce38249b17033ab1272906d017ee4bfc15a72",
    "calibration_groups": "f589c8d8ee99507ef028eb5afd03e36de628f697ff3d162184de996f51882737",
}

MODEL_LABELS = {
    "meta-llama/llama-3.1-70b-instruct": "Llama",
    "gpt-5-mini-2025-08-07": "GPT",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen",
}
MODEL_ORDER = list(MODEL_LABELS)
CONTEXT_ORDER = ["P-1", "P-2", "P-3", "P-4", "P-5", "P-6"]
CONTEXT_LABELS = [
    "Repo\nPath",
    "PR\nTitle",
    "File\nDiff",
    "File\nContents",
    "Linked\nIssue",
    "Multi-\nsource",
]
COLORS = {"Llama": "#0072B2", "GPT": "#D55E00", "Qwen": "#009E73"}
MARKERS = {"Llama": "o", "GPT": "s", "Qwen": "^"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"Unexpected source hash for {path}: {observed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-context", type=Path, required=True)
    parser.add_argument("--calibration-groups", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def plot_panel(ax, data, encoder, value_col, human_value, ylabel, title, zero=False):
    subset = data[(data["encoder"] == encoder) & data["model"].isin(MODEL_ORDER)].copy()
    for model in MODEL_ORDER:
        rows = subset[subset["model"] == model].set_index("variation").loc[CONTEXT_ORDER]
        label = MODEL_LABELS[model]
        ax.plot(
            range(6),
            rows[value_col],
            color=COLORS[label],
            marker=MARKERS[label],
            markersize=5.4,
            linewidth=1.6,
            label=label,
        )
    ax.axhline(human_value, color="#333333", linewidth=1.3, linestyle="--", label="Human")
    if zero:
        ax.axhline(0, color="#9A9A9A", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_title(title, loc="left", fontsize=10.2, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(6), CONTEXT_LABELS)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.2)


def main() -> None:
    args = parse_args()
    verify_hash(args.retrieval_context, EXPECTED_HASHES["retrieval_context"])
    verify_hash(args.calibration_groups, EXPECTED_HASHES["calibration_groups"])

    retrieval = pd.read_csv(args.retrieval_context)
    calibration = pd.read_csv(args.calibration_groups)
    required_retrieval = {"encoder", "model", "variation", "ai_mrr", "human_mrr"}
    required_calibration = {
        "encoder", "entry_type", "model", "variation", "margin_mean_median"
    }
    if not required_retrieval.issubset(retrieval.columns):
        raise RuntimeError("Retrieval summary schema does not match the frozen protocol")
    if not required_calibration.issubset(calibration.columns):
        raise RuntimeError("Calibration summary schema does not match the frozen protocol")

    human_margin = (
        calibration[calibration["entry_type"] == "human"]
        .set_index("encoder")["margin_mean_median"]
        .to_dict()
    )
    ai_calibration = calibration[calibration["entry_type"] == "ai"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.15), constrained_layout=True)

    for col, encoder in enumerate(("codebert", "graphcodebert")):
        enc_label = "CodeBERT" if encoder == "codebert" else "GraphCodeBERT"
        r = retrieval[retrieval["encoder"] == encoder]
        # Use the full 308-PR human retrieval baseline as the visual reference.
        # Context-specific paired tests retain their exact eligible human subset.
        human_mrr = float(r[r["variation"] == "P-1"]["human_mrr"].iloc[0])
        plot_panel(
            axes[0, col], r, encoder, "ai_mrr", human_mrr,
        "Mean reciprocal rank", f"({'a' if col == 0 else 'b'}) {enc_label}: correct-diff retrieval"
        )
        plot_panel(
            axes[1, col], ai_calibration, encoder, "margin_mean_median",
            float(human_margin[encoder]), "Median cosine margin",
            f"({'c' if col == 0 else 'd'}) {enc_label}: cosine calibration", zero=True
        )

    axes[0, 0].set_ylim(0, 0.11)
    axes[0, 1].set_ylim(0, 0.11)
    axes[1, 0].set_ylim(-0.012, 0.016)
    axes[1, 1].set_ylim(0, 0.048)
    for ax in axes.flat:
        ax.set_xlabel("Context configuration")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False, fontsize=9)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "rq3_retrieval_calibration.pdf"
    png_path = args.output_dir / "rq3_retrieval_calibration.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "protocol": "rq3-manuscript-figure-v1",
        "inputs": {
            str(args.retrieval_context): sha256(args.retrieval_context),
            str(args.calibration_groups): sha256(args.calibration_groups),
        },
        "outputs": {
            pdf_path.name: sha256(pdf_path),
            png_path.name: sha256(png_path),
        },
        "panels": {
            "top": "Established exhaustive correct-diff retrieval; MRR is descriptive by context.",
            "bottom": "Background-adjusted matched-minus-mean-mismatched cosine margin; median by context.",
        },
    }
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(pdf_path)


if __name__ == "__main__":
    main()
