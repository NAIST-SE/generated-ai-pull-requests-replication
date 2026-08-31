#!/usr/bin/env python3
"""Plot corrected RQ1 detector probabilities from verified analysis outputs.

SERVER ONLY. The script does not calculate inferential statistics: bracket selection and
labels are read from ``rq1_model_context_pairwise.csv`` so the figure cannot silently
diverge from the tabular RQ1 result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from server_guard import require_nokogiri  # noqa: E402


DETECTORS = {
    "roberta": ("ChatGPT Detector RoBERTa", "roberta_ai_probability", "roberta_success"),
    "roberta-qa": (
        "ChatGPT-QA Detector RoBERTa",
        "roberta-qa_ai_probability",
        "roberta-qa_success",
    ),
    "zerogpt": ("ZeroGPT", "zerogpt_ai_probability", "zerogpt_success"),
}
MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "gpt-5-mini-2025-08-07",
    "meta-llama/llama-3.1-70b-instruct",
]
MODEL_LABELS = {
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B",
    "gpt-5-mini-2025-08-07": "GPT-5-mini",
    "meta-llama/llama-3.1-70b-instruct": "Llama-3.1-70B",
}
MODEL_COLORS = {
    "Qwen/Qwen2.5-72B-Instruct": "#1f77b4",
    "gpt-5-mini-2025-08-07": "#ff7f0e",
    "meta-llama/llama-3.1-70b-instruct": "#2ca02c",
}
CONTEXTS = {
    "P-1": "Repo Path",
    "P-2": "PR Title",
    "P-3": "File Diff",
    "P-4": "File Contents",
    "P-5": "Linked Issue",
    "P-6": "Without File Contents",
}
CONTEXT_SHORT = ["Repo", "Title", "Diff", "File", "Issue", "No-FC"]
MAX_BRACKETS_PER_PANEL = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pairwise", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def draw_bracket(ax, x1: int, x2: int, y: float, label: str) -> None:
    tick = 0.022
    ax.plot(
        [x1, x1, x2, x2],
        [y - tick, y, y, y - tick],
        lw=0.7,
        color="black",
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2,
        y + 0.012,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        clip_on=False,
    )


def plot_panel(
    ax,
    ai: pd.DataFrame,
    tool: str,
    model: str,
    panel_pairs: pd.DataFrame,
    *,
    show_x: bool,
    show_y: bool,
) -> None:
    _, probability_col, _ = DETECTORS[tool]
    values = []
    counts = []
    for context_code in CONTEXTS:
        cell = ai[
            ai["model"].eq(model) & ai["variation"].eq(context_code)
        ][probability_col].dropna()
        values.append(cell.to_numpy())
        counts.append(int(len(cell)))

    positions = np.arange(len(CONTEXTS))
    color = MODEL_COLORS[model]
    ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.35},
        medianprops={"color": "black", "lw": 1.0},
        boxprops={"facecolor": color, "alpha": 0.65, "edgecolor": "black", "lw": 0.7},
        whiskerprops={"color": "black", "lw": 0.7},
        capprops={"color": "black", "lw": 0.7},
    )
    ax.set_ylim(-0.02, 2.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(positions)
    ax.set_xticklabels(CONTEXT_SHORT, fontsize=12, rotation=30, ha="right")
    if not show_x:
        ax.set_xticklabels([])
    if show_y:
        ax.set_ylabel("AI probability", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.axhline(0.5, color="gray", linestyle="--", lw=0.6, alpha=0.55)

    significant = panel_pairs[as_bool(panel_pairs["sig"])].copy()
    significant["abs_effect"] = significant["rank_biserial"].abs()
    selected = significant.sort_values("abs_effect", ascending=False).head(
        MAX_BRACKETS_PER_PANEL
    )
    name_to_code = {name: code for code, name in CONTEXTS.items()}
    for index, (_, row) in enumerate(selected.iterrows()):
        x1 = list(CONTEXTS).index(name_to_code[row["context_a"]])
        x2 = list(CONTEXTS).index(name_to_code[row["context_b"]])
        draw_bracket(
            ax,
            x1,
            x2,
            1.08 + index * 0.17,
            f"r={row['rank_biserial']:+.2f}{row['Sig']}",
        )
    if len(significant) > len(selected):
        ax.text(
            0.98,
            0.985,
            f"+{len(significant) - len(selected)} more significant",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="dimgray",
            style="italic",
        )

    y_offset = -38 if show_x else -8
    for x, count in zip(positions, counts):
        ax.annotate(
            f"n={count}",
            xy=(x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9,
            color="dimgray",
            annotation_clip=False,
        )


def main() -> None:
    require_nokogiri()
    args = parse_args()
    input_path = args.input.resolve()
    pairwise_path = args.pairwise.resolve()
    output_dir = args.output_dir.resolve()

    frame = pd.read_csv(input_path)
    ai = frame[frame["entry_type"].astype(str).str.lower().eq("ai")].copy()
    ai["model"] = ai["model"].astype(str).str.strip()
    ai["variation"] = ai["variation"].astype(str).str.strip()
    pairs = pd.read_csv(pairwise_path)
    pairs = pairs[pairs["view"].eq("operational")].copy()

    for _, (_, probability_col, success_col) in DETECTORS.items():
        if not as_bool(ai[success_col]).all():
            raise RuntimeError(f"Corrected figure input contains failed calls: {success_col}")
        ai[probability_col] = pd.to_numeric(ai[probability_col], errors="raise")
        if not ai[probability_col].between(0, 1).all():
            raise RuntimeError(f"Probability outside [0, 1]: {probability_col}")

    expected_panels = {
        (label, model) for label, _, _ in DETECTORS.values() for model in MODELS
    }
    observed_panels = set(zip(pairs["detector"], pairs["model"]))
    if observed_panels != expected_panels:
        raise RuntimeError("Pairwise result does not contain the expected nine panels")

    figure, axes = plt.subplots(3, 3, figsize=(14.5, 13.5), sharey=False)
    for row_index, (tool, (detector_label, _, _)) in enumerate(DETECTORS.items()):
        for column_index, model in enumerate(MODELS):
            axis = axes[row_index, column_index]
            panel_pairs = pairs[
                pairs["detector"].eq(detector_label) & pairs["model"].eq(model)
            ]
            plot_panel(
                axis,
                ai,
                tool,
                model,
                panel_pairs,
                show_x=row_index == 2,
                show_y=column_index == 0,
            )
            if row_index == 0:
                axis.set_title(MODEL_LABELS[model], fontsize=12, pad=8)
            if column_index == 0:
                axis.text(
                    -0.34,
                    0.5,
                    detector_label,
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=12,
                )

    legend = [
        Patch(
            facecolor=MODEL_COLORS[model],
            alpha=0.65,
            edgecolor="black",
            label=MODEL_LABELS[model],
        )
        for model in MODELS
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.005),
        frameon=False,
        fontsize=9.5,
    )
    figure.tight_layout(rect=[0.03, 0.035, 1, 1.0])

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "rq1_probability_boxplots_paired.pdf"
    png_path = output_dir / "rq1_probability_boxplots_paired.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "pairwise": str(pairwise_path),
        "pairwise_sha256": sha256(pairwise_path),
        "figure_policy": "operational rows; all corrected detector calls must be successful",
        "panels": 9,
        "significant_contrasts": int(as_bool(pairs["sig"]).sum()),
        "maximum_brackets_per_panel": MAX_BRACKETS_PER_PANEL,
        "pdf_sha256": sha256(pdf_path),
        "png_sha256": sha256(png_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
