#!/usr/bin/env python3
"""Fixed-corpus BM25 retrieval analysis for RQ2.

SERVER ONLY. Each AI description queries one BM25 index of all human descriptions.
The paired human description is the sole relevant document. This intentionally avoids
per-cell fitting, min-max normalisation, and human self-similarity baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rank_bm25 import BM25Okapi
from scipy.stats import friedmanchisquare, rankdata

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import paired_stats as PS  # noqa: E402
from server_guard import require_nokogiri  # noqa: E402


VARIATION_TO_CONTEXT = {
    "P-1": "Repo Path",
    "P-2": "PR Title",
    "P-3": "File Diff",
    "P-4": "File Contents",
    "P-5": "Linked Issue",
    "P-6": "Without File Contents",
}
CONTEXT_CODES = list(VARIATION_TO_CONTEXT)
PRIMARY_CONTEXT = "P-5"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenize(text: object) -> list[str]:
    if pd.isna(text):
        return []
    return TOKEN_PATTERN.findall(str(text).lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("phase3/datasets/temperature_1/run_3_cleaned/detection_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase3/analysis/outputs/deep_audit/bm25_retrieval"),
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("phase3/analysis/outputs/figures_pdf/bm25_retrieval.pdf"),
    )
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    require_nokogiri()
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    frame = pd.read_csv(input_path, usecols=["pr_id", "entry_type", "variation", "model", "text"])
    frame["normalized_pr_id"] = frame["pr_id"].astype(str).str.strip()

    human = frame[frame["entry_type"].astype(str).str.lower().eq("human")].copy()
    ai = frame[frame["entry_type"].astype(str).str.lower().eq("ai")].copy()
    if human["normalized_pr_id"].duplicated().any():
        raise RuntimeError("Human corpus contains duplicate PR IDs")
    if len(human) != 308:
        raise RuntimeError(f"Expected 308 human corpus documents; observed {len(human)}")
    if ai.duplicated(["normalized_pr_id", "model", "variation"]).any():
        raise RuntimeError("AI data contain duplicate PR/model/context rows")

    human = human.sort_values("normalized_pr_id").reset_index(drop=True)
    corpus_ids = human["normalized_pr_id"].tolist()
    corpus_tokens = human["text"].map(tokenize).tolist()
    if any(len(tokens) == 0 for tokens in corpus_tokens):
        raise RuntimeError("Human BM25 corpus contains an empty token sequence")
    relevant_index = {pr_id: index for index, pr_id in enumerate(corpus_ids)}
    bm25 = BM25Okapi(corpus_tokens, k1=args.k1, b=args.b)

    score_rows: list[dict] = []
    for row in ai.itertuples(index=False):
        query_tokens = tokenize(row.text)
        if not query_tokens:
            raise RuntimeError(
                f"Empty AI query for {row.normalized_pr_id}, {row.model}, {row.variation}"
            )
        target = relevant_index.get(row.normalized_pr_id)
        if target is None:
            raise RuntimeError(f"No paired human document for PR {row.normalized_pr_id}")
        scores = np.asarray(bm25.get_scores(query_tokens), dtype=float)
        ranks = rankdata(-scores, method="average")
        rank = float(ranks[target])
        score_rows.append(
            {
                "normalized_pr_id": row.normalized_pr_id,
                "model": str(row.model),
                "variation": str(row.variation),
                "context": VARIATION_TO_CONTEXT.get(str(row.variation), str(row.variation)),
                "query_tokens": int(len(query_tokens)),
                "relevant_bm25_score": float(scores[target]),
                "relevant_rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "recall_at_1": int(rank <= 1),
                "recall_at_5": int(rank <= 5),
                "recall_at_10": int(rank <= 10),
            }
        )

    scores = pd.DataFrame(score_rows)
    unknown_contexts = sorted(set(scores["variation"]).difference(CONTEXT_CODES))
    if unknown_contexts:
        raise RuntimeError(f"Unexpected variation codes: {unknown_contexts}")

    summary = (
        scores.groupby(["model", "variation", "context"], as_index=False)
        .agg(
            n=("normalized_pr_id", "size"),
            median_rank=("relevant_rank", "median"),
            mean_reciprocal_rank=("reciprocal_rank", "mean"),
            median_reciprocal_rank=("reciprocal_rank", "median"),
            recall_at_1=("recall_at_1", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            recall_at_10=("recall_at_10", "mean"),
        )
        .sort_values(["model", "variation"])
    )

    friedman_rows: list[dict] = []
    for model, group in scores.groupby("model"):
        wide = group.pivot(
            index="normalized_pr_id", columns="variation", values="reciprocal_rank"
        ).reindex(columns=CONTEXT_CODES).dropna()
        if len(wide) == 0:
            raise RuntimeError(f"No complete six-context BM25 cases for {model}")
        result = friedmanchisquare(*(wide[code].to_numpy() for code in CONTEXT_CODES))
        friedman_rows.append(
            {
                "model": model,
                "n_complete_prs": int(len(wide)),
                "statistic": float(result.statistic),
                "df": len(CONTEXT_CODES) - 1,
                "p": float(result.pvalue),
            }
        )

    contrast_rows: list[dict] = []
    for model, group in scores.groupby("model"):
        primary = group[group["variation"].eq(PRIMARY_CONTEXT)][
            ["normalized_pr_id", "reciprocal_rank"]
        ]
        for comparison_code in CONTEXT_CODES:
            if comparison_code == PRIMARY_CONTEXT:
                continue
            comparison = group[group["variation"].eq(comparison_code)][
                ["normalized_pr_id", "reciprocal_rank"]
            ]
            result = PS.paired_test_from_frames(
                primary,
                comparison,
                "reciprocal_rank",
                key="normalized_pr_id",
            )
            contrast_rows.append(
                {
                    "model": model,
                    "context_a": VARIATION_TO_CONTEXT[PRIMARY_CONTEXT],
                    "context_b": VARIATION_TO_CONTEXT[comparison_code],
                    "sign": "positive values favour Linked Issue",
                    **result,
                }
            )
    contrasts = PS.apply_fdr(pd.DataFrame(contrast_rows), pcol="p")

    context_order = [VARIATION_TO_CONTEXT[code] for code in CONTEXT_CODES]
    context_short = {
        "Repo Path": "Repo Path",
        "PR Title": "PR Title",
        "File Diff": "File Diff",
        "File Contents": "File Cont.",
        "Linked Issue": "Issue",
        "Without File Contents": "No File Cont.",
    }
    model_order = sorted(scores["model"].unique())
    fig, axes = plt.subplots(1, len(model_order), figsize=(18, 5.5), sharey=True)
    if len(model_order) == 1:
        axes = [axes]
    for axis, model in zip(axes, model_order):
        subset = scores[scores["model"].eq(model)].copy()
        subset["context_short"] = subset["context"].map(context_short)
        sns.boxplot(
            data=subset,
            x="context_short",
            y="reciprocal_rank",
            order=[context_short[name] for name in context_order],
            ax=axis,
            color="#9ecae1",
            showfliers=False,
        )
        axis.set_title(model, fontsize=14)
        axis.set_xlabel("Context", fontsize=14)
        axis.set_ylabel("Reciprocal rank" if axis is axes[0] else "", fontsize=14)
        axis.tick_params(axis="x", labelrotation=35, labelsize=12)
        axis.tick_params(axis="y", labelsize=12)
        axis.set_ylim(0, 1.02)
    fig.tight_layout()
    figure_output = args.figure_output.resolve()
    figure_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_output, bbox_inches="tight")
    plt.close(fig)

    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "bm25_retrieval_scores.csv", index=False)
    summary.to_csv(output_dir / "bm25_retrieval_summary.csv", index=False)
    pd.DataFrame(friedman_rows).to_csv(output_dir / "bm25_retrieval_friedman.csv", index=False)
    contrasts.to_csv(output_dir / "bm25_retrieval_linked_issue_contrasts.csv", index=False)
    metadata = {
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "corpus_documents": int(len(human)),
        "tokenizer": "lowercase ASCII alphanumeric-and-underscore regex",
        "token_pattern": TOKEN_PATTERN.pattern,
        "bm25": {"implementation": "rank_bm25.BM25Okapi", "k1": args.k1, "b": args.b},
        "tie_policy": "average rank",
        "relevant_document": "human description for the same normalized_pr_id",
        "post_hoc_family": "15 Linked Issue vs other-context comparisons; one BH-FDR family",
        "figure": str(figure_output),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(pd.DataFrame(friedman_rows).to_string(index=False))
    print(contrasts.to_string(index=False))
    print(f"Saved retrieval figure to {figure_output}")


if __name__ == "__main__":
    main()
