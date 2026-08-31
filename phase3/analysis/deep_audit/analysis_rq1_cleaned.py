#!/usr/bin/env python3
"""RQ1 paired context analysis on the cleaned run-3 derivative.

SERVER ONLY. The Friedman omnibus averages detector probability across available
generators for each PR/context, then retains PRs observed in all six contexts. Separate
model-specific paired contrasts are reported as follow-up estimands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import paired_stats as PS  # noqa: E402
from server_guard import require_nokogiri  # noqa: E402


DETECTORS = {
    "ZeroGPT": ("zerogpt_ai_probability", "zerogpt_success"),
    "ChatGPT Detector RoBERTa": ("roberta_ai_probability", "roberta_success"),
    "ChatGPT-QA Detector RoBERTa": ("roberta-qa_ai_probability", "roberta-qa_success"),
}
VARIATION_TO_CONTEXT = {
    "P-1": "Repo Path",
    "P-2": "PR Title",
    "P-3": "File Diff",
    "P-4": "File Contents",
    "P-5": "Linked Issue",
    "P-6": "Without File Contents",
}
CONTEXT_CODES = list(VARIATION_TO_CONTEXT)


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
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("phase3/datasets/temperature_1/run_3_cleaned/detection_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("phase3/analysis/outputs/deep_audit/rq1"),
    )
    return parser.parse_args()


def main() -> None:
    require_nokogiri()
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    frame = pd.read_csv(input_path)
    ai = frame[frame["entry_type"].astype(str).str.lower().eq("ai")].copy()
    ai["normalized_pr_id"] = ai["pr_id"].astype(str).str.strip()
    ai["variation"] = ai["variation"].astype(str).str.strip()
    ai["model"] = ai["model"].astype(str).str.strip()

    unknown = sorted(set(ai["variation"]).difference(CONTEXT_CODES))
    if unknown:
        raise RuntimeError(f"Unexpected variation codes: {unknown}")
    if ai.duplicated(["normalized_pr_id", "model", "variation"]).any():
        raise RuntimeError("Duplicate PR/model/context rows in RQ1 input")

    friedman_rows: list[dict] = []
    contrast_rows: list[dict] = []

    for detector, (probability_col, success_col) in DETECTORS.items():
        if probability_col not in ai or success_col not in ai:
            raise RuntimeError(f"Missing detector columns for {detector}")
        detector_frame = ai[
            ["normalized_pr_id", "model", "variation", probability_col, success_col]
        ].copy()
        detector_frame["success"] = as_bool(detector_frame[success_col])
        detector_frame["probability"] = pd.to_numeric(
            detector_frame[probability_col], errors="coerce"
        )

        for view in ("operational", "successful_only"):
            view_frame = detector_frame.copy()
            if view == "operational":
                # A detector that returns no score does not flag the description in use.
                view_frame.loc[~view_frame["success"], "probability"] = 0.0
            else:
                view_frame = view_frame[view_frame["success"]].copy()
            view_frame = view_frame.dropna(subset=["probability"])

            pooled = (
                view_frame.groupby(["normalized_pr_id", "variation"], as_index=False)[
                    "probability"
                ].mean()
            )
            wide = pooled.pivot(
                index="normalized_pr_id", columns="variation", values="probability"
            ).reindex(columns=CONTEXT_CODES).dropna()
            if len(wide) == 0:
                raise RuntimeError(f"No complete RQ1 cases for {detector}, {view}")
            result = friedmanchisquare(*(wide[code].to_numpy() for code in CONTEXT_CODES))
            friedman_rows.append(
                {
                    "view": view,
                    "detector": detector,
                    "aggregation": "mean across available generators per PR/context",
                    "n_complete_prs": int(len(wide)),
                    "statistic": float(result.statistic),
                    "df": len(CONTEXT_CODES) - 1,
                    "p": float(result.pvalue),
                }
            )

            for model, model_frame in view_frame.groupby("model"):
                for context_a, context_b in combinations(CONTEXT_CODES, 2):
                    a = model_frame[model_frame["variation"].eq(context_a)][
                        ["normalized_pr_id", "probability"]
                    ]
                    b = model_frame[model_frame["variation"].eq(context_b)][
                        ["normalized_pr_id", "probability"]
                    ]
                    paired = PS.paired_test_from_frames(
                        a,
                        b,
                        "probability",
                        key="normalized_pr_id",
                    )
                    contrast_rows.append(
                        {
                            "view": view,
                            "detector": detector,
                            "model": model,
                            "context_a": VARIATION_TO_CONTEXT[context_a],
                            "context_b": VARIATION_TO_CONTEXT[context_b],
                            "sign": "positive values mean context_a has higher detector probability",
                            **paired,
                        }
                    )

    friedman = pd.DataFrame(friedman_rows)
    contrasts = PS.apply_fdr_by(
        pd.DataFrame(contrast_rows),
        by=["view", "detector", "model"],
        pcol="p",
    )
    contrasts["Sig"] = contrasts["adj_p"].apply(PS.sig_mark)

    comparison_keys = ["detector", "model", "context_a", "context_b"]
    comparison_columns = comparison_keys + ["n", "rank_biserial", "adj_p", "sig"]
    operational = contrasts[contrasts["view"].eq("operational")][
        comparison_columns
    ].rename(
        columns={
            "n": "n_operational",
            "rank_biserial": "rank_biserial_operational",
            "adj_p": "adj_p_operational",
            "sig": "sig_operational",
        }
    )
    successful_only = contrasts[contrasts["view"].eq("successful_only")][
        comparison_columns
    ].rename(
        columns={
            "n": "n_successful_only",
            "rank_biserial": "rank_biserial_successful_only",
            "adj_p": "adj_p_successful_only",
            "sig": "sig_successful_only",
        }
    )
    robustness = operational.merge(
        successful_only,
        on=comparison_keys,
        how="inner",
        validate="one_to_one",
    )
    robustness["direction_reversal"] = np.sign(
        robustness["rank_biserial_operational"]
    ).ne(np.sign(robustness["rank_biserial_successful_only"]))
    robustness["significance_change"] = robustness["sig_operational"].ne(
        robustness["sig_successful_only"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    friedman.to_csv(output_dir / "rq1_friedman.csv", index=False)
    contrasts.to_csv(output_dir / "rq1_model_context_pairwise.csv", index=False)
    robustness.to_csv(
        output_dir / "rq1_operational_vs_successful_only.csv", index=False
    )
    metadata = {
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "contexts": VARIATION_TO_CONTEXT,
        "friedman_estimand": "per-PR/context mean probability across available generators",
        "pairwise_estimand": "within-model, within-PR context probability difference",
        "pairwise_fdr_family": "15 context pairs within each view/detector/model",
        "operational_failure_policy": "detector failure is probability 0 (not flagged)",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(friedman.to_string(index=False))
    print(contrasts.to_string(index=False))
    print(
        "Sensitivity comparison: "
        f"{len(robustness)} contrasts; "
        f"{int(robustness['direction_reversal'].sum())} direction reversals; "
        f"{int(robustness['significance_change'].sum())} significance changes"
    )


if __name__ == "__main__":
    main()
