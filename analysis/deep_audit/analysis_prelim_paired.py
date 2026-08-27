#!/usr/bin/env python3
"""Paired Preliminary-Question model comparisons on the File Diff context.

SERVER ONLY. Uses PR as the paired unit, Cochran's Q for the three-model omnibus,
and exact McNemar tests for post-hoc pairs. It never duplicates the human baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

from server_guard import require_nokogiri


DETECTORS = {
    "ZeroGPT": ("zerogpt_is_ai", "zerogpt_success"),
    "ChatGPT Detector RoBERTa": ("roberta_is_ai", "roberta_success"),
    "ChatGPT-QA Detector RoBERTa": ("roberta-qa_is_ai", "roberta-qa_success"),
}
FILE_DIFF = "P-3"


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
        default=Path("phase3/analysis/outputs/deep_audit/prelim"),
    )
    return parser.parse_args()


def main() -> None:
    require_nokogiri()
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    frame = pd.read_csv(input_path)
    ai = frame[
        frame["entry_type"].astype(str).str.lower().eq("ai")
        & frame["variation"].astype(str).eq(FILE_DIFF)
    ].copy()
    ai["normalized_pr_id"] = ai["pr_id"].astype(str).str.strip()

    models = sorted(ai["model"].astype(str).unique())
    if len(models) != 3:
        raise RuntimeError(f"Expected three models in File Diff; observed {models}")
    if ai.duplicated(["normalized_pr_id", "model"]).any():
        raise RuntimeError("Duplicate PR/model rows in File Diff input")

    omnibus_rows: list[dict] = []
    pair_rows: list[dict] = []

    for detector, (prediction_col, success_col) in DETECTORS.items():
        if prediction_col not in ai or success_col not in ai:
            raise RuntimeError(f"Missing detector columns for {detector}")
        detector_frame = ai[["normalized_pr_id", "model", prediction_col, success_col]].copy()
        detector_frame["prediction"] = as_bool(detector_frame[prediction_col]).astype(int)
        detector_frame["success"] = as_bool(detector_frame[success_col])

        for view in ("operational", "successful_only"):
            view_frame = detector_frame.copy()
            if view == "operational":
                # Failed detector calls do not flag a description in deployment.  Set
                # this explicitly rather than relying on the serialised prediction
                # value being false or missing on every failure row.
                view_frame.loc[~view_frame["success"], "prediction"] = 0
            else:
                success_wide = view_frame.pivot(
                    index="normalized_pr_id", columns="model", values="success"
                ).reindex(columns=models)
                eligible_ids = success_wide.dropna().all(axis=1)
                eligible_ids = eligible_ids[eligible_ids].index
                view_frame = view_frame[view_frame["normalized_pr_id"].isin(eligible_ids)]

            wide = view_frame.pivot(
                index="normalized_pr_id", columns="model", values="prediction"
            ).reindex(columns=models).dropna()
            matrix = wide.to_numpy(dtype=int)
            if len(wide) == 0:
                raise RuntimeError(f"No complete cases for {detector}, {view}")

            q_result = cochrans_q(matrix)
            omnibus_rows.append(
                {
                    "view": view,
                    "detector": detector,
                    "n_prs": int(len(wide)),
                    "q_stat": float(q_result.statistic),
                    "df": int(len(models) - 1),
                    "p": float(q_result.pvalue),
                    **{f"rate__{model}": float(wide[model].mean()) for model in models},
                }
            )

            for model_a, model_b in combinations(models, 2):
                a = wide[model_a].astype(int)
                b = wide[model_b].astype(int)
                table = pd.crosstab(a, b).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
                test = mcnemar(table.to_numpy(), exact=True)
                discordant_ab = int(((a == 1) & (b == 0)).sum())
                discordant_ba = int(((a == 0) & (b == 1)).sum())
                pair_rows.append(
                    {
                        "view": view,
                        "detector": detector,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_prs": int(len(wide)),
                        "rate_a": float(a.mean()),
                        "rate_b": float(b.mean()),
                        "paired_rate_difference_a_minus_b": float((a - b).mean()),
                        "discordant_a1_b0": discordant_ab,
                        "discordant_a0_b1": discordant_ba,
                        "mcnemar_stat": float(test.statistic),
                        "p": float(test.pvalue),
                    }
                )

    omnibus = pd.DataFrame(omnibus_rows)
    pairs = pd.DataFrame(pair_rows)
    pairs["adj_p"] = np.nan
    pairs["sig"] = False
    for view, index in pairs.groupby("view").groups.items():
        reject, adjusted, _, _ = multipletests(pairs.loc[index, "p"], method="fdr_bh")
        pairs.loc[index, "adj_p"] = adjusted
        pairs.loc[index, "sig"] = reject

    output_dir.mkdir(parents=True, exist_ok=True)
    omnibus.to_csv(output_dir / "prelim_cochrans_q.csv", index=False)
    pairs.to_csv(output_dir / "prelim_mcnemar_pairwise.csv", index=False)
    metadata = {
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "context": FILE_DIFF,
        "models": models,
        "omnibus": "Cochran's Q per detector and view",
        "post_hoc": "exact McNemar; BH-FDR separately over nine tests per view",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(omnibus.to_string(index=False))
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
