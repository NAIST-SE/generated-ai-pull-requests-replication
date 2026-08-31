#!/usr/bin/env python3
"""Regenerate cleaned descriptive detector metrics for Preliminary-RQ and RQ1.

SERVER ONLY. This complements the paired inferential scripts: it computes the operational
confusion-matrix metrics used by the manuscript's File-Diff table and the per-context F1
ceiling/floor table. Detector failures are explicit negative predictions in the primary
view; the successful-only view is retained as sensitivity evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from server_guard import require_nokogiri


DETECTORS = {
    "ZeroGPT": ("zerogpt_is_ai", "zerogpt_success"),
    "ChatGPT Detector RoBERTa": ("roberta_is_ai", "roberta_success"),
    "ChatGPT-QA Detector RoBERTa": ("roberta-qa_is_ai", "roberta-qa_success"),
}
VARIATION_TO_CONTEXT = {
    "P-1": "Repo Path",
    "P-2": "PR Title",
    "P-3": "File Diff",
    "P-4": "File Contents",
    "P-5": "Linked Issue",
    "P-6": "Without File Contents",
}


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
        default=Path("phase3/analysis/outputs/deep_audit/detection_descriptive"),
    )
    return parser.parse_args()


def confusion_counts(truth: pd.Series, prediction: pd.Series) -> dict[str, int]:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    return {
        "tp": int((truth & prediction).sum()),
        "fp": int((~truth & prediction).sum()),
        "tn": int((~truth & ~prediction).sum()),
        "fn": int((truth & ~prediction).sum()),
    }


def main() -> None:
    require_nokogiri()
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    frame = pd.read_csv(input_path)
    frame["entry_type"] = frame["entry_type"].astype(str).str.lower()
    human = frame[frame["entry_type"].eq("human")].copy()
    ai = frame[frame["entry_type"].eq("ai")].copy()
    ai["model"] = ai["model"].astype(str).str.strip()
    ai["variation"] = ai["variation"].astype(str).str.strip()
    models = sorted(ai["model"].unique())

    rows: list[dict] = []
    for detector, (prediction_col, success_col) in DETECTORS.items():
        if prediction_col not in frame or success_col not in frame:
            raise RuntimeError(f"Missing detector columns for {detector}")
        human_detector = human[[prediction_col, success_col]].copy()
        human_detector["prediction"] = as_bool(human_detector[prediction_col])
        human_detector["success"] = as_bool(human_detector[success_col])

        for model in models:
            for variation, context in VARIATION_TO_CONTEXT.items():
                ai_cell = ai[ai["model"].eq(model) & ai["variation"].eq(variation)][
                    [prediction_col, success_col]
                ].copy()
                ai_cell["prediction"] = as_bool(ai_cell[prediction_col])
                ai_cell["success"] = as_bool(ai_cell[success_col])

                for view in ("operational", "successful_only"):
                    h = human_detector.copy()
                    a = ai_cell.copy()
                    if view == "operational":
                        h.loc[~h["success"], "prediction"] = False
                        a.loc[~a["success"], "prediction"] = False
                    else:
                        h = h[h["success"]].copy()
                        a = a[a["success"]].copy()

                    truth = pd.Series([False] * len(h) + [True] * len(a), dtype=bool)
                    prediction = pd.concat(
                        [h["prediction"], a["prediction"]], ignore_index=True
                    ).astype(bool)
                    counts = confusion_counts(truth, prediction)
                    specificity_denominator = counts["tn"] + counts["fp"]
                    rows.append(
                        {
                            "view": view,
                            "detector": detector,
                            "model": model,
                            "variation": variation,
                            "context": context,
                            "n_human": int(len(h)),
                            "n_ai": int(len(a)),
                            "human_detector_failures": int((~human_detector["success"]).sum()),
                            "ai_detector_failures": int((~ai_cell["success"]).sum()),
                            **counts,
                            "accuracy": float(accuracy_score(truth, prediction)),
                            "precision": float(precision_score(truth, prediction, zero_division=0)),
                            "recall": float(recall_score(truth, prediction, zero_division=0)),
                            "f1": float(f1_score(truth, prediction, zero_division=0)),
                            "specificity": (
                                counts["tn"] / specificity_denominator
                                if specificity_denominator
                                else float("nan")
                            ),
                        }
                    )

    metrics = pd.DataFrame(rows)
    operational = metrics[metrics["view"].eq("operational")].copy()
    prelim = operational[operational["variation"].eq("P-3")].copy()

    extrema_rows: list[dict] = []
    for (detector, model), group in operational.groupby(["detector", "model"]):
        best = float(group["f1"].max())
        worst = float(group["f1"].min())
        extrema_rows.append(
            {
                "detector": detector,
                "model": model,
                "best_f1": best,
                "best_contexts": "; ".join(group.loc[group["f1"].eq(best), "context"]),
                "worst_f1": worst,
                "worst_contexts": "; ".join(group.loc[group["f1"].eq(worst), "context"]),
            }
        )
    extrema = pd.DataFrame(extrema_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "detector_metrics_by_view_model_context.csv", index=False)
    prelim.to_csv(output_dir / "prelim_file_diff_metrics.csv", index=False)
    extrema.to_csv(output_dir / "rq1_f1_extrema.csv", index=False)
    metadata = {
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "primary_view": "operational",
        "operational_failure_policy": "detector failure is prediction False (not flagged)",
        "successful_only_policy": "drop detector failures independently within the cell",
        "human_baseline_policy": "same 308 human rows reused as the negative class in each operational cell",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(prelim.to_string(index=False))
    print(extrema.to_string(index=False))


if __name__ == "__main__":
    main()
