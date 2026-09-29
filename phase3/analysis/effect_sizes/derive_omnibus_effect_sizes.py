"""Derive effect sizes for the omnibus tests reported in the manuscript.

The stored test outputs record the test statistics but no effect sizes. This script
derives them exactly from those statistics, without rerunning any test:

- Kendall's W for each Friedman test: W = chi2 / (n * (k - 1)). The Friedman statistics
  come from scipy.stats.friedmanchisquare, which applies the tie correction, so this is
  the tie-corrected W.
- eta^2_Q for each Cochran's Q test: eta2_Q = Q / (n * (k - 1))
  (Serlin, Carr, and Marascuilo, 1982, doi:10.1037/0033-2909.92.3.786).
- The range of absolute paired flag-rate differences across the McNemar comparisons,
  read directly from the stored paired_rate_difference_a_minus_b column.

Inputs are read from phase3/analysis/outputs; outputs are written to
phase3/analysis/outputs/omnibus_effect_sizes_v1. Each output row records the SHA-256 of
its source file.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
DEST = OUTPUTS / "omnibus_effect_sizes_v1"

COCHRAN = OUTPUTS / "corrected_detection_v1/prelim/prelim_cochrans_q.csv"
MCNEMAR = OUTPUTS / "corrected_detection_v1/prelim/prelim_mcnemar_pairwise.csv"
RQ1_FRIEDMAN = OUTPUTS / "corrected_detection_v1/rq1/rq1_friedman.csv"
BINOCULARS_FRIEDMAN = OUTPUTS / "tier_b/b1-modern-detectors-v1/rq1_friedman.csv"
BM25_FRIEDMAN = OUTPUTS / "corrected_rq23_v1/notebooks/outputs/bm25_retrieval/bm25_retrieval_friedman.csv"

N_CONTEXTS = 6  # Friedman tests compare the six context configurations
N_GENERATORS = 3  # Cochran's Q compares the three generators


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path, **filters: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if all(row[key] == value for key, value in filters.items())]


def rel(path: Path) -> str:
    return str(path.relative_to(OUTPUTS.parent))


def main() -> None:
    records: list[dict[str, object]] = []

    for row in read_rows(COCHRAN, view="operational"):
        n, q = int(row["n_prs"]), float(row["q_stat"])
        records.append({
            "analysis": "prelim", "test": "cochrans_q", "group": row["detector"],
            "n": n, "k": N_GENERATORS, "statistic": q, "df": int(row["df"]), "p": float(row["p"]),
            "effect_size_name": "eta2_Q", "effect_size": q / (n * (N_GENERATORS - 1)),
            "source_file": rel(COCHRAN), "source_sha256": sha256(COCHRAN),
        })

    friedman_sources = [
        ("rq1", RQ1_FRIEDMAN, {"view": "operational"}, "detector"),
        ("rq1_binoculars_posthoc", BINOCULARS_FRIEDMAN, {"subset": "successful_only"}, "detector"),
        ("rq2_bm25", BM25_FRIEDMAN, {}, "model"),
    ]
    for analysis, path, filters, group_col in friedman_sources:
        for row in read_rows(path, **filters):
            n, chi2 = int(row["n_complete_prs"]), float(row["statistic"])
            records.append({
                "analysis": analysis, "test": "friedman", "group": row[group_col],
                "n": n, "k": N_CONTEXTS, "statistic": chi2, "df": int(row["df"]), "p": float(row["p"]),
                "effect_size_name": "kendalls_w", "effect_size": chi2 / (n * (N_CONTEXTS - 1)),
                "source_file": rel(path), "source_sha256": sha256(path),
            })

    diffs = [abs(float(row["paired_rate_difference_a_minus_b"])) for row in read_rows(MCNEMAR, view="operational")]
    for name, value in (("abs_paired_rate_difference_min", min(diffs)), ("abs_paired_rate_difference_max", max(diffs))):
        records.append({
            "analysis": "prelim", "test": "mcnemar", "group": f"all {len(diffs)} comparisons",
            "n": "", "k": "", "statistic": "", "df": "", "p": "",
            "effect_size_name": name, "effect_size": value,
            "source_file": rel(MCNEMAR), "source_sha256": sha256(MCNEMAR),
        })

    DEST.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with (DEST / "omnibus_effect_sizes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    metadata = {
        "description": "Effect sizes derived from stored omnibus test statistics; no test was rerun.",
        "formulas": {
            "kendalls_w": "chi2 / (n * (k - 1)), k = 6 context configurations; tie-corrected via scipy's Friedman chi2",
            "eta2_Q": "Q / (n * (k - 1)), k = 3 generators (Serlin, Carr, and Marascuilo, 1982)",
            "mcnemar": "range of |paired_rate_difference_a_minus_b| across the nine comparisons",
        },
        "inputs": {rel(p): sha256(p) for p in (COCHRAN, MCNEMAR, RQ1_FRIEDMAN, BINOCULARS_FRIEDMAN, BM25_FRIEDMAN)},
        "n_records": len(records),
    }
    (DEST / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
