#!/usr/bin/env python3
"""Independent, standard-library verification of the additive reporting package.

Can run against the public package. Does not import the producer, rerun any experiment, or conduct
significance tests. Quantiles use statistics.quantiles; Wilson uses its closed form.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=1e-10), (a, b)


def wilson(k, n):
    z = statistics.NormalDist().inv_cdf(.975)
    p = k/n
    center = p + z*z/(2*n)
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    divisor = 1+z*z/n
    return max(0., (center-half)/divisor), min(1., (center+half)/divisor)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--b1-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    manifest = json.loads((a.output_dir/"manifest.json").read_text())
    for path, expected in manifest["source_sha256"].items():
        assert digest(a.repo_root/path) == expected, path
    for name, expected in manifest["output_sha256"].items():
        assert digest(a.output_dir/name) == expected, name
    rate_rows = rows(a.output_dir/"detector_recall_intervals.csv")
    source_scores = rows(a.b1_dir/"all_detector_scores.csv")
    performance = rows(a.b1_dir/"performance.csv")
    for r in rate_rows:
        ai = [s for s in source_scores if s["detector_key"] == r["detector"] and s["entry_type"] == "ai"
              and (r["generator"] == "pooled" or s["model"] == r["generator"])
              and (r["context_code"] == "all" or s["context_code"] == r["context_code"])]
        tp = sum(s["predicted_label"] == "AI" for s in ai)
        assert tp == int(r["tp"]) and len(ai) == int(r["n_ai"])
        near(tp/len(ai), r["recall"])
        if r["context_code"] != "all":
            assert len({s["pr_id"] for s in ai}) == len(ai)
            bounds = wilson(tp, len(ai))
        else:
            p, = [s for s in performance if s["subset"] == "successful_only" and
                  all(s[k] == r[k] for k in ("detector", "generator", "context_code", "view_level"))]
            bounds = float(p["recall_ci_low"]), float(p["recall_ci_high"])
        for k, v in zip(("recall_ci_low", "recall_ci_high"), bounds):
            near(r[k], v)
    humans = rows(a.output_dir/"human_false_positive_intervals.csv")
    for r in humans:
        h = [s for s in source_scores if s["detector_key"] == r["detector"] and s["entry_type"] == "human"]
        assert len(h) == len({s["pr_id"] for s in h}) == 308
        k = sum(s["predicted_label"] == "AI" for s in h)
        assert k == int(r["fp"])
        for key, value in zip(("fpr_ci_low", "fpr_ci_high"), wilson(k, 308)):
            near(r[key], value)
    # NIST's published worked example, plus boundary and symmetry checks.
    for v, target in zip(wilson(8, 30), (.141827, .444480)):
        assert abs(v-target) < 5e-7
    near(wilson(0,308)[0], 0)
    assert wilson(0,308)[1] > .012
    near(wilson(308,308)[1], 1)
    near(wilson(4,308)[0], 1-wilson(304,308)[1])
    feature_specs = {
        "Word Count": ("word_sentence_metrics.csv", "word_count"),
        "Sentence Count": ("word_sentence_metrics.csv", "sentence_count"),
        "Flesch Kincaid Grade": ("fkgl_metrics.csv", "flesch_kincaid_grade"),
        "Type-Token Ratio": ("ttr_metrics.csv", "ttr"),
        "MTLD Bidirectional": ("mtld_metrics.csv", "mtld_bidir"),
        "Lexical Density": ("lexical_density_metrics.csv", "lexical_density"),
    }
    base = a.repo_root/"phase3/analysis/outputs/corrected_rq23_v1/notebooks/outputs/row_level"
    source = {name: rows(base/name) for name, _ in feature_specs.values()}
    desc = rows(a.output_dir/"matched_textual_descriptives.csv")
    missing = []
    for r in desc:
        filename, column = feature_specs[r["metric"]]
        records = source[filename]
        human = {s["normalized_pr_id"]: s for s in records if s["entry_type"] == "human"}
        ai = {s["normalized_pr_id"]: s for s in records if s["entry_type"] == "ai"
              and s["model"].replace("meta-llama_", "meta-llama/") == r["model"] and s["variation"] == r["context_code"]}
        eligible = sorted(i for i in ai if ai[i][column] and human[i][column]
                          and math.isfinite(float(ai[i][column])) and math.isfinite(float(human[i][column])))
        assert len(eligible) == int(r["n_pairs"])
        assert len(ai)-len(eligible) == int(r["n_excluded_missing_metric"])
        if len(eligible) != len(ai):
            assert r["metric"] == "MTLD Bidirectional"
            missing.append({k:r[k] for k in ("model","context_code","n_pairs","n_excluded_missing_metric")})
        assert hashlib.sha256(("\n".join(eligible)+"\n").encode()).hexdigest() == r["pr_set_sha256"]
        for prefix, lookup in [("human_", human), ("ai_", ai)]:
            values = [float(lookup[i][column]) for i in eligible]
            qs = statistics.quantiles(values, n=4, method="inclusive")
            for key, value in zip(("q1", "median", "q3"), qs):
                near(r[prefix+key], value)
            near(r[prefix+"mean"], statistics.mean(values))
    assert len(desc) == 108 and len(rate_rows) == 88 and len(humans) == 4
    result = {"status":"PASS", "verifier_sha256":digest(Path(__file__)),
              "manifest_sha256":digest(a.output_dir/"manifest.json"),
              "descriptive_comparisons_independently_reconstructed":len(desc),
              "rate_intervals_independently_checked":len(rate_rows)+len(humans),
              "nist_worked_example_and_boundary_checks":"PASS",
              "source_and_output_hashes":"PASS", "existing_mtld_complete_pair_exclusions":missing}
    (a.output_dir/"independent_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "existing_mtld_complete_pair_exclusions"}, indent=2))


if __name__ == "__main__":
    main()
