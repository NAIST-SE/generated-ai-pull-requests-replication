#!/usr/bin/env python3
"""Add descriptive reporting and rate intervals from frozen outputs; run on Nokogiri.

No text extraction, generation, detector inference, threshold tuning, significance
testing or bootstrap rerun is performed. Existing pooled bootstrap intervals are
reused; Wilson intervals are added to independent-PR binomial rate summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.stats.proportion import proportion_confint

MODELS = ["Qwen/Qwen2.5-72B-Instruct", "gpt-5-mini-2025-08-07",
          "meta-llama/llama-3.1-70b-instruct"]
SHORT = dict(zip(MODELS, ["Qwen2.5-72B", "GPT-5-mini", "Llama-3.1-70B"]))
CONTEXT = dict(zip([f"P-{i}" for i in range(1, 7)],
                  ["Repo Path", "PR Title", "Patch", "Full Code", "Linked Issue", "Multi-source"]))
OLD_CONTEXT = {**CONTEXT, "P-3": "File Diff", "P-4": "File Contents", "P-6": "Without File Contents"}
DETECTORS = {"roberta": "RoBERTa", "roberta-qa": "RoBERTa-QA", "zerogpt": "ZeroGPT", "binoculars": "Binoculars"}
METRICS = [
    ("word_sentence_metrics.csv", "word_count", "Word Count", "Words", 1),
    ("word_sentence_metrics.csv", "sentence_count", "Sentence Count", "Sentences", 1),
    ("fkgl_metrics.csv", "flesch_kincaid_grade", "Flesch Kincaid Grade", "FKGL", 2),
    ("ttr_metrics.csv", "ttr", "Type-Token Ratio", "TTR", 3),
    ("mtld_metrics.csv", "mtld_bidir", "MTLD Bidirectional", "MTLD", 2),
    ("lexical_density_metrics.csv", "lexical_density", "Lexical Density", "Lexical density", 3),
]
PINNED = {
    "canonical_descriptive_stats.csv": "ec4c185ac9864ecf84f909758720a46c8c0a6791d2d75d91a31b3fe8c56f5703",
    "canonical_context_paired_stats.csv": "92a6db158e91363211b71f8cd096cf6851fa0a2606b980d381cfa9f7ce222131",
    "all_detector_scores.csv": "a63bc80f5a319e2c0ec80b6010a405794f75e53ffb6339b4f268deebeffd7396",
    "performance.csv": "f14e28fa4310b92732c69963c1b17c61d1950c0bdab4a48a6c9c16d52ec465aa",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_n(model, code):
    return 273 if code in {"P-5", "P-6"} else 302 if code == "P-4" and model != MODELS[2] else 308


def close(a, b):
    assert np.isclose(float(a), float(b), rtol=1e-10, atol=1e-12), (a, b)


def quantiles(values):
    v = np.asarray(values, dtype=float)
    assert np.isfinite(v).all() and len(v)
    q1, med, q3 = np.quantile(v, [.25, .5, .75], method="linear")
    return dict(n=len(v), mean=float(np.mean(v)), median=float(med), q1=float(q1), q3=float(q3))


def wilson(k, n):
    assert 0 <= k <= n and n > 0
    lo, hi = proportion_confint(k, n, alpha=.05, method="wilson")
    # Exact boundary values avoid tiny floating-point residues at k=0 or k=n.
    return 0.0 if k == 0 else float(lo), 1.0 if k == n else float(hi)


def normalize(frame):
    frame = frame.copy()
    frame["model"] = frame["model"].replace({"human": "Human", "meta-llama_llama-3.1-70b-instruct": MODELS[2]})
    frame["normalized_pr_id"] = frame["normalized_pr_id"].astype(str).str.strip()
    return frame


def build_descriptives(row_dir, canonical, paired, detector_scores):
    summaries, baselines, checks = [], [], []
    score_ai = detector_scores.query("detector_key == 'binoculars' and entry_type == 'ai'")
    score_keys = set(zip(score_ai.pr_id, score_ai.model, score_ai.context_code))
    human_ids = set(detector_scores.query("detector_key == 'binoculars' and entry_type == 'human'").pr_id)
    cache = {}
    for filename, column, metric, _, _ in METRICS:
        if filename not in cache:
            cache[filename] = normalize(pd.read_csv(row_dir / filename))
        frame = cache[filename]
        human = frame[frame.entry_type.eq("human")].set_index("normalized_pr_id")
        ai_all = frame[frame.entry_type.eq("ai")]
        assert len(human) == 308 and human.index.is_unique and set(human.index) == human_ids
        assert len(ai_all) == 5322 and not ai_all.duplicated(["normalized_pr_id", "model", "variation"]).any()
        assert set(zip(ai_all.normalized_pr_id, ai_all.model, ai_all.variation)) == score_keys
        assert not np.isinf(frame[column].to_numpy(dtype=float)).any()
        h_all = quantiles(human[column].dropna())
        c_h = canonical[(canonical.Metric == metric) & (canonical.Model == "Human")].iloc[0]
        for key in h_all:
            close(h_all[key], c_h[key])
        baselines.append({"metric": metric, **h_all})
        for model in MODELS:
            for code, display in CONTEXT.items():
                ai = ai_all[(ai_all.model == model) & (ai_all.variation == code)].set_index("normalized_pr_id").sort_index()
                assert len(ai) == expected_n(model, code)
                matched = pd.concat([human.loc[ai.index, column].rename("human"),
                                     ai[column].rename("ai")], axis=1).dropna()
                hs, gs = quantiles(matched.human), quantiles(matched.ai)
                c = canonical[(canonical.Metric == metric) & (canonical.Model == model) & (canonical.Variation == code)].iloc[0]
                marginal = quantiles(ai[column].dropna())
                for key in marginal:
                    close(marginal[key], c[key])
                p = paired[(paired.Metric == metric) & (paired.Model == model) & (paired.Context == OLD_CONTEXT[code])].iloc[0]
                close(p["n"], len(matched)); close(p.median_human, hs["median"]); close(p.median_ai, gs["median"])
                membership = "\n".join(sorted(matched.index)) + "\n"
                summaries.append({"metric": metric, "model": model, "context_code": code, "context": display,
                                  "n_pairs": len(matched), "n_available_generations": len(ai),
                                  "n_excluded_missing_metric": len(ai)-len(matched),
                                  "pr_set_sha256": hashlib.sha256(membership.encode()).hexdigest(),
                                  **{f"human_{k}": v for k, v in hs.items() if k != "n"},
                                  **{f"ai_{k}": v for k, v in gs.items() if k != "n"}})
                checks.append({"metric": metric, "model": model, "context_code": code,
                               "canonical_ai_summary_matches": True, "paired_medians_match": True, "pr_membership_matches": True})
    assert len(summaries) == 108
    return pd.DataFrame(summaries), pd.DataFrame(baselines), checks


def build_intervals(scores, performance):
    scores = scores.copy()
    assert len(scores) == 22520 and set(scores.detector_key) == set(DETECTORS)
    assert scores.success.astype(str).str.lower().eq("true").all()
    assert set(scores.predicted_label) <= {"AI", "Not AI"}
    scores["flag"] = scores.predicted_label.eq("AI").astype(int)
    assert not scores.duplicated(["description_id", "detector_key"]).any()
    original = performance[performance.subset.eq("successful_only")]
    assert len(original) == 88
    rows, humans, checks = [], [], []
    for detector in DETECTORS:
        d = scores[scores.detector_key.eq(detector)]
        h = d[d.entry_type.eq("human")]
        ai = d[d.entry_type.eq("ai")]
        assert len(h) == 308 and h.pr_id.is_unique and len(ai) == 5322
        fp = int(h.flag.sum())
        lo, hi = wilson(fp, len(h))
        humans.append({"detector": detector, "n_human": len(h), "fp": fp, "tn": len(h)-fp,
                       "fpr": fp/len(h), "fpr_ci_low": lo, "fpr_ci_high": hi, "method": "Wilson 95%"})
        views = [("pooled", "pooled", "all", ai)]
        views += [("generator", m, "all", ai[ai.model.eq(m)]) for m in MODELS]
        views += [("generator_context", m, c, ai[ai.model.eq(m) & ai.context_code.eq(c)]) for m in MODELS for c in CONTEXT]
        for level, model, code, group in views:
            src = original[(original.detector == detector) & (original.view_level == level) &
                           (original.generator == model) & (original.context_code == code)]
            assert len(src) == 1
            src = src.iloc[0]
            tp, n = int(group.flag.sum()), len(group)
            assert n > 0
            tn, fn = len(h)-fp, n-tp
            metrics = dict(tp=tp, fn=fn, tn=tn, fp=fp, n_ai=n, n_human=len(h), recall=tp/n,
                           specificity=tn/len(h), precision=tp/(tp+fp) if tp+fp else 0,
                           f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                           accuracy=(tp+tn)/(n+len(h)))
            for key, value in metrics.items():
                close(value, src[key])
            if level == "generator_context":
                assert group.pr_id.is_unique and n == expected_n(model, code)
                low, high = wilson(tp, n)
                method = "Wilson 95%"
            else:
                low, high = float(src.recall_ci_low), float(src.recall_ci_high)
                assert src.bootstrap_valid_replicates == 2000
                assert low < high, "Degenerate pooled interval requires separate consideration"
                method = "Reused PR-clustered percentile bootstrap 95%; 2000 replicates"
            assert 0 <= low <= tp/n <= high <= 1, (detector,model,code,tp,n,low,high)
            rows.append({"detector": detector, "view_level": level, "generator": model,
                         "context_code": code, "context": CONTEXT.get(code, "All contexts"),
                         **metrics, "recall_ci_low": low, "recall_ci_high": high, "method": method,
                         "source_bootstrap_recall_ci_low": src.recall_ci_low,
                         "source_bootstrap_recall_ci_high": src.recall_ci_high})
            checks.append({"detector": detector, "view_level": level, "generator": model,
                           "context_code": code, "all_point_estimates_match": True})
    return pd.DataFrame(rows), pd.DataFrame(humans), checks


def cell(row, prefix, digits):
    return f"{row[prefix+'median']:.{digits}f} [{row[prefix+'q1']:.{digits}f}, {row[prefix+'q3']:.{digits}f}]"


def write_descriptive_tex(summary, out):
    for group, specs in [("structural", METRICS[:3]), ("lexical", METRICS[3:])]:
        heading = "Structural and readability" if group == "structural" else "Lexical"
        caption = (heading + r" values by model and context (RQ2): median [Q1, Q3]. H is the human baseline matched to that model--context cell; A is AI. "
                   r"The same $n$ applies to H and A; the main $n$ reflects eligibility or unavailable generations. ")
        caption += (r"FKGL is the Flesch--Kincaid grade level." if group == "structural" else
                    r"TTR and lexical density are proportions. Bidirectional MTLD uses complete pairs only ($n_{\mathrm{MTLD}}$).")
        columns = "llrlll" if group == "structural" else "llrrlll"
        count_header = r"\textbf{$n$}" if group == "structural" else r"\textbf{$n$} & \textbf{$n_{\mathrm{MTLD}}$}"
        lines = ["% Generated from frozen per-record metrics; do not hand-edit.", r"\begin{table*}[tp]", r"\centering",
                 r"\caption{"+caption+"}", r"\label{tab:rq2_"+group+"_descriptives}", r"\footnotesize",
                 r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.08}",
                 r"\begin{tabular}{"+columns+"}", r"\toprule",
                 r"\textbf{Model} & \textbf{Context} & "+count_header+" & " + " & ".join(r"\textbf{"+s[3]+"}" for s in specs)+r" \\", r"\midrule"]
        for mi, model in enumerate(MODELS):
            for ci, (code, display) in enumerate(CONTEXT.items()):
                parts = [SHORT[model] if ci == 0 else "", display]
                values, mtld_n = [], None
                for _, _, metric, _, digits in specs:
                    r = summary[(summary.metric == metric) & (summary.model == model) & (summary.context_code == code)].iloc[0]
                    if metric == "MTLD Bidirectional": mtld_n = str(int(r.n_pairs))
                    values.append(r"\shortstack[l]{H: "+cell(r, "human_", digits)+r"\\A: "+cell(r,"ai_",digits)+"}")
                counts = [str(expected_n(model,code))] + ([mtld_n] if group == "lexical" else [])
                lines.append(" & ".join(parts+counts+values)+r" \\")
            if mi < 2: lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
        (out/f"rq2_{group}_descriptives.tex").write_text("\n".join(lines))


def interval(row, percent=False):
    scale, precision = (100,1) if percent else (1,3)
    return f"{scale*row.recall:.{precision}f} [{scale*row.recall_ci_low:.{precision}f}, {scale*row.recall_ci_high:.{precision}f}]"


def write_detection_tex(rates, humans, out):
    lines = ["% Generated from verified detector decisions; thresholds unchanged.", r"\begin{table*}[t]", r"\centering",
             r"\caption{Baseline detection performance on the \FileDiff~context (Preliminary Question). Recall includes a 95\% Wilson interval; each row has 308 AI and 308 human descriptions. For each of the three pre-specified detectors, the human false-positive rate is 0.000 [0.000, 0.012] (0/308); specificity is 1.000.}",
             r"\label{tab:prelim_rq_results}", r"\footnotesize", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{llrrrr}",r"\toprule",
             r"\textbf{Model} & \textbf{Tool} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall [95\% CI]} & \textbf{F1} \\",r"\midrule"]
    for mi, model in enumerate(MODELS):
        for di, detector in enumerate(list(DETECTORS)[:3]):
            r = rates[(rates.detector == detector) & (rates.generator == model) & (rates.context_code == "P-3")].iloc[0]
            lines.append(" & ".join([SHORT[model] if di==0 else "",DETECTORS[detector],f"{r.accuracy:.3f}",f"{r.precision:.3f}",interval(r),f"{r.f1:.3f}"])+r" \\")
        if mi < 2: lines.append(r"\midrule")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table*}",""]
    # Verify the rounded zero-event interval quoted in the caption.
    z = humans[humans.detector != "binoculars"]
    assert z.fp.eq(0).all() and all(f"{v:.3f}"=="0.012" for v in z.fpr_ci_high)
    (out/"prelim_detection_intervals.tex").write_text("\n".join(lines))

    common = [r"\small",r"\setlength{\tabcolsep}{4pt}",r"\renewcommand{\arraystretch}{1.15}"]
    lines = ["% Generated; pointwise intervals, not simultaneous comparisons.",r"\begin{table}[htbp]",r"\centering",
             r"\caption{Human false-positive rates on the 308 unique human descriptions. Intervals are 95\% Wilson intervals; these baselines are reused, not independent samples for each model or context.}",
             r"\label{tab:human_false_positive_intervals}",*common,r"\begin{tabular}{lrrr}",r"\toprule",
             r"\textbf{Detector} & \textbf{False positives} & \textbf{Human $n$} & \textbf{FPR (\%) [95\% CI]} \\",r"\midrule"]
    for _, r in humans.iterrows():
        lines.append(f"{DETECTORS[r.detector]} & {r.fp} & {r.n_human} & {100*r.fpr:.1f} [{100*r.fpr_ci_low:.1f}, {100*r.fpr_ci_high:.1f}]"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}",""]
    lines += [r"\begin{table}[htbp]",r"\centering",
              r"\caption{Recall (\%) pooled over available contexts, with the existing 95\% PR-clustered percentile-bootstrap intervals (2,000 replicates). AI counts are 1,772 for Qwen and GPT, 1,778 for Llama, and 5,322 overall. Binoculars is post-hoc.}",
              r"\label{tab:pooled_recall_intervals}",*common,r"\begin{tabular}{lrrrr}",r"\toprule",
              r"\textbf{Detector} & \textbf{Qwen2.5-72B} & \textbf{GPT-5-mini} & \textbf{Llama-3.1-70B} & \textbf{All models} \\",r"\midrule"]
    for detector in DETECTORS:
        cells = []
        for model in MODELS+["pooled"]:
            r = rates[(rates.detector==detector)&(rates.generator==model)&(rates.context_code=="all")].iloc[0]
            cells.append(interval(r,True))
        lines.append(" & ".join([DETECTORS[detector]]+cells)+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}",r"\FloatBarrier",""]
    for mi, model in enumerate(MODELS):
        lines += [r"\begin{table}[htbp]",r"\centering",
                  r"\caption{Recall (\%) by context for "+SHORT[model]+r", with 95\% Wilson intervals. Each column retains the detector's recorded decision rule; Binoculars is post-hoc.}",
                  r"\label{tab:context_recall_intervals_"+str(mi+1)+"}",*common,r"\begin{tabular}{lrrrrr}",r"\toprule",
                  r"\textbf{Context} & \textbf{AI $n$} & \textbf{RoBERTa} & \textbf{RoBERTa-QA} & \textbf{ZeroGPT} & \textbf{Binoculars} \\",r"\midrule"]
        for code, display in CONTEXT.items():
            cells = []
            for detector in DETECTORS:
                r = rates[(rates.detector==detector)&(rates.generator==model)&(rates.context_code==code)].iloc[0]
                cells.append(interval(r,True))
            lines.append(" & ".join([display,str(int(r.n_ai))]+cells)+r" \\")
        lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}",""]
    lines += [r"\FloatBarrier",""]
    (out/"detector_uncertainty_tables.tex").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert socket.gethostname().split(".")[0] == "nokogiri", "Run this reporting calculation on Nokogiri"
    root = args.repo_root.resolve()
    rq = root/"phase3/analysis/outputs/corrected_rq23_v1"
    row_dir = rq/"notebooks/outputs/row_level"
    source_paths = [rq/"verification"/name for name in list(PINNED)[:2]]
    source_paths += [row_dir/name for name in dict.fromkeys(m[0] for m in METRICS)]
    source_paths += [args.b1_dir/name for name in ["all_detector_scores.csv","performance.csv","metadata.json","analysis_verification.json"]]
    hashes = {str(p): sha(p) for p in source_paths}
    for p in source_paths:
        if p.name in PINNED: assert hashes[str(p)] == PINNED[p.name], p
    meta = json.loads((args.b1_dir/"metadata.json").read_text())
    for name in ["all_detector_scores.csv","performance.csv"]:
        assert sha(args.b1_dir/name) == meta["output_hashes"][name]
    assert meta["bootstrap_replicates"] == 2000
    canonical = pd.read_csv(rq/"verification/canonical_descriptive_stats.csv")
    paired = pd.read_csv(rq/"verification/canonical_context_paired_stats.csv")
    scores = pd.read_csv(args.b1_dir/"all_detector_scores.csv")
    performance = pd.read_csv(args.b1_dir/"performance.csv")
    desc, baseline, desc_checks = build_descriptives(row_dir,canonical,paired,scores)
    rates, humans, rate_checks = build_intervals(scores,performance)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    desc.to_csv(out/"matched_textual_descriptives.csv",index=False)
    baseline.to_csv(out/"full_human_baseline_descriptives.csv",index=False)
    rates.to_csv(out/"detector_recall_intervals.csv",index=False)
    humans.to_csv(out/"human_false_positive_intervals.csv",index=False)
    write_descriptive_tex(desc,out)
    write_detection_tex(rates,humans,out)
    source_after = {str(p):sha(p) for p in source_paths}
    assert hashes == source_after, "Source changed during run"
    receipt = {"status":"PASS", "descriptive_comparisons":desc_checks, "detector_views":rate_checks,
               "source_files_unchanged":True,"new_significance_tests":0,"detector_calls":0,"generation_calls":0}
    (out/"verification.json").write_text(json.dumps(receipt,indent=2)+"\n")
    manifest = {"created_utc":datetime.now(timezone.utc).isoformat(),"host":socket.gethostname(),
                "server_commit":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
                "replication_source_commit":"b843f7e264090227bcb090c15acfcc694e4353ed",
                "python":platform.python_version(),"packages":{"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"statsmodels":statsmodels.__version__},
                "script_sha256":sha(__file__),"source_sha256":hashes,
                "descriptive_policy":"Human subset matched by PR to each model/context and restricted to complete pairs per metric, matching canonical paired tests; median and linear-interpolated Q1/Q3; no text feature recomputation",
                "interval_policy":"Pointwise 95% Wilson for cell recall and unique-human FPR; existing PR-cluster bootstrap for pooled recall; no new inferential comparisons",
                "output_sha256":{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in {"manifest.json", "independent_verification.json"}}}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"status":"PASS","matched_feature_comparisons":len(desc),"detector_views":len(rates),
                      "human_baselines":len(humans),"output_dir":str(out)},indent=2))


if __name__ == "__main__":
    main()
