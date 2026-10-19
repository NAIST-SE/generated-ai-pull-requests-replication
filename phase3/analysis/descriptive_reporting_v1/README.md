# Descriptive values and detection uncertainty

Additive reporting package, 2026-10-01. This reports human/AI textual-feature
values and uncertainty around detection rates. It does not generate text,
extract features, call detectors, alter thresholds, invert decisions, or run
new significance tests.

## Inputs and execution

Executed on Nokogiri from `/mnt/ext-hdd1/ayo/llm-in-prs`, branch
`revision/co-author-review`, commit `492923572940e19611ebe612d6c689c2b11877f1`.
The interpreter was `/home/ayo-b/.pyenv/versions/3.12.9/bin/python` (`iwadi-main`).
Exact package versions, source hashes, script hash, and output hashes are in
`outputs/descriptive_reporting_v1/manifest.json` relative to `phase3/analysis/`.

- Textual inputs: frozen numeric rows and canonical verification tables in
  `outputs/corrected_rq23_v1/`. No text features were recomputed.
- Detector inputs: `all_detector_scores.csv`, `performance.csv`, `metadata.json`,
  and `analysis_verification.json` in `outputs/tier_b/b1-modern-detectors-v1/`.
  These were supplied from replication commit
  `b843f7e264090227bcb090c15acfcc694e4353ed`. Nokogiri lacked that result directory,
  so verified copies were staged in this script directory's `b1_verified_inputs/`.
  The staging directory is not needed in the replication package, where the
  original files already exist. Both locations contain identical bytes.

From the repository root **on Nokogiri**, with the existing inputs present:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/ayo-b/.pyenv/versions/3.12.9/bin/python \
  phase3/analysis/descriptive_reporting_v1/analyse_reporting.py \
  --repo-root "$PWD" \
  --b1-dir phase3/analysis/outputs/tier_b/b1-modern-detectors-v1 \
  --output-dir phase3/analysis/outputs/descriptive_reporting_v1

PYTHONDONTWRITEBYTECODE=1 /home/ayo-b/.pyenv/versions/3.12.9/bin/python \
  phase3/analysis/descriptive_reporting_v1/verify_reporting.py \
  --repo-root "$PWD" \
  --b1-dir phase3/analysis/outputs/tier_b/b1-modern-detectors-v1 \
  --output-dir phase3/analysis/outputs/descriptive_reporting_v1
```

During the recorded execution, verified B1 inputs were staged on Nokogiri because
that server checkout lacked the result directory. The published manifest maps
their unchanged hashes to the canonical public-package paths; it retains the
server timestamp, commit, environment, and original output hashes. The manifest
and independent receipt were reissued solely to make those source paths portable.
Rerunning the producer on Nokogiri replaces only this additive package's outputs;
original analysis outputs remain read-only.

## Descriptive reporting

- Six features, three generators, six configurations: 108 matched summaries.
- Each AI description is joined to its human counterpart by PR ID.
- Each feature uses complete pairs, exactly as in the existing paired tests.
  Medians and linearly interpolated Q1/Q3 describe these pairs; they are not
  medians of paired differences, effect sizes, or new significance tests.
- Available generations are 308 per cell, except Full Code for Qwen/GPT (302)
  and Linked Issue/Multi-source for every generator (273).
- MTLD has pre-existing missing values. The full human baseline has 297 valid
  values; matched MTLD counts range from 216 to 297. Its table reports its own
  sample sizes rather than implying all metrics use the same count. Nothing is
  filled in or recomputed. Other features are complete within available cells.
- Marginal valid-value summaries were checked against
  `canonical_descriptive_stats.csv`; matched medians and counts were checked
  against `canonical_context_paired_stats.csv`. These are different checks
  because marginal MTLD summaries need not describe the same pairs.

## Detection uncertainty

- Counts come from unchanged recorded binary decisions for all four detectors.
- Within a detector/model/context, one AI description per PR permits a binomial
  recall interval. We use two-sided 95% Wilson intervals, including zero events.
- Human false-positive intervals use 308 unique human descriptions, once per
  detector. The baseline is not replicated as additional independent evidence.
- Pooled recall reuses the existing 2,000-replicate PR-clustered percentile
  bootstrap intervals in `performance.csv`, preserving dependence across models
  and contexts from the same PR. No bootstrap is rerun.
- All intervals are pointwise, not simultaneous. Existing corrected paired tests
  are unchanged; interval overlap is not used as a significance test. These
  intervals do not cover between-project or repeated-generation variability.
- No finite-population correction is applied. Interpretation remains conditional
  on the sampled project, available configurations, models, and decision rules.
- Original specificity bootstrap intervals remain in the original package.
  The new standalone human FPR reporting consistently uses Wilson intervals:
  zero of 308 is 0.0% [0.0%, 1.2%]; four of 308 is 1.3% [0.5%, 3.3%].

Wilson formula and worked-example reference:
[NIST Dataplot](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/propconf.htm).

## Outputs and publication mapping

| Output | Purpose |
| --- | --- |
| `matched_textual_descriptives.csv` | Paired medians, quartiles, means, counts, exclusions, PR-set hashes |
| `full_human_baseline_descriptives.csv` | Full valid human baseline, separate from matched subsets |
| `detector_recall_intervals.csv` | 72 context-cell and 16 pooled recall summaries; 88 total |
| `human_false_positive_intervals.csv` | Four unique-human baseline intervals |
| `rq2_structural_descriptives.tex` | Archived full structural/readability table; numerical source for the publication view |
| `rq2_lexical_descriptives.tex` | Archived full lexical table, including MTLD counts; source for the publication view |
| `prelim_detection_intervals.tex` | Main-paper baseline table, original estimates plus recall CIs |
| `detector_uncertainty_tables.tex` | Supplementary Tables S1-S5; prior Binoculars table becomes S6 |
| `verification.json` | Producer consistency checks against canonical outputs |
| `independent_verification.json` | Independent reconstruction and formula verification |
| `manifest.json` | Execution provenance and checksums |

The active main-paper inputs are copied to `tables/` in the manuscript
workspace; the supplement includes
`supplementary/detector_uncertainty_tables.tex`. Existing RQ2 effect-size tables
and all original statistical tests remain intact. No RQ3, poster, pilot, or
public-release changes are included.

The independent verifier uses only the Python standard library, not the
producer's functions. It reconstructs all 108 summaries, checks all 92 rate
intervals, verifies the NIST worked example and zero/all-event boundaries, and
checks source/output hashes. Its receipt binds to the completed manifest; the
manifest excludes that later receipt to avoid a circular checksum dependency.

## Compact publication view (October 1)

The manuscript retains its word-count boxplot and combines FKGL, TTR, MTLD, and
lexical density in one descriptive table, `rq2_feature_descriptives.tex`. Word
and sentence counts are omitted from this table, not from the numerical package
or the unchanged effect-size tables. The illustrative PR quotation and its
interpretation have been removed from the manuscript; no general prose
tightening was performed.

A subsequent, separately approved length pass removes only the main-paper
BERTScore plot and points to its existing Supplementary Figure S6 (ROUGE is S5).
Its main-text finding, magnitude, and qualifications remain unchanged. This
record describes the source of the current manuscript table, not a new analysis.

`format_descriptive_table.py` copies the 72 retained H/A summary cells and pair
counts verbatim from the two checksum-pinned original TeX tables. It calculates
no statistics and leaves all server-generated outputs and their receipts intact.
The table font size is unchanged. The formatter can regenerate the manuscript
view, but the generated view and its presentation manifest are kept in the
manuscript workspace, not duplicated here.

From the replication repository, regenerate only the publication view with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  phase3/analysis/descriptive_reporting_v1/format_descriptive_table.py \
  --source-dir phase3/analysis/outputs/descriptive_reporting_v1 \
  --output-dir /path/to/manuscript-table-output
```

This formatting-only command can run locally; it does not rerun the statistical
analysis. The generated publication view is not duplicated in this replication package.
