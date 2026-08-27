# Deep-audit remediation analyses

These scripts implement the post-audit data-integrity and statistical repairs used by
the revised manuscript. The study team executed them only on `nokogiri`; the checked-in
results are exact copies of that run. External reproducers may run the public copy on a
different host by setting `LLM_PRS_ALLOW_REPLICATION_RUN=1` explicitly.

## Study-team execution record

- Server repository: `/mnt/ext-hdd1/ayo/llm-in-prs`
- Branch: `revision/co-author-review`
- Do not overwrite `phase3/datasets/temperature_1/run_3/`.
- Use the named `iwadi-main` kernel/interpreter for these scripts.
- The wrapper defaults standalone programs to
  `/home/ayo-b/.pyenv/versions/3.12.9/bin/python`; never use the server's Python 2.7
  `python` command.
- Use `iwadi-py312` only for the authoritative MTLD and lexical-density notebooks.
- Preload `/usr/lib/x86_64-linux-gnu/libcuda.so.1` before transformer notebooks.
- Do not copy outputs into the manuscript until the checks in this document pass.
- The original wrapper and each result-bearing standalone program failed closed when the
  hostname was not `nokogiri`. In this public copy, standalone programs retain that
  default but accept the explicit external-replication opt-in above.

## External replication from this checkout

The public repository uses `analysis/` and `datasets/` at its root rather than the
internal repository's `phase3/` prefix. Run from the repository root and pass the paths
explicitly:

```bash
export LLM_PRS_ALLOW_REPLICATION_RUN=1

python analysis/deep_audit/analysis_prelim_paired.py \
  --input datasets/temperature_1/run_3_cleaned/detection_results.csv \
  --output-dir analysis/outputs/deep_audit/prelim

python analysis/deep_audit/analysis_detection_descriptive.py \
  --input datasets/temperature_1/run_3_cleaned/detection_results.csv \
  --output-dir analysis/outputs/deep_audit/detection_descriptive

python analysis/deep_audit/analysis_rq1_cleaned.py \
  --input datasets/temperature_1/run_3_cleaned/detection_results.csv \
  --output-dir analysis/outputs/deep_audit/rq1

python analysis/deep_audit/analysis_rq2_bm25_retrieval.py \
  --input datasets/temperature_1/run_3_cleaned/detection_results.csv \
  --output-dir analysis/outputs/deep_audit/bm25_retrieval \
  --figure-output analysis/outputs/figures_pdf/bm25_retrieval.pdf

python analysis/deep_audit/verify_outputs.py \
  --clean-dir datasets/temperature_1/run_3_cleaned \
  --analysis-dir analysis
```

These commands recompute results. The study team's copy was not recomputed locally while
preparing this package. The precomputed files under `analysis/outputs/` can be inspected
without setting the opt-in variable.

To rebuild `run_3_cleaned` itself, first download the three generation CSVs from OSF into
`datasets/temperature_1/run_3/`, then run:

```bash
export LLM_PRS_ALLOW_REPLICATION_RUN=1
python analysis/deep_audit/prepare_cleaned_run3.py \
  --run-dir datasets/temperature_1/run_3 \
  --detection-csv datasets/temperature_1/run_3/detection_results.csv \
  --output-dir datasets/temperature_1/run_3_cleaned \
  --overwrite
```

## Original server execution order

From the repository root on the server:

```bash
bash phase3/analysis/deep_audit/run_server_pipeline.sh
```

The internal wrapper prepared the cleaned derivative, ran the standalone repairs, executed
the RQ2/RQ3 notebooks through the kernel-aware driver, and assembled the figure PDF. It
is retained as provenance for the actual server run and is not the public-layout driver.

The default cleaned dataset in the internal layout was:

```text
phase3/datasets/temperature_1/run_3_cleaned/detection_results.csv
```

The default output directory in the internal layout was:

```text
phase3/analysis/outputs/deep_audit/
```

## Expected integrity gate

`prepare_cleaned_run3.py` fails unless all of the following hold:

- 5,544 planned generation rows;
- 5,526 successful generation rows;
- 18 excluded generation rows;
- 5,528 AI rows in the frozen detection input;
- exactly two excluded generation rows had entered the frozen detection input;
- 5,526 AI rows and 308 human rows in the cleaned detection derivative.

The script preserves every raw file, writes SHA-256 hashes into
`integrity_summary.json`, and refuses to overwrite an existing cleaned directory unless
`--overwrite` is explicitly supplied.

## Statistical contracts

### Preliminary detector comparison

- PR is the paired unit.
- Cochran's Q is the per-detector omnibus across the three generators.
- Exact McNemar tests are the paired post-hoc comparisons.
- BH-FDR is applied once over the nine post-hoc tests.
- The primary `operational` view treats a detector failure as not flagged, matching the
  deployed outcome. A `successful_only` sensitivity view retains only PRs successfully
  scored for all three generators by that detector.
- `analysis_detection_descriptive.py` separately regenerates the operational
  confusion-matrix metrics used by the Preliminary File-Diff and RQ1 F1-extrema tables;
  it does not substitute an independent significance test for the paired analysis.

### RQ1

- Friedman uses the same PRs across all six contexts.
- Per PR/context detector probability is averaged across the available generators.
- `rq1_operational_vs_successful_only.csv` compares all 135 model-specific paired
  contrasts one-to-one across the two failure-handling views, including their sample
  sizes, rank-biserial directions, and adjusted-significance decisions.
- The exact complete-case `n` is reported per detector.
- Model-specific context contrasts use paired PR-level values. They are descriptive
  follow-ups, not post-hoc tests of the pooled-across-model Friedman estimand.

### RQ2 BM25 retrieval

- One fixed BM25 index contains the 308 human PR descriptions.
- Every AI description queries that same index.
- Its corresponding human description is the sole relevant document.
- Primary outcomes are rank, reciprocal rank, MRR, and Recall@1/5/10.
- No human self-comparison baseline and no per-cell min-max normalisation are used.
- Friedman is computed per model across the six contexts using reciprocal rank.
- The predeclared post-hoc family is Linked Issue versus each of the other five contexts
  for each generator (15 paired comparisons, one BH-FDR family).

## Required review before manuscript use

1. Inspect `integrity_summary.json` and `generation_exclusions.csv`.
2. Confirm every output's `input_sha256` matches the cleaned dataset.
3. Check sample sizes and missingness by model/context.
4. Compare descriptive values against the previous frozen outputs and explain every
   change caused by the two exclusions or the redesigned estimand.
5. Independently review the paired-test family definitions and sign conventions.
6. Only then update manuscript macros, tables, figures, and prose.

The reviewed RQ1 figure is
`analysis/outputs/figures_pdf/rq1_probability_boxplots_paired.pdf`. Its SHA-256 is
`2b35a4ab09741b43f761dcd0148673d6f60619a9e97cc47f29c721c0372a266d`.
