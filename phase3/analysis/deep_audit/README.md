# Deep-audit remediation analyses

These scripts implement the post-audit data-integrity and statistical repairs. They are
authored locally but **must only be executed on `nokogiri`**, in accordance with
`phase3/analysis/AGENTS.md`.

## Non-negotiable execution rules

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
- The wrapper and each result-bearing standalone program fail closed when the current
  hostname is not `nokogiri`.

## Intended order

From the repository root on the server:

```bash
bash phase3/analysis/deep_audit/run_server_pipeline.sh
```

The wrapper prepares the cleaned derivative, runs the three standalone repairs, executes
the RQ2/RQ3 notebooks through the kernel-aware driver, and assembles the figure PDF. It
refuses to run anywhere except a host whose short name is `nokogiri`.

The default cleaned dataset is:

```text
phase3/datasets/temperature_1/run_3_cleaned/detection_results.csv
```

The default new output directory is:

```text
phase3/analysis/outputs/deep_audit/
```

## Corrected detector rerun

The finalized all-success detector corpus is analysed separately from the historical
`run_3_cleaned` derivative. On `nokogiri`, run:

```bash
bash phase3/analysis/deep_audit/run_corrected_detection_pipeline.sh
```

This reads the hash-pinned
`phase3/datasets/temperature_1/run_3_corrected_detection_v1/detection_results.csv` and
writes only to `phase3/analysis/outputs/corrected_detection_v1/`. The existing descriptive,
Preliminary-RQ, and RQ1 calculators are invoked with explicit input/output arguments. The
corrected verifier then independently recomputes every descriptive cell, Cochran test,
McNemar test, pooled Friedman test, and model-specific paired contrast. It also asserts
that the historical operational and successful-only views are identical, as every
corrected detector call succeeded. Do not use the corrected outputs in the manuscript
unless `verification_report.json` reports `PASS` and the generated PDF has passed visual
review.

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

The reviewed RQ1 candidate is
`phase3/analysis/outputs/figures_pdf/rq1_probability_boxplots_paired.pdf`. Copy it to
`llm-in-prs-research-writing/figures/pdf/` only after completing the checks above. The
notebook deliberately does not write into the manuscript repository.
