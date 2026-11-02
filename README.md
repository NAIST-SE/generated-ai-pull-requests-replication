# Replication Package — AI-Generated Pull Request Descriptions: Weak Detection Despite Structural and Lexical Differences

This repository contains the public datasets, exact study code, precomputed results, and
supplementary material for:

> **AI-Generated Pull Request Descriptions: Weak Detection Despite Structural and Lexical Differences**
> Prepared for submission to Empirical Software Engineering (EMSE)

Current development is on `main`. The manuscript cites the reviewed snapshot
[`emse-review-2026-11-02`](https://github.com/NAIST-SE/generated-ai-pull-requests-replication/tree/emse-review-2026-11-02);
[`CITATION.cff`](CITATION.cff) records its citation metadata.

Historical large artifacts are archived on OSF at **https://osf.io/aq5rj/**. The corrected
corpus and the result packages used by the current manuscript are included directly in this
repository. See [`AUDIT_PROVENANCE.md`](AUDIT_PROVENANCE.md) for immutable source commits, hashes,
public-layout decisions, and exclusions. For the current RQ3 retrieval/calibration evidence,
see [`RQ3_PUBLICATION.md`](RQ3_PUBLICATION.md); older raw-cosine outputs are not the primary result.

## Current authoritative study state

The corrected study uses 308 human PR descriptions and 5,322 successful AI descriptions:

- P1--P4 were applicable to all 308 PRs;
- P5 (Linked Issue) and P6 (Multi-source context) were applicable only to the 273 PRs with a
  verified genuine GitHub issue;
- 12 P4 generations were unavailable (six Qwen and six GPT);
- the detector corpus contains 5,630 texts after adding the 308 human descriptions.

The primary public dataset is:

```text
datasets/temperature_1/run_3_corrected_detection_v1/detection_results.csv
```

The older `run_3_cleaned` derivative is retained for historical provenance only. Its P5/P6 rows
were built before the genuine-issue source audit and **must not be used for current manuscript
results**.

## Study questions and design

- **Prelim-RQ** — How accurately do the selected tools detect AI-generated PR descriptions?
- **RQ1** — How does generation context affect detector signal?
- **RQ2** — How does context shape structural, lexical, and generated-to-reference similarity?
- **RQ3** — How does PR-specific code-text association compare with the human baseline?

Models: `gpt-5-mini-2025-08-07`, `meta-llama/Llama-3.1-70B-Instruct`, and
`Qwen/Qwen2.5-72B-Instruct`. Temperature was explicitly 1 for Llama and Qwen. The GPT request
omitted temperature and used OpenAI's model-default decoding behaviour.

| ID | Configuration | Context supplied |
|---|---|---|
| P-1 | Repo Path | Repository name and affected paths |
| P-2 | PR Title | PR title |
| P-3 | Patch | Unified diff |
| P-4 | Full Code | Full contents of modified files, not the entire repository |
| P-5 | Linked Issue | Verified issue title and body |
| P-6 | Multi-source context | P1 + P2 + P3 + P5 |

### Reader-facing names and legacy labels

The manuscript uses **Patch** (P-3), **Full Code** (P-4), and **Multi-source context**
(P-6; **Multi-source** in compact tables and plots). Their legacy labels are
**File Diff**, **File Contents**, and **Without File Contents**, respectively.
The experimental conditions have not changed: P-6 combines repository path, PR title,
diff, and linked-issue text; it is not an ablation against an all-sources condition.

Frozen CSVs, prompts, filenames, notebooks, and historical outputs retain their original
labels and hashes. Join by P-1--P-6, not by display labels. The plot-only presentation
refresh is documented in [`CONTEXT_LABELS.md`](CONTEXT_LABELS.md); it uses cached
scores and does not rerun inference or statistical tests.

The manuscript's BM25 figure uses the separately versioned
`phase3/analysis/outputs/bm25_portrait_v2/bm25_retrieval_portrait.pdf`, with three
stacked model panels for print legibility. Its frozen scores and boxplot geometry
are unchanged; the previous horizontal figure remains in the historical bundle.

## Repository layout

```text
.
├── phase3/                         # Exact corrected-study source layout
│   ├── generation/                 # Corrected B4 generation and verification
│   ├── detection/corrected_rerun/  # Pinned three-detector implementation
│   └── analysis/
│       ├── corrected_rq23/         # RQ2 pipeline and original raw-cosine outputs
│       ├── rq3_mismatched_calibration/ # Current RQ3 retrieval/calibration extensions
│       ├── deep_audit/              # Corrected Prelim-RQ/RQ1 analyses
│       ├── b4_issue_source_audit/   # Genuine-issue provenance audit
│       ├── b4_linked_issue_reuse/   # Corrected B4 metrics, tests, and verifier
│       ├── generation_session_*     # Session diagnostics and robustness bridge
│       └── outputs/                 # Verified precomputed result packages
├── datasets/
│   └── temperature_1/
│       ├── run_3_corrected_detection_v1/  # Authoritative public corpus
│       ├── run_3_cleaned/                 # Historical, superseded
│       └── run_3/                         # Frozen historical detector export
├── analysis/                       # Historical flat-layout analyses retained for provenance
├── collection/                     # PR mining and sampling workflow
├── docs/                           # Client-side coding and inspection tools
└── supplementary/                  # Corrected supplementary results
```

`phase3/datasets` is a relative symlink to the root `datasets/` directory. This retains the exact
paths used by the verified experiment code without duplicating the public data. The older flat
`analysis/` tree remains available to audit the pre-B4 revision history, but its
`analysis/outputs/deep_audit/` results are not the current source of truth.

## Included datasets

| File | Status and purpose |
|---|---|
| `datasets/selected_prs_sampled.csv` | Final 308 human PR sample |
| `datasets/selected_prs_with_diffs_and_file_contents.csv` | PR diff and file-content source |
| `datasets/selected_prs_with_related_issues.csv` | Historical fetched-artifact snapshot used by the source audit |
| `datasets/temperature_1/run_3_corrected_detection_v1/corpus.csv` | Corrected 5,630-text corpus before detector columns are joined |
| `datasets/temperature_1/run_3_corrected_detection_v1/detection_results.csv` | Corrected corpus with all three finalized detector outputs |
| `datasets/temperature_1/run_3_corrected_detection_v1/corpus_manifest.json` | Source hashes, counts, and generation provenance |
| `datasets/temperature_1/run_3_corrected_detection_v1/completion.json` | Detector finalization/completion record |
| `datasets/temperature_1/b4-corrected-v3/completion.json` | Corrected-generation hash and count manifest; raw generation CSVs are not committed here |
| `datasets/temperature_1/run_3_cleaned/` | Superseded pre-B4 derivative, retained only for audit history |

The corrected corpus records public PR descriptions and generated descriptions, source hashes,
generation metadata, and detector results. Operational API logs, access tokens, local caches,
and Python bytecode are intentionally excluded.

## Authoritative result packages

- `phase3/analysis/outputs/corrected_detection_v1/` — corrected Prelim-RQ and RQ1 tables,
  contrasts, figure, and independent verification;
- `phase3/analysis/outputs/corrected_rq23_v1/` — executed RQ2/RQ3 notebooks, row-level numeric
  exports, figures, reconstructed verification tables, and completion manifest; RQ3 raw
  cosine is retained descriptively, with current inference in the following packages;
- `phase3/analysis/outputs/rq3_mismatched_calibration_v1/` — matched/nonmatched cosine,
  correct-diff ranks, margin comparisons, and verification;
- `phase3/analysis/outputs/rq3_embedding_postprocess_sensitivity_v1/` — all centering and
  leading-component-removal sensitivity results;
- `phase3/analysis/outputs/rq3_retrieval_inference_v1/` — primary paired reciprocal-rank
  comparisons, MRR intervals, and verification;
- `phase3/analysis/outputs/tier_b/b4-issue-source-audit-v1/` — verified 273-PR genuine-issue
  population and regeneration manifest;
- `phase3/analysis/outputs/tier_b/b4-linked-issue-reuse-v2/` — corrected prompt-exact
  linked-issue reuse scores, five predeclared statistical families, text-bearing inspection
  artifacts, completion hashes, and independent verification;
- `phase3/analysis/outputs/tier_b/b1-modern-detectors-v1/` — post-hoc Binoculars record-level
  scores, aggregate results, sanitized source/model/environment provenance, and independent
  verification;
- `phase3/analysis/outputs/tier_b/generation-session-diagnostic-v1/` — raw-text-free diagnostic
  comparisons only (the newly generated descriptions are deliberately excluded);
- `phase3/analysis/outputs/tier_b/generation-session-outcome-sensitivity-v1/` — verified
  raw-text-free outcome sensitivity results;
- `phase3/analysis/outputs/tier_b/generation-session-six-context-bridge-v1/` — verified
  six-context headline bridge;
- `phase3/analysis/outputs/omnibus_effect_sizes_v1/` — Kendall's W and Cochran's Q eta^2_Q
  derived from the stored omnibus test statistics, plus the McNemar paired-difference range
  (script: `phase3/analysis/effect_sizes/derive_omnibus_effect_sizes.py`);
- `phase3/analysis/outputs/descriptive_reporting_v1/` — matched human/AI medians and quartiles,
  cell-level recall and unique-human false-positive-rate Wilson intervals, and reused
  PR-clustered pooled recall intervals. This additive reporting package does not change
  detector decisions or prior tests. Its source, reproduction instructions, and independent
  verifier are in `phase3/analysis/descriptive_reporting_v1/`.

The old `b4-linked-issue-reuse-v1` execution is not published because it predates the
genuine-issue audit. The published corrected v2 analysis uses the 273-PR prompt-exact manifest,
4,902 AI-generated descriptions, and 273 historical human descriptions. Under P-5, median
normalized exact five-token containment is 1.20%--1.96% across generators and exceeds every
issue-unexposed control in the 12 predeclared corrected contrasts. P-6 transfer is weaker and
sparser. These results support limited issue-text transfer, not plagiarism, wholesale copying,
description quality, or a causal effect of adding issue text. The post-hoc B1 analysis adds
Binoculars to the three pre-specified detectors on the same 5,630-description corpus. Binoculars
has pooled recall 0.290, specificity 0.987, and strongly generator-dependent recall
(0.015--0.597); it does not overturn the conclusion that detector behaviour depends on generator
and context. Fast-DetectGPT was not run and remains outside this study unless an approved protocol amendment adds it.
B2 human-coding materials will be added separately after their publication boundary is approved.

## Reproducing the corrected workflow

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt_tab')"
```

MTLD uses a separate Python 3.12 environment with the TAALED installation recipe documented in
`requirements.txt`. The exact study launchers fail closed outside the configured Nokogiri server;
synthetic/unit tests and integrity inspection can run elsewhere.

### Corrected issue-source audit and generation

The public genuine-issue manifest is already available under the Tier-B output directory. A new
server-side generation requires explicit credentials for the recorded providers:

```bash
export OPENROUTER_API_KEY="..."  # Llama via OpenRouter/CoreWeave, BF16
export HF_TOKEN="..."            # Qwen via Hugging Face/Novita, BF16
export OPENAI_API_KEY="..."       # GPT-5-mini
bash run_generation.sh
```

The final corrected generation CSVs are not committed because they duplicate the descriptions in
the corrected public corpus while also retaining large stored prompts. Their hashes and counts
remain in `b4-corrected-v3/completion.json` and the corrected corpus manifest.

### Corrected detector rerun

The published `detection_results.csv` is already complete. To inspect or repeat the pinned
pipeline on the configured server:

```bash
bash run_corrected_detection_pipeline.sh --preflight
bash run_corrected_detection_pipeline.sh
```

ZeroGPT requires a token file supplied at runtime. Token values and API logs are never part of the
replication package. The two RoBERTa detectors use the pinned revisions documented in
`phase3/detection/corrected_rerun/README.md`.

### Post-hoc Binoculars robustness check

The exact B1 analysis implementation is under `phase3/detection/b1_modern/`, and the precomputed
record-level scores and aggregate outputs are under
`phase3/analysis/outputs/tier_b/b1-modern-detectors-v1/`. The output
README documents the Apple-MPS execution route, frozen Falcon revisions, published threshold,
analysis policy, result summary, and publication boundary. The package deliberately excludes the
blinded handoff, private re-identification map, operational logs, and returned archive. The
published score table contains no raw text, and its detector labels must not be interpreted as
evidence of real-world PR authorship. Fast-DetectGPT code is retained only as a dormant
contingency; it is not part of the reported analysis.

### Corrected RQ2/RQ3 analyses

```bash
cd phase3/analysis
bash corrected_rq23/run_pipeline.sh --preflight
bash corrected_rq23/run_pipeline.sh
```

The full execution is GPU/server work. Precomputed notebooks and outputs are included so the
reported values can be audited without rerunning the models.

### Corrected linked-issue reuse analysis

Synthetic tests may run on any host:

```bash
cd phase3/analysis/b4_linked_issue_reuse
python -m unittest -v test_b4_metrics.py test_b4_pipeline_synthetic.py
```

Study-data execution is fail-closed to the configured Nokogiri server. The curated public package
includes the exact source code and complete verified v2 outputs; the raw corrected generation
exports remain excluded because their descriptions already appear in the corrected public corpus
and their large stored prompts are not needed to audit the published scores. Run
`verify_public_package.py` to validate all 20 B4 output hashes and the independent verification
record without recomputing study metrics.

### Integrity-only public-package check

```bash
python verify_public_package.py
```

This check does not execute study analyses. It validates file presence, hashes, corrected corpus
counts, completion statuses, excluded operational artifacts, and the historical/current boundary.

## Supplementary results

The current manuscript combines all structural/lexical effect-size results with
ROUGE-L F1 and baseline-rescaled BERTScore F1 medians in one grouped RQ2 table.
It retains all 108 effect sizes and their significance markers, sample-size
qualifications, and correction details, alongside 36 descriptive similarity medians.
The supplement provides the corresponding metric distributions.

Supplementary Table S1 summarizes dataset filtering, sampled artifacts, and generation
coverage. Its source, `supplementary/dataset_summary.tex`, preserves the original
manuscript table, including all numerical cells and its coverage footnote.

Supplementary Table S2 reports detector input modes and truncation counts across
5,630 descriptions. The original main-paper table, including its successful-scoring
note, is preserved in `supplementary/detector_input_handling.tex`.

Supplementary Table S9 contains the matched human/AI medians and quartiles for FKGL,
TTR, bidirectional MTLD, and lexical density, with MTLD complete-pair counts explicit.
Its two portrait parts share one table number. The table source is
`supplementary/rq2_feature_descriptives.tex`, copied without changing any cells from
the verified manuscript table. Full word- and sentence-count summaries remain in the
reporting outputs.

Supplementary Figure S1 shows word-count distributions; S2--S5 show FKGL, TTR,
bidirectional MTLD, and lexical-density distributions. ROUGE-L is Figure S6,
baseline-rescaled BERTScore is Figure S7, and fixed-corpus BM25 retrieval is Figure S8.
The BERTScore plot accompanies its numerical interpretation, baseline-rescaling
qualifications, and truncation limits in the supplement. All plots reuse the
existing verified assets.

Human false-positive-rate intervals are in Supplementary Table S3, pooled recall
intervals for all detectors in S4, and per-context recall intervals in S5--S7.
Pooled Binoculars results are in S8. Linked-issue reuse tables follow the relocated
descriptive table. The unchanged example Repo Path prompt is Supplementary Listing S1
(`supplementary/generation_prompt.tex`).

Compile without writing generated files into the repository:

```bash
supp_build_dir="$(mktemp -d /tmp/llm-prs-supplement.XXXXXX)"
cd supplementary
tectonic -X compile --outdir "$supp_build_dir" --keep-logs supplementary.tex
```

## Interactive tools

`docs/index.html` is the description-only blind-coding interface; `docs/inspector.html` is the
filterable corpus inspector. Both run client-side through GitHub Pages.

## Citation

Use [`CITATION.cff`](CITATION.cff) to cite the replication package. The reviewed
version is [`emse-review-2026-11-02`](https://github.com/NAIST-SE/generated-ai-pull-requests-replication/tree/emse-review-2026-11-02).
This tag identifies the corrected dataset, study code, precomputed results, and
supplementary sources used by the manuscript. The historical OSF archive predates
this corrected-study version.
