# Replication Package — Context, Semantics, and Alignment in AI-Generated Pull Request Descriptions

This repository contains the code, analysis notebooks, and pre-computed results for the paper:

> **Context, Semantics, and Alignment in AI-Generated Pull Request Descriptions**
> Under review — Transactions on Software Engineering (TSE)

Large datasets are hosted on OSF: **https://osf.io/aq5rj/**

The cleaned-data audit, server execution commit, and artifact hashes are documented in
[`AUDIT_PROVENANCE.md`](AUDIT_PROVENANCE.md).

---

## Study Overview

The paper investigates AI-generated pull-request descriptions produced under six different context configurations and answers four research questions:

- **Prelim-RQ** — How accurate are existing tools at detecting AI-generated PR descriptions?
- **RQ1** — How does generation context affect the *distinguishability* of AI-generated PR descriptions?
- **RQ2** — How does generation context shape the *textual similarity* between AI-generated and human-written PR descriptions? (structural, lexical, and semantic metrics)
- **RQ3** — To what extent does the *code-text alignment* of AI-generated PR descriptions resemble that of human-written ones?

**Models:** `gpt-5-mini-2025-08-07` (proprietary), `meta-llama/Llama-3.1-70B-Instruct`, and `Qwen/Qwen2.5-72B-Instruct`. Temperature was explicitly set to 1 for Llama and Qwen. The GPT-5-mini request omitted the temperature parameter and therefore used the model's API-default behaviour.

**Detectors:** ZeroGPT (commercial), ChatGPT Detector RoBERTa, ChatGPT-QA Detector RoBERTa (both from Guo et al.).

**Context configurations** used when prompting each model:

| ID | Name | Context provided to the LLM |
|---|---|---|
| P-1 | Repo Path | Repository name + affected file paths |
| P-2 | PR Title | PR title only |
| P-3 | File Diff | Unified diff (baseline) |
| P-4 | File Contents | Full file contents |
| P-5 | Linked Issue | Linked GitHub issue title and body |
| P-6 | Without File Contents | All of P-1 + P-2 + P-3 + P-5 combined |

---

## Repository Structure

```
.
├── collection/          # Data collection — PR mining, diff/file fetching, issue linking
├── generation/          # LLM generation scripts (GPT-5-mini, Llama-3.1-70B, Qwen2.5-72B)
├── detection/           # AI-text detection pipeline (ZeroGPT, RoBERTa, RoBERTa-QA)
├── analysis/            # Analysis notebooks, audited scripts, results, and figures
│   ├── deep_audit/      # Cleaned-data integrity, paired tests, BM25 retrieval, verification
│   ├── outputs/deep_audit/  # Authoritative Prelim-RQ/RQ1/BM25 result families
│   └── sensitivity_successful_only/  # Superseded historical analysis; do not cite
├── supplementary/       # Standalone supplementary results and publication figures
├── docs/                # Browser-based tools: blind-coding interface + PR inspector
├── data_loader.py       # Shared dataset loader used across scripts and notebooks
├── requirements.txt     # Python dependencies
├── run_generation.sh    # Parallel generation entry point
├── run_detection_pipeline.sh  # Detection entry point
└── verify_detection.py  # Quick integrity check on detection_results.csv
```

---

## Supplementary Results

The paper's supplementary results are published in
[`supplementary/supplementary.tex`](supplementary/supplementary.tex) rather than embedded in
the manuscript. The document includes additional Preliminary-Question detail, RQ2
structural and lexical distributions, and the audited ROUGE, BERTScore, and fixed-corpus
BM25 interpretation.

Compile it without writing build artifacts into the repository:

```bash
supp_build_dir="$(mktemp -d /tmp/llm-prs-supplement.XXXXXX)"
cd supplementary
tectonic -X compile --outdir "$supp_build_dir" --keep-logs supplementary.tex
```

---

## Datasets

Most datasets are included in this repository under `datasets/`. The large per-model generation CSVs are hosted on OSF (**https://osf.io/aq5rj/**) and are only needed if you want to re-run the detection pipeline from scratch (Step 3).

### Included in this repo

| File | Needed for | Description |
|---|---|---|
| `datasets/jabref-prs.csv` | Step 1 (optional) | Raw JabRef PR dump mined from GitHub |
| `datasets/jabref_prs_additional.csv` | Step 1 (optional) | Supplementary PRs added in a later collection pass |
| `datasets/selected_prs.csv` | Step 1 output | Filtered PRs before final sampling |
| `datasets/selected_prs_sampled.csv` | Steps 2–4 | 308 human-written PR descriptions (final sample) |
| `datasets/selected_prs_with_diffs_and_file_contents.csv` | Step 2 | PR diffs and file contents per PR |
| `datasets/selected_prs_with_related_issues.csv` | Step 2 | Linked GitHub issues per PR |
| `datasets/temperature_1/run_3/detection_results.csv` | Audit provenance | Frozen detector export: 5,528 AI rows + 308 human rows. Preserved unchanged; not the revised analysis source of truth. |
| `datasets/temperature_1/run_3_cleaned/detection_results.csv` | Step 4 | Audited derivative: 5,526 successful AI rows + 308 human rows — **revised primary results file**. |
| `datasets/temperature_1/run_3_cleaned/generation_exclusions.csv` | Integrity audit | The 18 excluded File-Contents generation attempts, including the two serialized error responses present in the frozen detector export. |
| `datasets/temperature_1/run_3_cleaned/integrity_summary.json` | Integrity audit | Expected/observed counts and SHA-256 hashes for frozen inputs and audited outputs. |
| `datasets/temperature_1/run_2/detection_results.csv` | `compare_runs.ipynb` | Detection scores from an independent run (consistency validation) |
| `datasets/temperature_1/run_1/detection_results_run_1.csv` | `compare_runs.ipynb` | Detection scores from an independent run (consistency validation) |

### On OSF only (large generation outputs)

Download from OSF and place under `datasets/temperature_1/run_3/` to re-run detection:

| File | Size | Description |
|---|---|---|
| `temperature_1/run_3/chatgpt_results_exp_chatgpt_1771584462.csv` | 46 MB | GPT-5-mini generated descriptions |
| `temperature_1/run_3/meta-llama_llama-3.1-70b-instruct_1771584467.csv` | 37 MB | Llama-3.1-70B generated descriptions |
| `temperature_1/run_3/Qwen_Qwen2.5-72B-Instruct_1771584467.csv` | 37 MB | Qwen2.5-72B generated descriptions |

> **Analysis only:** `datasets/temperature_1/run_3_cleaned/detection_results.csv` is already included. Use it for all revised Prelim-RQ, RQ1, RQ2, RQ3, and sensitivity analyses.

---

## Reproducing the Study

### Prerequisites

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt_tab')"
```

For GPU-based generation (Llama, Qwen) on an HPC cluster, see `HPC_GUIDE.md`.

### Step 1 — Data Collection (optional; datasets on OSF)

```bash
# Authoritative 308-PR sampling workflow (fixed seed 23)
jupyter notebook collection/sampling_and_distribution.ipynb

# Fetch PR file diffs and contents
python collection/fetch_pr_files.py

# Fetch linked GitHub issues
python collection/fetch_pr_issues.py
```

> Requires a GitHub token in `collection/env/github_token.txt`. The standalone `collection/filter_and_sample_prs.py` script is an earlier 150-record prototype and was not used to construct the manuscript dataset.

### Step 2 — Generation

The design planned 308 PRs × 6 context configurations × 3 models = 5,544 generations. The final audited dataset contains 5,526 successful generations; 18 File-Contents attempts failed, six per model.

```bash
export OPENROUTER_API_KEY="..."   # for Llama and Qwen via OpenRouter / HuggingFace router
export OPENAI_API_KEY="..."       # for GPT-5-mini (temperature omitted)
bash run_generation.sh
```

> The File Contents context (P-4) can exceed provider token limits. See `generation/P4_TRUNCATION_GUIDE.md`.

### Step 3 — Detection

```bash
# Merge generated CSVs into datasets/generated_descriptions.csv, then run three detectors
bash run_detection_pipeline.sh
```

> ZeroGPT requires tokens in `detection/env/zero-gpt-tokens.txt` (one token per line).

Verify results after detection:
```bash
python verify_detection.py
```

### Step 4 — Revised analysis

Open the notebooks in `analysis/` with Jupyter **from the repository root** — this ensures relative paths to `datasets/` resolve correctly:

```bash
cd generated-ai-pull-requests-replication
jupyter notebook
```

To run an individual notebook non-interactively:
```bash
python -m papermill analysis/<notebook>.ipynb analysis/<notebook>.ipynb --cwd analysis
```

| Notebook | Answers | GPU? |
|---|---|---|
| `deep_audit/analysis_detection_descriptive.py` | Prelim-RQ/RQ1 operational and successful-only descriptive metrics | No |
| `deep_audit/analysis_prelim_paired.py` | Prelim-RQ — Cochran's Q and exact paired McNemar tests | No |
| `deep_audit/analysis_rq1_cleaned.py` | RQ1 — Friedman omnibus and within-PR Wilcoxon contrasts | No |
| `rq1_probability_boxplots.ipynb` | RQ1 — paired probability contrasts and revised figure | No |
| `analysis_rq2_word_sentence_count.ipynb` | RQ2 — word count, sentence count | No |
| `analysis_rq2_reading_ease.ipynb` | RQ2 — Flesch-Kincaid readability (FKGL) | No |
| `analysis_rq2_ttr.ipynb` | RQ2 — Type-Token Ratio (TTR) | No |
| `analysis_rq2_mtld.ipynb` | RQ2 — MTLD lexical diversity | No |
| `analysis_rq2_lexical_density.ipynb` | RQ2 — lexical density | No |
| `analysis_rq2_rouge.ipynb` | RQ2 — ROUGE-1/2/L lexical overlap | No |
| `deep_audit/analysis_rq2_bm25_retrieval.py` | RQ2 — fixed-corpus BM25 retrieval | No |
| `analysis_rq2_bertscore.ipynb` | RQ2 — BERTScore semantic similarity | **Yes** — loads DeBERTa-xlarge (~3 GB) |
| `analysis_rq3_codebert.ipynb` | RQ3 — CodeBERT code-text alignment | **Yes** — loads `microsoft/codebert-base` (~500 MB) |
| `analysis_rq3_graphcodebert.ipynb` | RQ3 — GraphCodeBERT code-text alignment | **Yes** — loads `microsoft/graphcodebert-base` (~500 MB) |
| `deep_audit/verify_outputs.py` | Integrity — hashes, sample sizes, result-family cardinalities, and required figures | No |

> **GPU note:** The three GPU notebooks auto-detect CUDA → MPS → CPU and will fall back to CPU, but BERTScore will take several hours without a GPU. On a modern GPU all three take roughly 30–60 minutes combined.

The old `analysis_rq2_bm25.ipynb`, `analysis_rq1_rq2.ipynb`, and `sensitivity_successful_only/` files are retained only for provenance. They do not define the revised estimands and must not be used as the source of manuscript results.

### Audited results and verification

The authoritative precomputed results are under `analysis/outputs/deep_audit/`; paired RQ2/RQ3 result tables are under `analysis/outputs/`; and the corresponding PDFs are under `analysis/outputs/figures_pdf/`. The independent verification record is:

```text
analysis/outputs/deep_audit/verification_report.json
status: PASS
cleaned input SHA-256: 6af94e923188ebd57023d67f25a79e0f1aac7c760a8b4220a6b0fe292c438c63
```

The exact revision pipeline was executed on the study server. External reproducers can use the commands and explicit path arguments in `analysis/deep_audit/README.md`; those scripts require an explicit environment opt-in when run away from the study server. This package itself was synchronized from experiment commit `4037f0b`.

---

## Interactive Tools

The `docs/` folder contains two client-side browser tools (no server needed):

- **`docs/index.html`** — Blind-coding interface for classifying PR descriptions as Human vs. AI
- **`docs/inspector.html`** — PR description inspector with filtering by model, context, and origin

Enable via GitHub Pages (Settings → Pages → Deploy from `/docs`).

---

## Citation

_Citation will be added upon acceptance._
