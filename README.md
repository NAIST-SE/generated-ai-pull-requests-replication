# Replication Package — Context, Semantics, and Alignment in Generated AI Pull Requests

This repository contains the code, analysis notebooks, and pre-computed results for the paper:

> **Context, Semantics, and Alignment in Generated AI Pull Requests**  
> Under review — Transactions on Software Engineering (TSE)

Large datasets are hosted on OSF: **https://osf.io/aq5rj/**

---

## Study Overview

The paper investigates AI-generated pull-request descriptions produced under six different context configurations and answers four research questions:

- **Prelim-RQ** — How accurate are existing tools at detecting AI-generated PR descriptions?
- **RQ1** — How does generation context affect the *distinguishability* of AI-generated PR descriptions?
- **RQ2** — How does generation context shape the *textual similarity* between AI-generated and human-written PR descriptions? (structural, lexical, and semantic metrics)
- **RQ3** — To what extent does the *code-text alignment* of AI-generated PR descriptions resemble that of human-written ones?

**Models:** `gpt-5-mini-2025-08-07` (proprietary), `meta-llama/Llama-3.1-70B-Instruct`, `Qwen/Qwen2.5-72B-Instruct` — all generated at **temperature = 1**.

**Detectors:** ZeroGPT (commercial), ChatGPT Detector RoBERTa, ChatGPT-QA Detector RoBERTa (both from Guo et al.).

**Context configurations** used when prompting each model:

| ID | Name | Context provided to the LLM |
|---|---|---|
| P-1 | Repo Path | Repository name + affected file paths |
| P-2 | PR Title | PR title only |
| P-3 | PR Diffs | Unified diffs (baseline) |
| P-4 | File Contents | Full file contents |
| P-5 | Linked Issue | Linked GitHub issue descriptions |
| P-6 | Without File Contents | All of P-1 + P-2 + P-3 + P-5 combined |

---

## Repository Structure

```
.
├── collection/          # Data collection — PR mining, diff/file fetching, issue linking
├── generation/          # LLM generation scripts (GPT-5-mini, Llama-3.1-70B, Qwen2.5-72B)
├── detection/           # AI-text detection pipeline (ZeroGPT, RoBERTa, RoBERTa-QA)
├── analysis/            # Analysis notebooks for all RQs + sensitivity analysis
│   └── sensitivity_successful_only/  # Detector-failure sensitivity rerun (cited in Threats)
├── docs/                # Browser-based tools: blind-coding interface + PR inspector
├── data_loader.py       # Shared dataset loader used across scripts and notebooks
├── requirements.txt     # Python dependencies
├── run_generation.sh    # Parallel generation entry point
├── run_detection_pipeline.sh  # Detection entry point
└── verify_detection.py  # Quick integrity check on detection_results.csv
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
| `datasets/temperature_1/run_3/detection_results.csv` | Step 4 | Detection scores for all 5,836 entries — **primary results file** |
| `datasets/temperature_1/run_2/detection_results.csv` | `compare_runs.ipynb` | Detection scores from an independent run (consistency validation) |
| `datasets/temperature_1/run_1/detection_results_run_1.csv` | `compare_runs.ipynb` | Detection scores from an independent run (consistency validation) |

### On OSF only (large generation outputs)

Download from OSF and place under `datasets/temperature_1/run_3/` to re-run detection:

| File | Size | Description |
|---|---|---|
| `temperature_1/run_3/chatgpt_results_exp_chatgpt_1771584462.csv` | 46 MB | GPT-5-mini generated descriptions |
| `temperature_1/run_3/meta-llama_llama-3.1-70b-instruct_1771584467.csv` | 37 MB | Llama-3.1-70B generated descriptions |
| `temperature_1/run_3/Qwen_Qwen2.5-72B-Instruct_1771584467.csv` | 37 MB | Qwen2.5-72B generated descriptions |

> **Analysis only:** `datasets/temperature_1/run_3/detection_results.csv` is already in this repo — jump straight to Step 4 without downloading anything from OSF.

---

## Reproducing the Study

### Prerequisites

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For GPU-based generation (Llama, Qwen) on an HPC cluster, see `HPC_GUIDE.md`.

### Step 1 — Data Collection (optional; datasets on OSF)

```bash
python collection/filter_and_sample_prs.py

# Fetch PR file diffs and contents
python collection/fetch_pr_files.py

# Fetch linked GitHub issues
python collection/fetch_pr_issues.py
```

> Requires a GitHub token in `collection/env/github_token.txt`.

### Step 2 — Generation

Generates PR descriptions for all 308 PRs × 6 context configurations (P-1 – P-6) for each model at **temperature = 1**.

```bash
export OPENROUTER_API_KEY="..."   # for Llama and Qwen via OpenRouter / HuggingFace router
export OPENAI_API_KEY="..."       # for GPT-5-mini
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

### Step 4 — Analysis

Open the notebooks in `analysis/` with Jupyter **from the repository root** — this ensures relative paths to `datasets/` resolve correctly:

```bash
cd generated-ai-pull-requests-replication
jupyter notebook
```

> **Note on notebook naming:** The `analysis_rq3_*` prefix is an internal project code (not "Research Question 3"). Those notebooks cover both RQ2 (textual similarity) and RQ3 (code-text alignment) as indicated in the table below.

| Notebook | Answers | GPU? |
|---|---|---|
| `analysis_detectability_by_llm.ipynb` | Prelim-RQ + RQ1 | No |
| `rq1_probability_boxplots.ipynb` | RQ1 — generates Figure 3 (probability boxplots) | No |
| `analysis_rq3_word_sentence_count.ipynb` | RQ2 — word count, sentence count | No |
| `analysis_rq3_reading_ease.ipynb` | RQ2 — Flesch-Kincaid readability (FKGL) | No |
| `analysis_rq3_ttr.ipynb` | RQ2 — Type-Token Ratio (TTR) | No |
| `analysis_rq3_mtld.ipynb` | RQ2 — MTLD lexical diversity | No |
| `analysis_rq3_lexical_density.ipynb` | RQ2 — lexical density | No |
| `analysis_rq3_rouge.ipynb` | RQ2 — ROUGE-1/2/L lexical overlap | No |
| `analysis_rq3_bm25.ipynb` | RQ2 — BM25 lexical relevance | No |
| `analysis_rq3_bertscore.ipynb` | RQ2 — BERTScore semantic similarity | **Yes** — loads DeBERTa-xlarge (~3 GB) |
| `analysis_rq3_codebert.ipynb` | RQ3 — CodeBERT code-text alignment | **Yes** — loads `microsoft/codebert-base` (~500 MB) |
| `analysis_rq3_graphcodebert.ipynb` | RQ3 — GraphCodeBERT code-text alignment | **Yes** — loads `microsoft/graphcodebert-base` (~500 MB) |
| `compare_runs.ipynb` | Validity — paired statistical tests (Wilcoxon + t-test) confirming detection scores are consistent across two independent generation runs at temperature = 1 | No |
| `sensitivity_successful_only/sensitivity_successful_only.ipynb` | Threats — detector-failure sensitivity rerun | No |

> **GPU note:** The three GPU notebooks auto-detect CUDA → MPS → CPU and will fall back to CPU, but BERTScore will take several hours without a GPU. On a modern GPU all three take roughly 30–60 minutes combined.

---

## Interactive Tools

The `docs/` folder contains two client-side browser tools (no server needed):

- **`docs/index.html`** — Blind-coding interface for classifying PR descriptions as Human vs. AI
- **`docs/inspector.html`** — PR description inspector with filtering by model, context, and origin

Enable via GitHub Pages (Settings → Pages → Deploy from `/docs`).

---

## Citation

_Citation will be added upon acceptance._
