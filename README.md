# Replication Package — Context, Semantics, and Alignment in Generated AI Pull Requests

This repository contains the code, analysis notebooks, and pre-computed results for the paper:

> **Context, Semantics, and Alignment in Generated AI Pull Requests**  
> Under review — Transactions on Software Engineering (TSE)

Large datasets are hosted on OSF: **https://osf.io/aq5rj/**

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

Download from OSF (https://osf.io/aq5rj/) and place under a `datasets/` directory at the repo root:

| File | Needed for | Description |
|---|---|---|
| `jabref-prs.csv` | Step 1 (optional) | Raw JabRef PR dump mined from GitHub |
| `jabref_prs_additional.csv` | Step 1 (optional) | Supplementary PRs added in a later collection pass |
| `selected_prs_sampled.csv` | Steps 2–4 | 308 human-written PR descriptions (final sample) |
| `selected_prs_with_diffs_and_file_contents.csv` | Step 2 | PR diffs and file contents per PR |
| `selected_prs_with_related_issues.csv` | Step 2 | Linked GitHub issues per PR |
| `temperature_1/run_3/detection_results.csv` | Step 4 | Detection scores for all 5,836 entries (primary results file) |
| `temperature_1/run_3/<model>_*.csv` | Step 3–4 | Per-model generated descriptions |

The `detection_results.csv` from run 3 is the source of truth for Prelim-RQ, RQ1, and the sensitivity analysis.

> **Skipping collection and generation:** If you only want to reproduce the analysis, download `selected_prs_sampled.csv` and `temperature_1/run_3/detection_results.csv` from OSF and jump to Step 4. `jabref-prs.csv` is only needed if you want to re-run the PR filtering and sampling from scratch.

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
# Download jabref-prs.csv from OSF (https://osf.io/aq5rj/) into datasets/ first, then:
python collection/filter_and_sample_prs.py

# Fetch PR file diffs and contents
python collection/fetch_pr_files.py

# Fetch linked GitHub issues
python collection/fetch_pr_issues.py
```

> Requires a GitHub token in `collection/env/github_token.txt`.

### Step 2 — Generation

```bash
# Runs Llama, Qwen (via OpenRouter / HuggingFace router),
# and GPT-5-mini in parallel. Set API keys as env vars first.
export OPENROUTER_API_KEY="..."
export OPENAI_API_KEY="..."
bash run_generation.sh
```

See `generation/P4_TRUNCATION_GUIDE.md` for handling the File Contents (P-4) context, which can exceed provider token limits.

### Step 3 — Detection

```bash
# Merge generated CSVs, then run three detectors in parallel
bash run_detection_pipeline.sh
```

> ZeroGPT requires tokens in `detection/env/zero-gpt-tokens.txt` (one token per line).

Verify results after detection:
```bash
python verify_detection.py
```

### Step 4 — Analysis

Open the notebooks in `analysis/` with Jupyter. They expect `datasets/` to be populated.

| Notebook | Content | GPU? |
|---|---|---|
| `analysis_detectability_by_llm.ipynb` | Prelim-RQ + RQ1 (detectability, Cliff's δ) | No |
| `rq1_probability_boxplots.ipynb` | Generates Figure 3 (RQ1 probability boxplots) | No |
| `analysis_rq3_bertscore.ipynb` | BERTScore semantic similarity (RQ2) | **Yes** — loads DeBERTa-xlarge (~3 GB) |
| `analysis_rq3_codebert.ipynb` | CodeBERT code-text alignment (RQ3) | **Yes** — loads `microsoft/codebert-base` (~500 MB) |
| `analysis_rq3_graphcodebert.ipynb` | GraphCodeBERT code-text alignment (RQ3) | **Yes** — loads `microsoft/graphcodebert-base` (~500 MB) |
| `analysis_rq3_rouge.ipynb` | ROUGE-1/2/L overlap (RQ2) | No |
| `analysis_rq3_bm25.ipynb` | BM25 lexical retrieval score (RQ2) | No |
| `analysis_rq3_word_sentence_count.ipynb` | Word and sentence count verbosity (RQ2) | No |
| `analysis_rq3_ttr.ipynb` | Type-Token Ratio lexical diversity (RQ2) | No |
| `analysis_rq3_mtld.ipynb` | MTLD lexical diversity (RQ2) | No |
| `analysis_rq3_lexical_density.ipynb` | Lexical density (RQ2) | No |
| `analysis_rq3_reading_ease.ipynb` | Flesch-Kincaid readability (RQ2) | No |
| `sensitivity_successful_only/sensitivity_successful_only.ipynb` | Detector-failure sensitivity rerun (Threats) | No |

> **GPU note:** The three GPU notebooks auto-detect the best available device (CUDA → MPS → CPU) and will run on CPU as a fallback, but expect several hours for BERTScore without a GPU. On a modern GPU the three notebooks together take roughly 30–60 minutes.

---

## Interactive Tools

The `docs/` folder contains two client-side browser tools (no server needed):

- **`docs/index.html`** — Blind-coding interface for classifying PR descriptions as Human vs. AI
- **`docs/inspector.html`** — PR description inspector with filtering by model, context, and origin

Enable via GitHub Pages (Settings → Pages → Deploy from `/docs`).

---

## Citation

_Citation will be added upon acceptance._
