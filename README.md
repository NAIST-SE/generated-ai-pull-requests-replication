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

| File | Description |
|---|---|
| `selected_prs_sampled.csv` | 308 human-written PR descriptions (final sample) |
| `selected_prs_with_diffs_and_file_contents.csv` | PR diffs and file contents per PR |
| `selected_prs_with_related_issues.csv` | Linked GitHub issues per PR |
| `temperature_1/run_3/detection_results.csv` | Detection scores for all 5,836 entries (primary results file) |
| `temperature_1/run_3/<model>_*.csv` | Per-model generated descriptions |

The `detection_results.csv` from run 3 is the source of truth for Prelim-RQ, RQ1, and the sensitivity analysis.

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
# Filter and sample PRs from the raw JabRef dump
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

| Notebook | Content |
|---|---|
| `analysis_detectability_by_llm.ipynb` | Prelim-RQ + RQ1 (detectability, Cliff's δ) |
| `rq1_probability_boxplots.ipynb` | Generates Figure 3 (RQ1 probability boxplots) |
| `analysis_rq3_*.ipynb` | RQ2 + RQ3 textual and semantic metrics |
| `sensitivity_successful_only/sensitivity_successful_only.ipynb` | Detector-failure sensitivity rerun (Threats section) |

---

## Interactive Tools

The `docs/` folder contains two client-side browser tools (no server needed):

- **`docs/index.html`** — Blind-coding interface for classifying PR descriptions as Human vs. AI
- **`docs/inspector.html`** — PR description inspector with filtering by model, context, and origin

Enable via GitHub Pages (Settings → Pages → Deploy from `/docs`).

---

## Citation

_Citation will be added upon acceptance._
