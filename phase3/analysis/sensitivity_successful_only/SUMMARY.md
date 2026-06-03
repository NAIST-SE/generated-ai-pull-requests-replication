# Sensitivity Analysis: Successful-Detection Subset — Summary

## Question
Are the Prelim-RQ / RQ1 conclusions driven by genuine indistinguishability, or by detector failure on short outputs (1,201 RoBERTa failures, concentrated in Llama-3.1-70B)?

## Method
Re-ran Prelim-RQ and RQ1 on three subsets of `phase3/datasets/temperature_1/run_3/detection_results.csv`:
- **original** — no filter (matches the paper).
- **{tool}_success** — keep only rows where the named tool returned a successful score; for each headline finding the "per-tool" view is the matching `{tool}_success` view.
- **common_success** — keep only rows where all three tools succeeded (the strictest, most apples-to-apples view across detectors).

Paired filtering is applied per subset so that the human baseline is restricted to the same set of PRs as the AI rows it is being compared against.

## Subset sizes
| view              | paired PRs | AI rows | Human rows |
|-------------------|-----------:|--------:|-----------:|
| original          |        308 |   5,528 |        308 |
| roberta_success   |        263 |   3,808 |        263 |
| roberta-qa_success|        263 |   3,808 |        263 |
| zerogpt_success   |        308 |   5,528 |        308 |
| common_success    |        263 |   3,808 |        263 |

ZeroGPT had 0 failures on this run, so `zerogpt_success` ≡ `original`. RoBERTa and RoBERTa-QA failed on the same 1,201 rows.

## Headline result: original effects survive
44 medium-or-large pairwise context effects were tabulated in the paper's Table 4. Across all 44:

- **Sign flips (per-tool view): 0**
- **Sign flips (common-success view): 0**
- **Magnitude class drops of 2+ levels (per-tool view): 0**
- **Magnitude class drops of 2+ levels (common-success view): 1**
  - ZeroGPT × Qwen2.5-72B × "PR Title vs Without File Contents": Large (−0.507) → Small (−0.315). All other ZeroGPT × Qwen rows hold within one magnitude class.

Critically, the open-source model effects (Llama-3.1-70B), which is where the detector failures concentrated, are if anything **larger** in the per-tool view than in the original (e.g., RoBERTa × Llama × PR Title vs Linked Issue: 0.713 → 0.846; RoBERTa-QA × Llama × PR Title vs Linked Issue: 0.710 → 0.838). The narrative-context-anchoring story is therefore not an artefact of detector failures.

## Notable side-finding: Prelim-RQ F1 for Llama is conservatively reported
When restricted to scoreable rows, Prelim-RQ F1 for Llama increases substantially:

| tool       | model                       | original F1 | per-tool subset F1 | Δ      |
|------------|-----------------------------|------------:|-------------------:|-------:|
| roberta    | meta-llama/llama-3.1-70b    |      0.161  |             0.375  | +0.214 |
| roberta-qa | meta-llama/llama-3.1-70b    |      0.150  |             0.329  | +0.178 |
| roberta    | Qwen/Qwen2.5-72B            |      0.335  |             0.349  | +0.014 |
| roberta-qa | Qwen/Qwen2.5-72B            |      0.590  |             0.610  | +0.019 |
| roberta    | gpt-5-mini-2025-08-07       |      0.013  |             0.015  | +0.002 |
| roberta-qa | gpt-5-mini-2025-08-07       |      0.000  |             0.000  |  0.000 |

Interpretation: Llama produces a large tail of very short outputs that detectors flag as unscoreable. Including them as "missed detections" understates detectability on text the detectors could actually score. The qualitative claim ("Llama outputs are hard to detect under linked-issue context, less hard under code-dense context") still holds; the precise F1 magnitudes for Llama in the paper are conservative.

GPT-5-mini stays essentially undetectable by RoBERTa-based tools in every view (F1 ≤ 0.02), reinforcing the RQ1 statement that the detection-as-proxy framing is uninformative for this model.

## Recommendation for the paper
Add a sensitivity-analysis paragraph to Threats-to-Validity stating:
1. RQ1 was re-run after excluding detector-failure rows under three subset views.
2. None of the 44 medium-or-large context effects flipped sign in the per-tool subset; only 1 effect dropped a magnitude class in the strictest common-success view.
3. The open-source Prelim-RQ F1 magnitudes are conservative because failed scoring on very short outputs is counted as missed detection; the qualitative context-anchoring story is unchanged.

Artifacts for citation/reproducibility live under `phase3/analysis/sensitivity_successful_only/outputs/`.
