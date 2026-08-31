# Generation-session sensitivity diagnostic — report

## Outcome

The historical and current Qwen/GPT generations are not textually reproducible under the
combined generation-protocol change. None of the 389 successful old/new pairs matched
exactly. GPT changed more than Qwen on both surface and semantic similarity measures even
though median output length remained broadly similar.

This establishes a genuine reproducibility concern. It does **not** by itself establish that
the manuscript's RQ-level findings change: the diagnostic compares old and new descriptions
to each other, not their detector/quality/similarity outcomes relative to the same human
baseline.

## Frozen design and execution

- Commit: `baed3dd7a161887e2c1e95c3ce84c2e63b2dab64`
- Population: 267 of the 273 audited issue-linked PRs had successful historical Qwen and GPT
  records in all P1--P4 contexts.
- Sample: 50 PRs sampled without replacement with seed `20260831`; the same PRs were used for
  both models and all four contexts.
- Prompt control: all 400 reconstructed prompts matched the historical prompts byte for byte.
- Qwen current protocol: Hugging Face/Novita, BF16, temperature 1, request seed 42.
- GPT current protocol: `gpt-5-mini-2025-08-07`, OpenAI, temperature omitted, request seed 42.
- Unique full-run tasks: 200 Qwen and 200 GPT after a two-call smoke test.
- Actual full-run request attempts: 222 Qwen (200 first-pass plus two retries for each of
  11 persistent failures) and 200 GPT, for 422 full-run requests; including smoke, 424
  requests were made in total.
- Valid outputs: GPT 200/200; Qwen 189/200; total 389/400.

The test is a **combined generation-protocol sensitivity diagnostic**, not a pure temporal
drift experiment. The historical runners recorded seed 42 but did not transmit it, whereas
the current runners transmitted it; API date and backend implementation may also differ.

## Old/new similarity

All intervals below are deterministic 95% bootstrap intervals for the median. BERTScore uses
the same pinned DeBERTa-xlarge-MNLI revision, baseline rescaling, and 512-token limit as the
corrected RQ2 analysis.

| Model/context | Pairs | Exact | Sequence median | ROUGE-L F1 median | BERTScore F1 median (95% CI) | Word-count ratio median |
|---|---:|---:|---:|---:|---:|---:|
| GPT P1 | 50/50 | 0 | 0.297 | 0.230 | 0.319 (0.296--0.349) | 1.101 |
| GPT P2 | 50/50 | 0 | 0.329 | 0.265 | 0.354 (0.334--0.380) | 1.031 |
| GPT P3 | 50/50 | 0 | 0.411 | 0.342 | 0.415 (0.389--0.436) | 1.009 |
| GPT P4 | 50/50 | 0 | 0.344 | 0.254 | 0.362 (0.317--0.387) | 1.153 |
| **GPT overall** | **200/200** | **0** | **0.334** | **0.265** | **0.358 (0.347--0.376)** | **1.042** |
| Qwen P1 | 50/50 | 0 | 0.556 | 0.463 | 0.583 (0.529--0.606) | 0.969 |
| Qwen P2 | 50/50 | 0 | 0.617 | 0.534 | 0.633 (0.595--0.667) | 1.027 |
| Qwen P3 | 50/50 | 0 | 0.482 | 0.406 | 0.461 (0.417--0.506) | 1.058 |
| Qwen P4 | 39/50 | 0 | 0.397 | 0.319 | 0.414 (0.374--0.455) | 1.108 |
| **Qwen overall** | **189/200** | **0** | **0.526** | **0.435** | **0.537 (0.514--0.564)** | **1.041** |

BERTScore does not have a universal threshold at which two descriptions become “equivalent.”
The defensible interpretation is comparative: GPT's repeated generations were markedly less
similar to their historical counterparts than Qwen's, while neither model reproduced a
single description exactly.

## Qwen endpoint limitation

Eleven Qwen P4 requests failed in all three configured passes. The failed prompts had
historical prompt-token counts of 33,979--85,210; successful sampled Qwen P4 prompts had at
most 32,408 tokens. This sharp separation strongly indicates that the current Novita endpoint
enforces an effective context limit near 32,768 tokens. The runner retained only a generic API
error marker, so the exact limit is an evidence-based inference rather than a quoted provider
error.

The missing Qwen P4 comparisons are therefore non-random with respect to prompt length. The
39 observed P4 comparisons must not be generalized to the 11 omitted long prompts. No PR was
replaced, no prompt was truncated, and no provider was substituted post hoc.

## BERTScore truncation

Twenty-four of 389 semantic pairs exceeded the pinned 512-token DeBERTa limit: all were GPT
pairs (P1: 2, P2: 1, P3: 9, P4: 12). Their semantic scores reflect the same truncation policy
used in the corrected RQ2 analysis and are flagged row by row.

## What this changes—and what it does not

Supported:

- The historical/current generation split is not negligible at the text-reproduction level.
- Seeded API generation is not exact reproduction here; moreover, the historical calls did
  not actually transmit the recorded seed.
- A full same-route Qwen regeneration is currently blocked for long P4 prompts by an apparent
  endpoint context limit.

Not yet supported:

- That RQ1 detector conclusions, RQ2 human-relative differences, or RQ3 findings reverse.
- That API date alone caused the differences.
- That the observed Qwen P4 subset represents long-file-content PRs.
- That a full corpus regeneration is already necessary.

## Recommended next decision point

Before changing the manuscript or regenerating the full corpus, run a targeted
outcome-sensitivity analysis on the same frozen diagnostic pairs:

1. Compare historical and current outputs against the same human descriptions using the
   study's RQ2 metrics, focusing on paired changes in effect direction and magnitude.
2. Re-score the same pairs with the original detectors, beginning with the two local RoBERTa
   detectors; use ZeroGPT only with separate external-data/cost authorization.
3. Treat Qwen P4 as 39 observed plus 11 structurally missing long prompts; do not impute,
   replace, or silently truncate them.
4. Consider a full regeneration only if outcome-level conclusions materially shift and after
   predeclaring a single feasible long-context Qwen provider/protocol for every context.

Until that outcome-facing check is complete, the manuscript should not claim that the mixed
sessions are harmless, but it also should not claim that they changed the substantive RQs.
