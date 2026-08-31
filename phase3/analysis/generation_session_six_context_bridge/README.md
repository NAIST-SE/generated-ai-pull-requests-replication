# Six-context generation-session bridge diagnostic

This server-only, no-API diagnostic checks whether the study's six-context
similarity ranking remains stable when the newly regenerated/scored P1--P4
descriptions replace the historical P1--P4 descriptions. The corrected P5 and
P6 row-level results are held fixed. It does not regenerate text, rescore any
model, alter the manuscript, or replace the full 308-PR analysis.

## Frozen design

- PRs: the complete-case subset of the frozen 50-PR generation-session sample.
  GPT has 50 PRs; Qwen has 39 because the 11 persistent Qwen P4 generation
  failures are not replaced or imputed.
- Bridge versions:
  - `historical_bridge`: historical P1--P4 plus corrected P5--P6;
  - `current_bridge`: current P1--P4 plus the same corrected P5--P6.
- Metrics and headline aggregations:
  - ROUGE-L F1: median;
  - baseline-rescaled BERTScore F1: median;
  - BM25 reciprocal rank: arithmetic mean (MRR).
- Ranking: aggregate values are ranked descending within each model, metric,
  and bridge version. Average ranks are assigned to exact ties.
- P6 diagnostics: top aggregate status, gap from the best non-P6 context, and
  the shares of PRs for which P6 is a tied or strict within-PR maximum.

## Interpretation boundary and decision rule

The bridge isolates whether replacing P1--P4 with the newer generation session
changes the six-context headline while P5--P6 remain fixed. It is not a
single-session six-context replication and cannot prove equivalence between
generation sessions.

The predeclared operational rule is:

- if P6 remains top-ranked for every model-by-headline-metric cell in the
  `current_bridge`, the bridge does not trigger a complete regeneration rerun;
- if P6 loses top rank in any such cell, flag that cell for review and escalate
  the decision about a complete rerun.

No significance test is used to manufacture a pass/fail threshold for this
ranking diagnostic. The output preserves all aggregate values and context ranks
so the decision can be audited directly.

## Authoritative inputs

The script reads only existing numeric, raw-text-free server outputs:

- `outputs/tier_b/generation-session-outcome-sensitivity-v1/`
  (`pair_manifest.csv`, `main_metrics.csv`, `bertscore_metrics.csv`);
- `outputs/corrected_rq23_v1/notebooks/outputs/row_level/`
  (`rouge_metrics.csv`, `bertscore_metrics.csv`);
- `outputs/corrected_rq23_v1/notebooks/outputs/bm25_retrieval/`
  (`bm25_retrieval_scores.csv`).

## Server run

From `/mnt/ext-hdd1/ayo/llm-in-prs` on Nokogiri:

```bash
bash phase3/analysis/generation_session_six_context_bridge/run.sh
```

The launcher runs model-free tests, performs fail-closed coverage and duplicate
checks, writes only numeric/hashed identifiers, and creates `completion.json`
only after the output verifier passes.
