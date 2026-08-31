# Generation-session outcome-sensitivity diagnostic

This isolated, server-only diagnostic asks whether regenerating the frozen 50-PR
Qwen/GPT P1--P4 sample changes the *study outcomes*, rather than merely producing
different wording. It does not replace the full 308-PR experiment, and it does not
change the manuscript.

## Frozen scope

- Inputs: the existing `generation-session-diagnostic-v1` sample and successful
  current generations, the corrected 5,630-row detector corpus, and the corrected
  RQ2 row-level outputs.
- Pair denominators: GPT has 50 PRs in each P1--P4 cell; Qwen has 50 in P1--P3
  and 39 in P4. The 11 persistent Qwen P4 failures are not replaced or imputed.
- Versions: `historical` and `current`, compared for the same PR, model, context,
  prompt, and human reference.
- RQ2 structural/lexical outcomes: word count, sentence count, FKGL, TTR,
  MTLD-original, MTLD-bidirectional, and lexical density.
- RQ2 generated-to-reference outcomes: ROUGE-L F1, pinned baseline-rescaled
  BERTScore F1, and reciprocal rank from the fixed 308-human-description BM25
  corpus.
- RQ1 robustness outcomes: the two pinned local RoBERTa detectors only. ZeroGPT
  is deliberately excluded because it would require a new external paid run.
- P5 and P6 were not regenerated. Claims that depend on all six contexts, or on
  P5/P6 being best/worst, cannot be retested by this diagnostic.

## Exact methods

Metric definitions match the corrected study implementations:

- word count: `len(text.split())`;
- sentence count: `nltk.sent_tokenize`;
- FKGL: `textstat.flesch_kincaid_grade`;
- TTR: lowercased `nltk.word_tokenize`, types divided by tokens;
- MTLD: lowercased `nltk.word_tokenize` followed by TAALED `lexdiv`; texts with
  fewer than 10 tokens are undefined; `mtldo` and bidirectional `mtld` are saved;
- lexical density: spaCy `en_core_web_sm`, excluding punctuation/space, with
  `NOUN`, `PROPN`, `VERB`, `ADJ`, and `ADV` treated as content words;
- ROUGE: `RougeScorer([rouge1, rouge2, rougeL], use_stemmer=True)`, human as
  reference and AI as candidate;
- BERTScore: pinned `microsoft/deberta-xlarge-mnli` revision
  `5b07a9086c1dbb79981ff7b05b4d1ad83b3af51c`, 40 layers, English rescaling
  baseline, 512-token limit, batch size 8;
- BM25: one index of all 308 human descriptions, lowercase
  `[A-Za-z0-9_]+` tokens, `BM25Okapi(k1=1.5, b=0.75)`, average rank for ties,
  paired human description as the sole relevant item;
- detectors: pinned `Hello-SimpleAI/chatgpt-detector-roberta` in single-text mode
  and `Hello-SimpleAI/chatgpt-qa-detector-roberta` in title--description pair
  mode, 512-token truncation and label-1 probability, threshold 0.5.

Historical metric values are recomputed and compared to the authoritative
corrected row-level outputs before current-session results are interpreted. A
stage fails closed if the historical validation differs beyond the declared
floating-point tolerance. The non-neural and linguistic metrics use `1e-10`.
BERTScore uses `5e-6`: an initial fail-closed check at `2e-6` found one of 389
historical values outside that bound by `2.5779008865356445e-6`, while all other
values passed. The relaxed bound and its audit trail are stored in the metadata;
the discrepancy is below the study's reported precision.

## Statistical interpretation

For each model and P1--P4 context:

1. Structural/lexical human-versus-AI comparisons use paired Wilcoxon tests,
   Cliff's delta (positive means human values tend to be larger), and matched-pair
   rank-biserial correlation. BH-FDR is applied across the eight model-by-context
   cells separately for each metric and generation version.
2. Historical-versus-current values use direct per-PR paired differences. The
   diagnostic reports medians, the median current-minus-historical change with a
   deterministic percentile bootstrap interval, paired Wilcoxon tests, and
   matched-pair rank-biserial correlation. BH-FDR is applied across the eight
   cells separately for each outcome.
3. Detector probabilities use the same paired analysis. Binary changes report
   both discordant directions and an exact McNemar/binomial p-value. Performance
   is recomputed against the same corresponding human PRs for both versions.

No arbitrary equivalence margin is imposed after seeing the data. The report
flags transparent qualitative changes: Cliff-delta direction reversals, magnitude
category changes at the study's existing 0.147/0.33/0.474 thresholds, changes in
the top-ranked P1--P4 context, and detector flag/F1 changes. Significance changes
are reported but are not treated as the primary evidence of instability.

## Server run

From the Nokogiri repository root:

```bash
bash phase3/analysis/generation_session_outcome_sensitivity/run_pipeline.sh
```

The launcher refuses non-Nokogiri hosts, runs model-free tests first, uses the
`iwadi-main` and `iwadi-py312` interpreters explicitly, and writes a completion
manifest only after all validations pass. Generated CSV/JSON/Markdown outputs do
not contain raw prompts or descriptions.
