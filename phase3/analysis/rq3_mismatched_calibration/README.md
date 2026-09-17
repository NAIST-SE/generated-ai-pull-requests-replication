# RQ3 matched-versus-mismatched calibration

This frozen diagnostic tests whether the raw CodeBERT and GraphCodeBERT cosine
similarities used in RQ3 distinguish the correct PR description--diff pairing from
nonmatching diffs. It does not change the manuscript, replace the declared RQ3
analysis, or perform centering, whitening, principal-component removal, or layer
selection.

## Protocol

The diagnostic reuses the exact corrected corpus, PR-diff source, preprocessing,
model revisions, 512-token limit, and final-layer first-token representation from
the verified `corrected_rq23_v1` run.

For each of the 5,630 descriptions, the analysis embeds the description and all 308
cleaned PR diffs. The matching diff is the PR's own diff; all other 307 diffs form a
deterministic nonmatching reference set. No random shuffle or seed-dependent sample
is used.

The primary per-description estimand is:

```text
calibrated margin = cosine(correct diff) - mean cosine(307 incorrect diffs)
```

Secondary diagnostics are the median-background margin, the proportion of positive
margins, the rank of the correct diff among all 308 candidates, mean reciprocal
rank, and Recall@1/5/10. With one relevant diff among 308, the random-reference rates
are 1/308, 5/308, and 10/308; the expected random mean reciprocal rank is
`H_308 / 308`.

## Frozen inferential design

1. Within each encoder, one-sided Wilcoxon signed-rank tests assess whether the
   mean-background margin is greater than zero for the human cell and each of the
   18 model--context cells. Benjamini--Hochberg correction is applied across these
   19 tests.
2. For each model, AI margins are averaged across its available contexts per PR and
   paired with the human margin for that PR. Two-sided Wilcoxon tests and
   matched-pairs rank-biserial correlations are reported. Benjamini--Hochberg is
   applied across the three model comparisons within each encoder and metric.
3. Human-versus-AI margin comparisons are also reported for the 18 model--context
   cells, with one Benjamini--Hochberg family per encoder.
4. Raw matched cosine is retained as a secondary comparison so that changes between
   the raw and calibrated conclusions are visible.

The diagnostic does not use Mann--Whitney U or ordinary Cliff's delta because the
human and AI observations are paired by PR.

## Interpretation gate

- Clear correct-pair separation across source cells and retrieval substantially
  above the random reference would support interpreting the score as PR-specific
  code--text association.
- Small or inconsistent separation would limit the result to a model-specific
  embedding score rather than strong semantic alignment.
- Failure to separate correct from incorrect diffs would mean that raw cosine cannot
  support the alignment interpretation. Only then should a separately pre-specified
  centering or dominant-component-removal sensitivity analysis be considered.

Statistical significance alone is not treated as evidence of practical separation;
the margin distribution and retrieval diagnostics must be considered together.

## Frozen inputs and checkpoints

- Corrected description corpus SHA-256:
  `07136384785d6d02d17762535ab51a2007a7a47b8c41b77fbb311ce9c8780306`
- PR diff source SHA-256:
  `45c71d0170fc9343d999998ad6515e9b5b8298f63b66215f4110b3911f681d0c`
- Existing CodeBERT row scores SHA-256:
  `932dbb166dfcc221e48a7696161ed541e24057e059bc79e057bd1e331e5c1550`
- Existing GraphCodeBERT row scores SHA-256:
  `2c96eb065c971f07fc21ffd145f9cbab097695b532454303504b8c1875b2820b`
- CodeBERT revision:
  `3b0952feddeffad0063f274080e3c23d75e7eb39`
- GraphCodeBERT revision:
  `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`

## Execution

Run on Nokogiri only:

```bash
cd /mnt/ext-hdd1/ayo/llm-in-prs/phase3/analysis
bash rq3_mismatched_calibration/run_pipeline.sh --preflight
bash rq3_mismatched_calibration/run_pipeline.sh
```

The full run writes to the immutable namespace
`outputs/rq3_mismatched_calibration_v1/`. Completion requires both
`verification/verification_report.json` with `status: pass` and
`completion.json` with `status: complete_and_verified`.
