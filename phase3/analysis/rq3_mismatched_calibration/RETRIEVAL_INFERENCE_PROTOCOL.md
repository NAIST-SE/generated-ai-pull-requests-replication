# Paired retrieval-inference protocol

This extension treats correct-diff retrieval as the primary conventional diagnostic
for RQ3. It reuses the verified row-level ranks from
`rq3-mismatched-calibration-v1`; it performs no model inference and does not modify
the completed calibration namespace.

## Frozen outcomes

Each description is a query over the same fixed corpus of 308 JabRef PR diffs. The
single relevant item is the query PR's own diff. The row-level outcomes are:

- reciprocal rank of the correct diff (primary inferential outcome);
- rank of the correct diff;
- Recall@1, Recall@5, and Recall@10.

Mean reciprocal rank (MRR) is the arithmetic mean of the per-query reciprocal
ranks. With one relevant item among 308 candidates, the random expectations are
`H_308 / 308` for MRR and `k / 308` for Recall@k.

## Frozen comparisons

For each encoder and generator, AI reciprocal ranks are averaged over all available
context configurations within each PR. This gives every PR equal weight despite the
P-5/P-6 eligibility restriction and the twelve unavailable P-4 generations. The
per-PR AI mean is paired with that PR's human reciprocal rank.

The primary pooled comparisons use:

- two-sided Wilcoxon signed-rank tests;
- matched-pairs rank-biserial correlations, with positive values meaning that the
  human reciprocal rank is larger;
- Benjamini--Hochberg correction across the three generator comparisons within each
  encoder; and
- a deterministic 20,000-resample paired percentile bootstrap interval for the
  human-minus-AI MRR difference.

Context-specific human--AI reciprocal-rank comparisons are also reported using the
same paired test and effect size, with one 18-test Benjamini--Hochberg family per
encoder. Recall@k is descriptive; no additional hypothesis tests are introduced.

Ordinary Cliff's delta and Mann--Whitney U are not reported because they discard the
within-PR pairing.

## Frozen inputs

- Completed calibration manifest:
  `032e69cac929ee2745aaed5b3d78b04a096cb7f83a138b95cdd6dc76eff5926e`
- CodeBERT row diagnostics:
  `90038f1deec7449427c5ac012a0320ad9b16e1f5c0d66d6c080aab80a1add17e`
- GraphCodeBERT row diagnostics:
  `571760f6e4c05c077372b5c390bc9fdf950c41a5a47b58a56d1d3daee1ca7f79`

## Execution

This analysis is lightweight and may run on the Mac because all model-derived ranks
are already frozen and verified:

```bash
cd phase3/analysis
bash rq3_mismatched_calibration/run_retrieval_inference.sh --preflight
bash rq3_mismatched_calibration/run_retrieval_inference.sh
```

Outputs are isolated under `outputs/rq3_retrieval_inference_v1/`. Completion requires
an independent verification report and a `complete_and_verified` completion manifest.
