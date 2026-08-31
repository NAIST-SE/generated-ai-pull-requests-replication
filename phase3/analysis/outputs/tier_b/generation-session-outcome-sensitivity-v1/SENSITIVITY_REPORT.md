# Generation-session outcome-sensitivity results

Status: analysis complete; final integrity verification is performed separately.

## Scope

- 389 successful per-PR/model/context pairs, each evaluated historically and currently.
- GPT: 50 PRs in each P1--P4 cell; Qwen: 50 in P1--P3 and 39 in P4.
- Structural/lexical RQ2 outcomes, ROUGE-L, BERTScore, fixed-corpus BM25, and the two pinned local RoBERTa detectors.
- ZeroGPT and P5/P6 were not rerun. This diagnostic cannot retest six-context or P5/P6 claims.

## Mechanical sensitivity flags

- Structural/lexical cells: 56.
- Cliff-delta direction reversals: 4.
- Cliff-delta magnitude-category changes: 9.
- Changes of at least two magnitude tiers: 0.
- BH-adjusted significance-status changes: 4 (secondary diagnostic only).
- P1--P4 top-context changes across the six model-by-similarity-metric comparisons: 2.
- Detector binary flips across all paired AI descriptions and both detectors: 114.
- Largest absolute detector flag-rate change in one detector/model/context cell: 0.179.
- Largest absolute detector F1 change in one detector/model/context cell: 0.199.

These counts are transparent flags, not post-hoc equivalence thresholds. Interpret the
underlying continuous estimates and tables before deciding whether a manuscript claim is
stable. The diagnostic cannot identify whether any difference is caused by nondeterminism,
provider/backend evolution, or another generation-protocol detail.
