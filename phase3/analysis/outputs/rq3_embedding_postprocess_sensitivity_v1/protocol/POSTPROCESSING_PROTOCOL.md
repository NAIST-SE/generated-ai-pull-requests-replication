# Conditional embedding-space sensitivity protocol

The completed `rq3-mismatched-calibration-v1` diagnostic found substantial
background similarity, especially for CodeBERT, and therefore activated this
pre-specified second-stage sensitivity analysis. This stage reuses the verified
retained embeddings; it performs no model inference and does not alter the
manuscript or the primary RQ3 analysis.

## Frozen transformation

For each encoder separately, the reference set contains exactly:

- the 308 human-description embeddings; and
- the 308 corresponding PR-diff embeddings.

This balanced reference avoids allowing the 5,322 AI descriptions to dominate the
estimated mean or principal directions. A single common mean and common principal
directions are fitted to the combined 616-vector reference, preserving the shared
coordinate system for descriptions and diffs. The same frozen transform is then
applied to every human description, AI description, and PR diff.

All four variants are reported:

1. `centered`: subtract the common reference mean;
2. `remove_top_1`: center, then remove the leading principal direction;
3. `remove_top_2`: center, then remove the leading two directions; and
4. `remove_top_3`: center, then remove the leading three directions.

The raw calibration is carried into the comparison tables as `raw`. No transformed
variant will be selected as preferred after inspecting its result. Removing principal
directions is treated as a robustness sensitivity, not as automatic debiasing or a
replacement metric.

Cosine similarities, all-307-nonmatch margins, retrieval ranks, paired tests,
matched-pairs rank-biserial correlations, and Benjamini--Hochberg families are
identical to the frozen calibration protocol.

## Frozen retained inputs

- `diff_ids.csv`:
  `1a6c6ff3ae9b6c326a9ae3b5c9b6cfa991b3632decd40d3f4b0859df920b9d65`
- `row_metadata.csv`:
  `23827b92ceecffbc12c622887b277f796c053a1363c2bc4264a75e24c0db6e71`
- CodeBERT description embeddings:
  `ee5644512f9140238ea257779f93689d34eff539713298d1a68268eef818a8fb`
- CodeBERT diff embeddings:
  `fbf6c21bf414dfcf8708ca2c1ca3f3d5422d6a49a656fb7902709320e47299b7`
- GraphCodeBERT description embeddings:
  `e535c27de1996573e9841e3ad947864c0e61d5e987478a01c2e34110e91fcd33`
- GraphCodeBERT diff embeddings:
  `bb6559833eac50f0f104ebbf437f7700102bd28b4f38245a8a78ea95a49dd273`
- Raw CodeBERT calibration rows:
  `90038f1deec7449427c5ac012a0320ad9b16e1f5c0d66d6c080aab80a1add17e`
- Raw GraphCodeBERT calibration rows:
  `571760f6e4c05c077372b5c390bc9fdf950c41a5a47b58a56d1d3daee1ca7f79`

## Execution

Run on Nokogiri only:

```bash
cd /mnt/ext-hdd1/ayo/llm-in-prs/phase3/analysis
bash rq3_mismatched_calibration/run_postprocess_sensitivity.sh --preflight
bash rq3_mismatched_calibration/run_postprocess_sensitivity.sh
```

Outputs are isolated under
`outputs/rq3_embedding_postprocess_sensitivity_v1/` and are complete only when
independent reconstruction writes `complete_and_verified` to `completion.json`.
