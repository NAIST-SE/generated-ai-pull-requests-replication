# Corrected-corpus RQ2/RQ3 refresh

This pipeline reruns the remaining manuscript analyses against the immutable corrected corpus
without overwriting the historical notebook outputs.

## Frozen inputs

- Description corpus:
  `phase3/datasets/temperature_1/run_3_corrected_detection_v1/corpus.csv`
  (`07136384785d6d02d17762535ab51a2007a7a47b8c41b77fbb311ce9c8780306`)
- PR diff source:
  `phase3/datasets/selected_prs_with_diffs_and_file_contents.csv`
  (`45c71d0170fc9343d999998ad6515e9b5b8298f63b66215f4110b3911f681d0c`)
- Expected population: 5,322 generated descriptions and 308 human descriptions.

The pipeline prepares clean execution copies of the nine canonical RQ2/RQ3 notebooks. It
changes only their frozen input path, transformer revision selection, unsupported ROUGE and
BERTScore human-self comparisons, and row-level export cells. The original notebooks remain
unchanged for audit comparison. All execution copies and outputs are retained under
`phase3/analysis/outputs/corrected_rq23_v1/`.

## Method decisions

- Structural and lexical metrics retain their original implementations: whitespace word count,
  NLTK sentence segmentation and TTR, `textstat` FKGL, `taaled` MTLD, and spaCy lexical density.
- Human/AI inferential comparisons remain paired by PR. AI values are compared per
  model/context, or averaged across the contexts available for that model/PR for the pooled
  estimand. Wilcoxon signed-rank, matched-pairs rank-biserial correlation, and BH-FDR are
  independently reconstructed from the exported row-level values. The archived tables retain
  ordinary Cliff's delta only as an unpaired marginal sensitivity; it is not the primary effect
  size, and its magnitude labels do not apply to rank-biserial correlation.
- ROUGE and BERTScore are generated-to-reference similarity measures. The human description for
  the same PR is the reference; human-self values are not treated as a sampled baseline
  distribution and no effect size against the artificial value 1 is reported.
- BM25 uses one fixed index of the 308 human descriptions. Each generated description is a query
  and its paired human description is the sole relevant document.
- CodeBERT and GraphCodeBERT retain the manuscript's declared 512-token, first-token embedding
  procedure. The exact cached checkpoint commits are pinned and offline mode is mandatory.

## Server execution

Run only on Nokogiri:

```bash
cd /mnt/ext-hdd1/ayo/llm-in-prs/phase3/analysis
bash corrected_rq23/run_pipeline.sh --preflight
bash corrected_rq23/run_pipeline.sh
```

The full run is intended for `tmux`. Completion requires `completion.json` with status
`complete_and_verified`; notebook completion alone is insufficient.
