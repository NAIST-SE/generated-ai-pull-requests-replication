# Reader-facing context names

Approved September 29, 2026. This is a presentation change, not a new experiment.

| ID | Frozen/legacy label | Current manuscript label |
|---|---|---|
| P-1 | Repo Path | Repo Path |
| P-2 | PR Title | PR Title |
| P-3 | File Diff | Patch |
| P-4 | File Contents | Full Code |
| P-5 | Linked Issue | Linked Issue |
| P-6 | Without File Contents | Multi-source context (compact: Multi-source) |

Full Code supplies the contents of modified files, not the whole repository.
Multi-source combines repository path, PR title, diff, and verified linked-issue text.
No all-sources comparator was tested; P-6 is not a removal-of-code ablation.

## Preservation boundary

Use P-1--P-6 as the join keys. Frozen datasets, prompts, historical notebooks,
statistical results, and result-package filenames retain their original labels.
The current supplement and manuscript plots use the new names. The original
`corrected_detection_v1`, `corrected_rq23_v1`, and all RQ3 analysis packages remain
unchanged, including the historical figure copies anchored by their original hashes.
Frozen human-assessment packets are not relabeled.

## Current figures and evidence

`phase3/analysis/outputs/context_display_v1/` contains:

- `rq1/rq1_probability_boxplots_paired.pdf`: the original nine-panel chart, with
  renamed ticks and additional clearance between longer labels and sample counts.
- `textual_features_boxplots.pdf`: the same ten-page ordering; pages 9--10 retain
  historical raw-cosine plots for bundle compatibility, not the primary RQ3 result.
- `rq3/rq3_retrieval_calibration.pdf`: current retrieval/calibration chart.
- `presentation_manifest.json`: 27 source/input hashes, plotting environment,
  generated-file hashes, and 15 before/after plot-geometry checks (PDF/PNG saves).

The renderer loads cached row-level scores and summary CSVs. It does not load models,
generate descriptions, compute new metric scores, or run statistical tests. Existing
plot-only functions and selected notebook plotting cells are reused; historical
data keys and statistical bracket selection remain intact. Relabeling is applied
only when saving the figure. All input hashes were checked before and after the run;
plotted data coordinates, box geometry, tick locations, and axis limits were unchanged.

The initial cached retrieval summary and RQ3 plotting source were absent from the
live server checkout. Exact copies from this verified replication package were
staged in the new run directory and checked against the pinned hashes; no RQ3
analysis was rerun. These path overrides are recorded in the presentation manifest.

## Reproduce the presentation layer

### BM25 portrait follow-up (originally Figure 7)

The manuscript and current supplement use the standalone
`phase3/analysis/outputs/bm25_portrait_v2/bm25_retrieval_portrait.pdf` rather than
page 8 of the ten-page bundle. Three vertically stacked model panels replace the
horizontal strip that became unreadable at single-column width. The original
bundle and numerical result packages remain unchanged.

`phase3/analysis/presentation_labels/render_bm25_portrait.py` is the exact source
executed on Nokogiri. It loads all 5,322 frozen BM25 score rows, checks all 18 cell
counts, executes only the original source's plotting block, and rearranges those
same artists. The before/after hashes of box, whisker, and median coordinates,
axis limits, and tick positions are identical. No retrieval scoring, summary
computation, or statistical tests are run; outliers remain hidden as in the
original chart. Source/input hashes, environment versions, and output hashes are
in `bm25_portrait_v2/presentation_manifest.json`. The initial server-only v1 preview
was superseded during visual QA to restore y-axis labels on every stacked panel.

Reproduce on Nokogiri, using a new output directory:

```bash
/home/ayo-b/.pyenv/versions/3.12.9/bin/python -B \
  phase3/analysis/presentation_labels/render_bm25_portrait.py \
  --study-root /path/to/complete/study-tree \
  --output-dir /path/to/new/bm25-portrait-output
```

The supplementary figure copy is hash-checked against this asset by
`verify_public_package.py`. Historical label-refresh files and their verifier
anchors are retained, not replaced. The manuscript uses column width; the
single-column supplement uses 65% of its wider text block.

### Original context-label refresh

Run on Nokogiri using its existing `iwadi-main` Python 3.12.9 environment. The
complete study tree must provide the files listed in
`phase3/analysis/presentation_labels/context_label_inputs.json` (the replication
tree's `phase3/datasets` symlink supplies the dataset path). Use a **new** output
directory; the renderer refuses to overwrite a previous run.

```bash
/home/ayo-b/.pyenv/versions/3.12.9/bin/python \
  phase3/analysis/presentation_labels/render_context_labels.py \
  --study-root /path/to/complete/study-tree \
  --output-dir /path/to/new/presentation-output
```

If using the live experiment tree without the two exported RQ3 files, supply
`--rq3-plot-source` and `--retrieval-summary` with exact copies from this package.
The same hash checks apply. Do not substitute another dataset or run whole notebooks.

`python3 verify_public_package.py` checks the new presentation package alongside
the historical source/results anchors, including the supplement's current bundle.
It is an integrity check, not a rerun of the analyses. PDF binary hashes may differ
on a fresh render because of timestamps; numeric inputs and recorded geometry checks
must remain identical. No manuscript PDF or Overleaf build products are stored here.
