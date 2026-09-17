# Current RQ3 retrieval and calibration package

The manuscript now uses correct-diff retrieval as its primary RQ3 outcome and
matched-minus-mismatched cosine margins as complementary evidence. Raw cosine
outputs in `corrected_rq23_v1` are retained for provenance, not as the primary
association result.

## Evidence and execution sequence

1. `phase3/analysis/outputs/rq3_mismatched_calibration_v1/`: all 5,630 descriptions
   against 308 diff candidates; matched and nonmatched scores, correct-diff ranks,
   group summaries, and paired margin comparisons.
2. `phase3/analysis/outputs/rq3_embedding_postprocess_sensitivity_v1/`: mean-centering
   and removal of the leading one, two, and three principal components. Every
   variant is reported, not selected for a favourable outcome.
3. `phase3/analysis/outputs/rq3_retrieval_inference_v1/`: paired reciprocal-rank
   comparisons, BH-adjusted tests, rank-biserial effects, and paired MRR intervals.
4. `phase3/analysis/outputs/rq3_manuscript_figures_v1/`: the manuscript figure.

The exact source is under `phase3/analysis/rq3_mismatched_calibration/`. Its README
is the frozen initial diagnostic protocol; the later `POSTPROCESSING_PROTOCOL.md`
and `RETRIEVAL_INFERENCE_PROTOCOL.md` document the extensions. This sequence was
completed before promotion of retrieval to the main manuscript outcome.

Each result directory preserves its original completion manifest, source snapshots,
and independent verification report unchanged. `rq3_publication_manifest.json`
anchors the public selection. The calibration source was untracked in the local
experiment checkout at export; no experiment commit is claimed to contain it.
File hashes and the archived protocol snapshots establish the executed versions.

## Interpretation

GraphCodeBERT assigns higher PR-specific association to context-averaged AI
descriptions than to their paired human descriptions across all three generators.
The direction survives calibration and all post-processing variants, but absolute
retrieval is modest. Human MRR is 0.0364; AI MRRs are 0.0477--0.0623, against a
random-ranking reference of 0.0205. All three paired rank tests favour AI, while
the GPT mean-difference bootstrap interval includes zero. CodeBERT is generator-
and estimator-dependent. These measurements do not establish factual accuracy,
completeness, or reviewer usefulness.

## Publication boundary and verification

The public selection contains source, protocols, numeric per-record results,
summaries, metadata, and verification reports. It adds no raw descriptions, prompts,
private B2 records, credentials, or operational logs. Ten derived `.npy` embedding
and transformation caches are omitted and explicitly listed, with original hashes,
in the publication manifest. They can be regenerated from the published corpus
using the pinned model revisions and recorded environment; they are not required
to inspect the reported numeric results.

Run `python3 verify_public_package.py` for artifact integrity. This checks the
public selection, not a new scientific replication. To reconstruct scores from
embeddings, regenerate the omitted caches first in a separate output namespace;
the original analysis verifiers require the complete run artifacts. The launchers
retain the original server paths and environment guards rather than silently
substituting a different computation environment.
