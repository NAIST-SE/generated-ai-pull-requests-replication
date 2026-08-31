# Corrected-study replication provenance

This branch contains two intentionally separated provenance layers:

1. the historical flat-layout package synchronized before the genuine-issue audit; and
2. the current corrected-study package under `phase3/`, synchronized from the experiment
   repository after corrected generation, detector reruns, RQ2/RQ3 refresh, and independent
   verification.

The historical layer remains in Git to make the revision auditable. It is not the source of the
current manuscript results.

## Source repositories and commits

- Experiment repository: `JosephAyo/llm-in-prs`
- Branch: `revision/co-author-review`
- Corrected RQ2/RQ3 execution commit: `baed3dd7a161887e2c1e95c3ce84c2e63b2dab64`
- Publicly synchronized code head: `483f180` (`f07e747` adds the session diagnostics;
  `483f180` corrects detector/B4 status documentation)
- Heavy analyses were executed and verified on `nokogiri`; the workstation performed only
  copying, hashing, documentation adaptation, and public-package integrity checks.

## Current integrity anchors

| Artifact | SHA-256 |
|---|---|
| Corrected 5,630-text corpus | `07136384785d6d02d17762535ab51a2007a7a47b8c41b77fbb311ce9c8780306` |
| Corrected three-detector results | `b995b02848b4e7487682379afa22c6b7d6a8a06a404f35dac3ebca60daf9aa91` |
| Corrected corpus manifest | `546ab556c1b7e5030f72d8e86a29d009e31e069fbb98dad613ddd209ebe9c9a9` |
| Corrected detector completion | `a2b6d050d526ed941cc27023e767de0789c8faa117339ad7454d54c65a23b001` |
| Corrected Prelim-RQ/RQ1 verification report | `6b8bb6cbc056de2f7408352d80730b311bf0eab4fd4e74fa1ce2575e8a4ac074` |
| Corrected RQ1 probability figure | `07f0b8a0da6a31f881e8bf7eee210cd30cf21552d2840c49d727581326351af7` |
| Corrected RQ2/RQ3 completion | `bdd1866d1310a424c50b28a299435bda68bb3dc8f4eeb017f062ada42e9a08b5` |
| Corrected ten-page analysis figure | `ac84ef02dcb96cae44da7ee14ed5d3d4fc6ef1fe68dd708c2ccf21fa05b2b715` |
| Genuine-issue audit completion | `5f02893cc9b767358aa7da40bbe13735600c7fadf2dadd0d7ebb67418eb3cf89` |
| Session diagnostic verification | `229352273082f1531820185c2914c3049098f9f37f033a08940dabe67849a05a` |
| Outcome-sensitivity completion | `804d6bbb8effe44e5a9805ab5310576518da1573a2c7c825b5be868ddad9b64f` |
| Six-context bridge completion | `29cd3172dae3214297b2c624f99f1426794fb1c09ab539caa50047078c5ef89b` |

The embedded completion and verification files provide finer-grained hashes for their result
families.

## Corrected data lineage

The genuine-issue audit decomposed the historical aggregated artifact fields, checked every
reference against the corresponding PR body, verified artifact identity/type, and retained 273
PRs with at least one genuine issue. The corrected generation then produced:

- all Llama P1--P6 cells in scope;
- corrected Qwen/GPT P5--P6 cells for the 273 eligible PRs; and
- retained verified Qwen/GPT P1--P4 controls, excluding 12 unavailable P4 outputs.

The corrected corpus builder combines those sources into 5,322 AI descriptions and appends the
308 human descriptions. All 5,630 texts were successfully scored by ZeroGPT and both pinned
RoBERTa detectors before finalization.

## Public-layout policy

The exact corrected source layout is retained under `phase3/`; `phase3/datasets` is a relative
symlink to the public root `datasets/`. This avoids silently rewriting tested paths while avoiding
duplicate data.

Included:

- corrected corpus and detector results;
- corrected source code and immutable model/configuration metadata;
- verified numeric/hashed result packages;
- executed notebooks and figures;
- genuine-issue audit outputs needed to reproduce corrected generation; and
- raw-text-free generation-session diagnostics.

Excluded:

- credentials and token files;
- API retry and scoring logs;
- local caches and Python bytecode;
- newly generated diagnostic description CSVs and smoke outputs;
- raw corrected generation CSVs containing large stored prompts (their descriptions are already
  present in the corrected corpus and their hashes are preserved in the generation/corpus
  manifests); and
- `b4-linked-issue-reuse-v1`, because it was executed before the genuine-issue audit and is not
  valid evidence for the corrected construct.

`phase3/analysis/outputs/corrected_rq23_v1/completion.json` is preserved from the complete server
run. Its source-package hash list includes one Python bytecode file; that bytecode is deliberately
omitted from the public package. `verify_public_package.py` verifies the curated public selection
instead.

## Historical pre-B4 layer

The earlier package was synchronized from experiment commit
`4037f0bf4aaffd6e39a88a92a7be92df6bab917e`. Its key artifacts include
`datasets/temperature_1/run_3_cleaned/` and `analysis/outputs/deep_audit/`. Those files are frozen
for provenance, but their P5/P6 population and dependent analyses were superseded by the genuine-
issue repair. Documentation and code inside that historical tree may still contain the old 5,526
AI-row expectations; these are historical assertions, not current instructions.
