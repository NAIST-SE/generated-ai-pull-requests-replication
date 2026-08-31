# B4 linked-issue reuse analysis

Status: corrected v2 implementation frozen for the verified v3.2 generations; synthetic tests,
source preflight, result-bearing execution, and independent verification are separate gates.

The historical `outputs/tier_b/b4-linked-issue-reuse-v1/` run is invalid. Its prompt source was
the target PR title/body and it included pull-request artifacts returned through GitHub's issues
endpoint. It is retained only as diagnostic provenance and must never support a linked-issue
reuse claim.

`b4-linked-issue-reuse-v2` consumes exactly:

- the hash-pinned 273-PR `b4-issue-source-audit-v1/regeneration_sources.jsonl` manifest;
- the complete-and-verified `b4-corrected-generation-v3.2` outputs;
- unchanged run-3 Qwen/GPT P-1--P-4 controls selected by the corrected-corpus builder; and
- the hash-pinned 5,630-row corrected detector corpus, filtered to the 273 eligible PRs and
  byte-checked against the raw generation files.

The v2 analysis contains 4,902 AI descriptions and 273 human descriptions. Qwen and GPT each
have 267 eligible P-4 controls because six historical P-4 calls per model did not finish
successfully; their P-5-versus-P-4 paired tests therefore use `n=267`. Every other predeclared
contrast uses `n=273`.

For Qwen and GPT, the unchanged P-1--P-4 controls and corrected P-5/P-6 outputs come from
different generation sessions; Llama's six contexts were regenerated together under the pinned
CoreWeave route. The session difference remains a limitation on exact Qwen/GPT context-effect
estimates and is recorded in the output metadata. B4 is interpreted as an information-transfer
robustness analysis, not as an isolated causal effect of adding issue prose.

## Frozen outcomes

The confirmatory metric is conservative-normalized occurrence-level generated 5-gram
containment. The primary family is P-5 versus P-1--P-4 for each generator, with one BH
correction across 12 paired Wilcoxon tests. Matched-pairs rank-biserial correlation is the native
paired effect size; Cliff's delta is retained only for continuity. Separate secondary families
cover P-6, P-5 versus P-6, BERTScore F1, and BM25 reciprocal rank. Alternative n-grams,
ROUGE-L, longest spans, title/body attribution, raw-view scores, human baselines, and the
closest-length unrelated-issue baseline are descriptive robustness evidence.

The corrected issue manifest contains 265 unique rendered blocks among 273 PRs: eight blocks
occur for two PRs. Before any v2 result was viewed, the unrelated-control selector was amended
to exclude candidates with the same issue-text SHA-256, as well as the target PR itself. This is
a fail-closed correction to the descriptive unrelated-issue diagnostic only; it does not change
the confirmatory, BERTScore, or BM25 estimands. BM25 retains all 273 PR-indexed issue documents
and assigns average rank to score ties.

Study-data execution is fail-closed to Nokogiri. Only synthetic tests may run elsewhere. The
result directory is immutable and the program refuses to overwrite it.

```bash
cd /mnt/ext-hdd1/ayo/agent-worktrees/b4-linked-issue-reuse/phase3/analysis/b4_linked_issue_reuse
/home/ayo-b/.pyenv/versions/3.12.9/bin/python -m unittest -v \
  test_b4_metrics.py test_b4_pipeline_synthetic.py

cd /mnt/ext-hdd1/ayo/agent-worktrees/b4-linked-issue-reuse
/home/ayo-b/.pyenv/versions/3.12.9/bin/python \
  phase3/analysis/b4_linked_issue_reuse/analysis_b4_linked_issue_reuse.py \
  --repo-root /mnt/ext-hdd1/ayo/llm-in-prs \
  --preflight-only

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  /home/ayo-b/.pyenv/versions/3.12.9/bin/python \
  phase3/analysis/b4_linked_issue_reuse/analysis_b4_linked_issue_reuse.py \
  --repo-root /mnt/ext-hdd1/ayo/llm-in-prs \
  --with-bertscore

/home/ayo-b/.pyenv/versions/3.12.9/bin/python \
  phase3/analysis/b4_linked_issue_reuse/verify_b4_outputs.py \
  --repo-root /mnt/ext-hdd1/ayo/llm-in-prs
```

No output may be interpreted until `completion.json` reports `complete` and the separately
written v2 verification report reports `verified`.
