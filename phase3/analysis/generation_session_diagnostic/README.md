# Generation-session sensitivity diagnostic

This is a **diagnostic**, not a confirmatory extension of the study. It tests whether the
historical Qwen and GPT P1--P4 descriptions are sensitive to being regenerated under the
current corrected generation protocol.

## Estimand and interpretation

For the same PR and the same P1--P4 prompt, compare the historical run-3 description with a
new description from the same named model. The comparison is deliberately called a
**combined generation-protocol sensitivity diagnostic**, not a pure temporal-drift test:

- the historical Qwen and GPT runners recorded seed 42 but did not send it to the APIs;
- the current runners send seed 42;
- the API execution date and possibly backend implementation differ; and
- Qwen remains pinned to Hugging Face/Novita BF16 while GPT remains pinned to OpenAI with
  temperature omitted.

The diagnostic can reveal whether the mixed generation sessions are plausibly consequential.
It cannot isolate which of the changes caused any difference, and a 50-PR sample cannot
establish equivalence.

## Frozen design

- Population: PRs in the hash-pinned 273-PR genuine-issue manifest that have successful,
  unique historical Qwen and GPT descriptions in every P1--P4 context.
- Sample: 50 PRs sampled without replacement from the sorted complete-case population using
  Python's `random.Random(20260831)`.
- Pairing: the same 50 PRs are used for both models and all four contexts.
- Unique full-run tasks: 50 PRs x 4 contexts x 2 models = 400, after a two-call
  smoke test. Persistent failures may create additional request attempts under the frozen
  three-pass retry policy and must be reported separately from unique tasks.
- Qwen: `Qwen/Qwen2.5-72B-Instruct`, Hugging Face router, Novita provider, BF16,
  temperature 1, request seed 42.
- GPT: `gpt-5-mini-2025-08-07`, OpenAI, provider-managed precision, temperature omitted,
  request seed 42, JSON-object response mode.
- Prompts: the preflight must prove byte-for-byte equality between the newly constructed
  prompt and the historical prompt for all 400 pairs before any full-run call.
- Selection and prompt manifests record SHA-256 hashes of every source file and frozen ID set.

## Outputs

The default output directory is
`phase3/analysis/outputs/tier_b/generation-session-diagnostic-v1/` and contains:

- `sample_manifest.json`: population and deterministic sample provenance;
- `prompt_preflight.json`: prompt-equality result without prompt text;
- `smoke/`: the two paid smoke-test rows;
- `qwen_current.csv` and `gpt_current.csv`: resumable current-session outputs;
- `verification.json`: completion and provenance checks;
- `paired_comparison.csv`: text hashes and pairwise descriptive metrics, without raw text;
- `summary.csv` and `summary.json`: model/context summaries and bootstrap intervals.
- `bertscore_pairs.csv` and `bertscore_summary.{csv,json}`: offline semantic similarity
  using the corrected-RQ2 pinned DeBERTa snapshot and baseline, again without raw text.

Raw prompts and descriptions remain in the generation CSVs. Do not paste them into chat or
other reports without explicit approval.

## Server-only procedure

Run from the Nokogiri repository root. The script refuses to run on other hosts unless a
model-free test explicitly supplies `--allow-non-server`.

```bash
python phase3/analysis/generation_session_diagnostic/generation_session_diagnostic.py select
python phase3/analysis/generation_session_diagnostic/generation_session_diagnostic.py preflight

# Requires HF_TOKEN and OPENAI_API_KEY in the process environment.
python phase3/analysis/generation_session_diagnostic/generation_session_diagnostic.py smoke
python phase3/analysis/generation_session_diagnostic/generation_session_diagnostic.py run
python phase3/analysis/generation_session_diagnostic/generation_session_diagnostic.py verify
python phase3/analysis/generation_session_diagnostic/generation_session_diagnostic.py analyze

# Offline GPU semantic follow-up using the study's pinned BERTScore configuration.
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  /home/ayo-b/.pyenv/versions/3.12.9/bin/python \
  phase3/analysis/generation_session_diagnostic/semantic_similarity.py
```

The run is resumable: successful `(pr_id, variation)` pairs are retained, while unsuccessful
pairs are retried. `verify` distinguishes a fully successful run from a structurally verified
partial run with persistent endpoint failures. Failed sampled pairs are never replaced,
truncated, or silently omitted; `failure_audit.csv` records their safe metadata, and summaries
show both observed and expected pair counts.
