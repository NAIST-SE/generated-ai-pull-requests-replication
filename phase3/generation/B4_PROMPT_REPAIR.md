# B4 P5/P6 prompt-source repair

The corrected P5/P6 path consumes only the frozen output of
`b4-issue-source-audit-v1/regeneration_sources.jsonl`. It does not parse the legacy aggregate
`selected_prs_with_related_issues.csv` during generation.

Fail-closed guarantees:

- the manifest SHA-256 must equal the frozen audit hash;
- it must contain 273 unique PRs and 293 genuine `/issues/` artifacts;
- every stored title, body, combined prompt block, and hash must reconstruct exactly;
- multi-issue order is preserved from the audited reference order;
- all three builders use the same issue-section renderer;
- legacy aggregate `title`/`body` and `issue_titles`/`issue_bodies` fallbacks are disabled;
- a rendered issue block equal to the target PR title/body is rejected; and
- P5/P6 generation requires the audited manifest, while the 35 pull-only PRs are skipped only for
  issue-exposed contexts.

Run synthetic tests locally or on Nokogiri:

```bash
python -m pytest -q \
  phase3/generation/test_issue_sources.py \
  phase3/generation/test_generation_experiments.py
```

Run the exhaustive real-data prompt smoke only on Nokogiri. It makes zero model/API calls:

```bash
/home/ayo-b/.pyenv/versions/3.12.9/bin/python \
  phase3/generation/smoke_test_issue_prompts.py \
  --repo-root /mnt/ext-hdd1/ayo/llm-in-prs
```

Then independently verify the new output directory before any corrected generation:

```bash
/home/ayo-b/.pyenv/versions/3.12.9/bin/python \
  phase3/generation/verify_issue_prompt_smoke.py \
  --output-dir phase3/analysis/outputs/tier_b/b4-prompt-smoke-v1 \
  --report phase3/analysis/outputs/tier_b/b4-prompt-smoke-v1/independent_verification.json
```

Passing the smoke test authorizes review of prompt construction only. It does not authorize model
generation, B4 scoring, B2 packet distribution, or manuscript changes.
