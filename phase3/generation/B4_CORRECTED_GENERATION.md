# Corrected B4 generation

`run_generation.sh` is the server-only launcher for the corrected Linked Issue
regeneration. The v3.1 protocol responds to the execution-time change in OpenRouter's
available BF16 routes for Llama 3.1 70B: it pins CoreWeave with fallbacks disabled and
regenerates every Llama context through that same provider. This avoids comparing new
CoreWeave P5/P6 outputs with historical P1-P4 outputs from another provider. It also sends
the recorded seed to every API route and records any returned system fingerprint. The v2
paid pilot exposed that the earlier runners recorded `seed=42` without placing it in the
request; those ten rows are retained as a diagnostic and cannot enter the study.
Provider documentation treats seeded sampling as best-effort rather than guaranteed
determinism, so fingerprints and exact output hashes remain part of the provenance record.

Protocol v3.1 added the approved P4 context-limit rule. Seven Llama P4 prompts repeatedly
requested 131,073 total tokens against CoreWeave's 131,072-token ceiling; the other 301
P4 prompts completed and had at most 85,840 prompt tokens. The launcher therefore applies
the existing head/tail-preserving file-content truncation only when P4 file content exceeds
a 90,000-token budget. The full-run verifier requires exactly seven Llama P4 rows to carry
positive truncation metadata no greater than 90,000 and rejects truncation metadata on any
other model or context.

Protocol v3.2 adds an explicit 4,096-token completion cap only to the seven truncated
Llama P4 requests. Under v3.1, CoreWeave accepted five of those requests but rejected two
because its implicit completion reservation made each request exactly one token larger
than the 131,072-token context window. The cap is deliberately non-binding for this task:
the 1,776 successful Llama descriptions observed before the amendment used 12--200 output
tokens, including 79--160 for the five successful truncated prompts. To keep the seven
amended rows under identical request parameters, all seven are regenerated under v3.2.
The verifier requires `max_tokens=4096` in the recorded decoding parameters exactly when
Llama P4 truncation metadata is present and rejects that cap on every other row.

| Model | Contexts and PRs | Router | Provider | Precision | Decoding |
|---|---|---|---|---|---|
| Llama 3.1 70B | P1-P4: 308; P5/P6: 273 | OpenRouter | CoreWeave | bf16 | temperature 1, request seed 42; P4 file content capped at 90,000 tokens only when needed; those truncated P4 requests use `max_tokens=4096` |
| Qwen 2.5 72B | P5/P6: 273 | Hugging Face Inference Providers | Novita | bf16 | temperature 1, request seed 42 |
| GPT-5-mini | P5/P6: 273 | OpenAI | OpenAI | provider-managed | temperature omitted as in run 3, request seed 42 |

OpenRouter requests specify `provider.order=["coreweave"]`,
`provider.allow_fallbacks=false`, and `provider.quantizations=["bf16"]`. A request must
fail instead of silently using another provider or precision.

The launcher reads credentials only from `OPENROUTER_API_KEY`, `HF_TOKEN`, and
`OPENAI_API_KEY`. It never places credentials in command-line arguments or logs. Do not
commit or paste their values.

Run a model-free launcher preflight on Nokogiri:

```bash
bash run_generation.sh --dry-run --pilot --workers 1 --run-id b4-corrected-v3-dry-run
```

Run a paid pilot using the same eligible PR for every requested context. This makes ten
calls: six for Llama and two each for Qwen and GPT-5-mini.

```bash
bash run_generation.sh --pilot --workers 1 --run-id b4-corrected-v3-pilot
```

After inspecting and approving the pilot, run the complete protocol:

```bash
bash run_generation.sh --run-id b4-corrected-v3
```

The complete run makes 2,870 calls: 1,778 Llama descriptions, 546 Qwen descriptions,
and 546 GPT-5-mini descriptions. The launcher is resumable while its output directory is
incomplete and refuses to modify a run after `completion.json` exists.

Completion requires all expected model/context rows, the hash-pinned 308-PR sample and
273-PR issue manifest, identical issue-context PR sets across generators, exact P5/P6
issue blocks, the pinned models/providers/precision, the frozen seed and decoding
settings, `stop` finish reasons, positive token counts, and no empty, error, or mock
descriptions. Generation is not complete merely because the provider processes exit:
`completion.json` must exist and report `complete_and_verified` under protocol
`b4-corrected-generation-v3.2`.
