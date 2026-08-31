# Corrected rerun of the original three detectors

Status: corrected corpus construction, all three detector runs, finalization, and independent
verification complete. The finalized corpus is
`phase3/datasets/temperature_1/run_3_corrected_detection_v1/detection_results.csv`; the
authoritative Prelim-RQ/RQ1 outputs are under
`phase3/analysis/outputs/corrected_detection_v1/`.

This directory replaces the unsafe global-file workflow in `prepare_data.py` and
`run_detection.py` for the corrected study corpus. It never overwrites the historical
`temperature_1/run_3` or `run_3_cleaned` artifacts.

## Frozen corpus

The builder verifies SHA-256 hashes before reading any input and writes a versioned corpus to
`phase3/datasets/temperature_1/run_3_corrected_detection_v1/`:

- 308 human PR descriptions;
- 2,452 unchanged Qwen/GPT control descriptions from P-1--P-4, retaining only structured
  `finish_reason=stop` rows;
- all 2,870 verified corrected-generation rows: Llama P-1--P-6 and Qwen/GPT P-5--P-6;
- 5,630 detector entries total (5,322 AI and 308 human).

Raw variation names are normalized and validated as `P-1`--`P-6`; the builder fails if the
model/PR/context key set or frozen per-cell counts differ. Every row records its text hash,
source-file hash, source row, generation state, and deterministic scoring order. The generation
files' legacy `prompt_hash` field is explicitly verified as MD5 (the algorithm used by the
generators), while the corpus adds an unambiguous SHA-256 of the full stored prompt.

## Corrected detector methods

- **ZeroGPT:** send the complete description to the same `api.zerogpt.com` endpoint used by the
  historical code. Do not apply the historical 15,000-character cut. Validate HTTP status and
  the response schema strictly; missing score/label fields are failures, never zero-valued
  successes. Use `fakePercentage / 100` as the continuous AI score. ZeroGPT's numeric `isHuman`
  verdict uses values below 50 for AI-leaning categories, 50 for mixed, and values above 50 for
  human-leaning categories; therefore flag `isHuman < 50` as AI and conservatively leave mixed
  unflagged. The proprietary service does not expose a pin-able model revision, so the run records
  response metadata and UTC time and must be described as a current-service rerun.
- **ChatGPT Detector RoBERTa:** use the frozen Hugging Face revision
  `d2b342c61775d5dd0221808a79983ed3b86ffd86`. Score every non-empty description. Tokenization
  uses the checkpoint's 512-token limit and records both untruncated and used token counts.
- **ChatGPT-QA Detector RoBERTa:** use frozen revision
  `25aad95e922d92631f86164642a3db104deee087`. The authors' official evaluation passes a
  question and answer as a paired tokenizer input. For this PR-domain adaptation, use the PR
  title as the question and the description as the answer for both human and AI rows. This
  common question policy prevents generation-context text from entering the detector input.

The old 250-character gate is removed. It was a local pipeline rule, not a checkpoint
requirement, and exactly caused the historical RoBERTa “failures.” An optional
`roberta-qa-legacy-text-only` adapter exists only to diagnose the historical misapplication; it
is not part of the confirmatory three-detector output.

The paired-input choice follows the detector authors' official implementation, which supplies
`text=question` and `text_pair=answer` with 512-token truncation:
https://github.com/Hello-SimpleAI/chatgpt-comparison-detection/blob/main/detect/dl_eval.py

## Server workflow

Run from the repository root. On Nokogiri, the launcher automatically selects the approved
`/home/ayo-b/.pyenv/versions/3.12.9/bin/python` environment and preloads
`/usr/lib/x86_64-linux-gnu/libcuda.so.1` for local-detector preflight/scoring. Elsewhere, set the
task-specific `DETECTION_PYTHON` variable explicitly. Do not use Nokogiri's system `python3`; it
does not contain Torch. Local detectors also run with `HF_HUB_OFFLINE=1`, so a missing frozen
asset fails instead of silently downloading a different snapshot.

The equivalent direct commands are:

```bash
python3 -m phase3.detection.corrected_rerun.build_corpus

python3 -m phase3.detection.corrected_rerun.run_detector \
  --detector roberta --preflight-only
python3 -m phase3.detection.corrected_rerun.run_detector \
  --detector roberta-qa --preflight-only
python3 -m phase3.detection.corrected_rerun.run_detector \
  --detector zerogpt --preflight-only \
  --tokens-file /absolute/path/to/tokens.txt
```

Preflight makes no detector calls and does not load model weights. After approval, run the two
local detectors sequentially on one GPU; do not use the legacy five-worker mode, which loads
multiple copies of each model. Run ZeroGPT separately so API retries and temporal metadata are
isolated.

```bash
python3 -m phase3.detection.corrected_rerun.run_detector --detector roberta --device cuda
python3 -m phase3.detection.corrected_rerun.run_detector --detector roberta-qa --device cuda
python3 -m phase3.detection.corrected_rerun.run_detector \
  --detector zerogpt --tokens-file /absolute/path/to/tokens.txt

python3 -m phase3.detection.corrected_rerun.finalize finalize
python3 -m phase3.detection.corrected_rerun.finalize verify
```

Each detector writes an append-only attempt log, a successful-results CSV, and a status
manifest. Resume is allowed only when the corpus and effective detector-configuration hashes
match. Finalization fails unless all 5,630 entries have a valid probability from all three
detectors. No failed call is imputed as probability zero.
