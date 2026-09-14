# B1 additional zero-shot detector run

The active B1 extension scores the frozen corrected corpus with:

- **Binoculars:** official Falcon-7B/Falcon-7B-Instruct pair, BF16, 512-token limit, official
  low-FPR threshold.

The package also retains a fully specified **Fast-DetectGPT** implementation using the
maintainers' January 2026
  Llama-3-8B/Llama-3-8B-Instruct recommendation, BF16, 8,192-token limit, and the official
  pair-specific Gaussian calibration. It is a contingency only: do not run or include it in
  the active analysis unless a TSE revision explicitly requests it and the protocol is amended.

The input is exactly 5,630 records from
`run_3_corrected_detection_v1/corpus.csv`, SHA-256
`07136384785d6d02d17762535ab51a2007a7a47b8c41b77fbb311ce9c8780306`.
The scripts do not rebuild or normalize the text.

## Binoculars execution route

The accepted Binoculars output was produced on the collaborator's 64-GB Apple M4 Pro through
the reviewed MPS fork. Its archive, source revision, exact repeated-smoke scores, reported
package inventory, model revisions, and hardware record are provenance-gated by the importer.

## Contingency GPU requirements

These requirements apply only if a future TSE revision requests Fast-DetectGPT.

- Linux host approved for study computation;
- NVIDIA GPU with BF16 and at least 48 GB VRAM (one GPU is sufficient; two can be used by the
  official Binoculars implementation);
- at least 100 GB free local storage for four pinned model snapshots and environments;
- Python 3.10--3.12, Git, and outbound HTTPS during setup;
- a Hugging Face account that has accepted the Meta Llama 3 licence; and
- `HF_TOKEN` exported only in the shell/session. Never write it to this repository, a script,
  terminal transcript, or chat.

The frozen Python environment uses PyTorch 2.4.1 and Transformers 4.44.2. It uses a newer
Transformers version than the 2023 Fast-DetectGPT environment because the selected Llama-3
checkpoints did not exist then. The detector mathematics come directly from the pinned official
repositories; model loading is made explicit to prevent Fast-DetectGPT's current demo alias from
falling back to mutable or FP32 defaults.

## Contingency one-time setup

From the experiment repository root:

```bash
export B1_RUNTIME_ROOT=/absolute/lab/path/b1-runtime
bash phase3/detection/b1_modern/run_b1.sh probe

export B1_APPROVED_HOST=1
export HF_TOKEN='set-this-in-the-shell-only'

bash phase3/detection/b1_modern/run_b1.sh setup
bash phase3/detection/b1_modern/run_b1.sh models
bash phase3/detection/b1_modern/run_b1.sh prepare
bash phase3/detection/b1_modern/run_b1.sh preflight
```

Run `probe` before installation and retain its output with the lab notes. If it reports a GPU
architecture unsupported by PyTorch 2.4.1 (for example, Blackwell), stop and amend the frozen
runtime rather than changing packages inside an existing B1 output directory.

`setup` clones detached, clean snapshots of the official repositories and installs the pinned
environment. `models` downloads only configuration, tokenizer, remote-code, and safetensor files
for the four exact Hugging Face revisions. `prepare` verifies the corpus and creates metadata-only
input/smoke manifests under the versioned output directory. `preflight` fails unless the checkouts,
official detector entry points, snapshots, CUDA/BF16 support, and approximately 48 GB GPU capacity
are all present.

## Contingency smoke tests (do not skip)

If the contingency is activated, each command is resumable and safe inside `tmux`. The
following Binoculars commands document the internal CUDA route but are not needed for the
already accepted collaborator output:

```bash
tmux new -s b1-binoculars
bash phase3/detection/b1_modern/run_b1.sh smoke binoculars 1
bash phase3/detection/b1_modern/run_b1.sh smoke binoculars 2
bash phase3/detection/b1_modern/run_b1.sh verify-smoke binoculars
bash phase3/detection/b1_modern/run_b1.sh reference-parity binoculars
```

Detach with `Ctrl-b d`; reattach with `tmux attach -t b1-binoculars`. Run Fast-DetectGPT only
after the amendment and on an approved GPU:

```bash
tmux new -s b1-fast-detectgpt
bash phase3/detection/b1_modern/run_b1.sh smoke fast-detectgpt 1
bash phase3/detection/b1_modern/run_b1.sh smoke fast-detectgpt 2
bash phase3/detection/b1_modern/run_b1.sh verify-smoke fast-detectgpt
bash phase3/detection/b1_modern/run_b1.sh reference-parity fast-detectgpt
```

The ten deterministic cases cover human/AI, all three generators, shortest/longest cases, and
Markdown/code. Selection uses no detector output. Review each `verification.json` for all-ten
success, repeatability within absolute `1e-6`/relative `1e-5`, truncation, runtime, and projected
full-run time.

`reference-parity` recomputes the first smoke case through the pinned official scoring entry point,
checks it against the recorded wrapper result, and saves only the case ID/hash and numerical
comparison. Review that report, then record the approved infrastructure, licence review,
responsible researcher, and cost ceiling:

```bash
bash phase3/detection/b1_modern/run_b1.sh approve binoculars \
  --approved-by 'RESEARCHER NAME' \
  --infrastructure 'LAB HOST AND GPU' \
  --cost-ceiling 'APPROVED LIMIT OR INSTITUTIONAL GPU' \
  --configuration-approved --licences-reviewed --manual-reference-parity-confirmed
```

Fast-DetectGPT approval cannot be recorded before both smoke repeats pass. If the team changes a
model, precision, maximum length, threshold, upstream commit, or active detector set, record a
protocol amendment instead of silently editing an existing output directory.

## Full scoring and analysis

The collaborator-produced Binoculars archive must not be copied into the internal CUDA-run
directory. Create the provenance JSON from
`external_binoculars_provenance.json.example`, change its status to `accepted_for_analysis`, and
run the explicit importer from the repository root:

```bash
export B1_OUTPUT_ROOT=/absolute/lab/path/b1-output
export B1_PYTHON=/absolute/path/to/analysis-python

bash phase3/detection/b1_modern/run_b1.sh import-binoculars \
  --archive /secure/path/b1-20260908T140156Z-1-001.zip \
  --handoff-manifest /secure/path/handoff/manifest.json \
  --reidentification-map /secure/path/local-only/reidentification_map.csv \
  --package-inventory /secure/path/provenance/binoculars-package-inventory.txt \
  --provenance /secure/path/provenance/accepted-binoculars-provenance.json
```

The importer verifies the immutable archive, handoff, private map, corrected-corpus manifest,
score direction, threshold labels, attempts, token accounting, source hashes, smoke receipt,
package-inventory hash, and exact model revisions. It emits no raw text and refuses to write
accepted scores while the provenance status is pending.

For the active Binoculars-only analysis:

```bash
bash phase3/detection/b1_modern/run_b1.sh verify-full --use-external-binoculars
bash phase3/detection/b1_modern/run_b1.sh analyse
bash phase3/detection/b1_modern/run_b1.sh verify-analysis
```

If Fast-DetectGPT is later requested, run its smoke, parity, approval, and full-run gates before
amending the active detector set and creating a new analysis output. Do not mix it into the
already frozen Binoculars analysis directory.

Do not execute `analyse` until the implementation-readiness audit has been resolved. The final
analysis must use the native continuous criterion with a monotonic sign reversal for Binoculars,
the published threshold, PR-clustered bootstrap intervals, and the frozen paired-test/FDR plan.
It compares Binoculars with the three already verified original detectors, reports
generator-by-context views, and uses a strict all-detector common-set sensitivity; failures are
never imputed as zero.

## Outputs and handling

All runtime outputs are under
`phase3/analysis/outputs/tier_b/b1-modern-detectors-v1/` unless `B1_OUTPUT_ROOT` is explicitly set.
They include metadata-only manifests, smoke repeats, execution gates, append-only attempt logs,
the provenance-accepted external Binoculars table, accepted score tables, full verification, and
machine-readable analysis tables. Raw descriptions remain only in the existing corrected corpus
and are not copied into B1 outputs.

The team approved B1 manuscript integration on 2026-09-14 after an independent verifier checked
counts, hashes, model/configuration revisions, score direction, and the repeated smoke cases. The
public replication selection includes this exact source, aggregate text-free tables, independent
verification, and sanitized reproducibility metadata. The record-level score table, private
handoff materials, returned archive, screenshots, and operational logs remain excluded pending
separate publication approval.
