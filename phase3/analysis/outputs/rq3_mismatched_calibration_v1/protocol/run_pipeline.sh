#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ANALYSIS_DIR/../.." && pwd)"

CORPUS="$REPO_ROOT/phase3/datasets/temperature_1/run_3_corrected_detection_v1/corpus.csv"
DIFFS="$REPO_ROOT/phase3/datasets/selected_prs_with_diffs_and_file_contents.csv"
CORRECTED_ROOT="$ANALYSIS_DIR/outputs/corrected_rq23_v1/notebooks/outputs"
EXPECTED_CODEBERT="$CORRECTED_ROOT/row_level/codebert_metrics.csv"
EXPECTED_GRAPHCODEBERT="$CORRECTED_ROOT/row_level/graphcodebert_metrics.csv"
OUTPUT_ROOT="$ANALYSIS_DIR/outputs/rq3_mismatched_calibration_v1"

CORPUS_SHA256="07136384785d6d02d17762535ab51a2007a7a47b8c41b77fbb311ce9c8780306"
DIFFS_SHA256="45c71d0170fc9343d999998ad6515e9b5b8298f63b66215f4110b3911f681d0c"
CODEBERT_ROWS_SHA256="932dbb166dfcc221e48a7696161ed541e24057e059bc79e057bd1e331e5c1550"
GRAPHCODEBERT_ROWS_SHA256="2c96eb065c971f07fc21ffd145f9cbab097695b532454303504b8c1875b2820b"
CODEBERT_REVISION="3b0952feddeffad0063f274080e3c23d75e7eb39"
GRAPHCODEBERT_REVISION="2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d"

PYTHON_BIN="${RQ3_CALIBRATION_PYTHON:-/home/ayo-b/.pyenv/versions/3.12.9/bin/python}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

LIBCUDA="/usr/lib/x86_64-linux-gnu/libcuda.so.1"
if [ -f "$LIBCUDA" ]; then
  export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${LIBCUDA}"
fi

if [ "$(hostname -s)" != "nokogiri" ]; then
  echo "ERROR: the RQ3 calibration must run on nokogiri." >&2
  exit 1
fi

check_sha() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "$path" | awk '{print $1}')"
  if [ "$observed" != "$expected" ]; then
    echo "ERROR: SHA-256 mismatch for $path" >&2
    echo "  expected: $expected" >&2
    echo "  observed: $observed" >&2
    exit 1
  fi
}

echo "=== RQ3 mismatched-pair calibration preflight ==="
echo "Host: $(hostname -s)"
echo "Repository: $REPO_ROOT"
echo "Commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
check_sha "$CORPUS" "$CORPUS_SHA256"
check_sha "$DIFFS" "$DIFFS_SHA256"
check_sha "$EXPECTED_CODEBERT" "$CODEBERT_ROWS_SHA256"
check_sha "$EXPECTED_GRAPHCODEBERT" "$GRAPHCODEBERT_ROWS_SHA256"
echo "✓ Frozen inputs and reference row scores match"

for required in \
  "/home/ayo-b/.cache/huggingface/hub/models--microsoft--codebert-base/snapshots/$CODEBERT_REVISION/config.json" \
  "/home/ayo-b/.cache/huggingface/hub/models--microsoft--graphcodebert-base/snapshots/$GRAPHCODEBERT_REVISION/config.json"; do
  if [ ! -f "$required" ]; then
    echo "ERROR: missing pinned offline model snapshot: $required" >&2
    exit 1
  fi
done
echo "✓ Pinned CodeBERT and GraphCodeBERT snapshots are present"

"$PYTHON_BIN" -c 'import numpy,pandas,scipy,statsmodels,torch,transformers; assert torch.cuda.is_available(); print(f"✓ CUDA: {torch.__version__} / {torch.cuda.get_device_name(0)}")'
"$PYTHON_BIN" "$SCRIPT_DIR/analysis.py" \
  --preflight \
  --corpus "$CORPUS" \
  --diffs "$DIFFS" \
  --expected-codebert "$EXPECTED_CODEBERT" \
  --expected-graphcodebert "$EXPECTED_GRAPHCODEBERT"

if [ "${1:-}" = "--preflight" ]; then
  echo "=== Preflight complete; no study embeddings or metrics were computed ==="
  exit 0
elif [ -n "${1:-}" ]; then
  echo "Usage: bash rq3_mismatched_calibration/run_pipeline.sh [--preflight]" >&2
  exit 2
fi

if [ -e "$OUTPUT_ROOT" ]; then
  echo "ERROR: output namespace already exists; refusing to overwrite: $OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT/protocol"
cp "$SCRIPT_DIR/README.md" "$SCRIPT_DIR/analysis.py" "$SCRIPT_DIR/verify_outputs.py" "$SCRIPT_DIR/run_pipeline.sh" "$OUTPUT_ROOT/protocol/"
exec > >(tee -a "$OUTPUT_ROOT/pipeline.log") 2>&1

echo "=== Running frozen RQ3 calibration ==="
"$PYTHON_BIN" "$SCRIPT_DIR/analysis.py" \
  --corpus "$CORPUS" \
  --diffs "$DIFFS" \
  --expected-codebert "$EXPECTED_CODEBERT" \
  --expected-graphcodebert "$EXPECTED_GRAPHCODEBERT" \
  --output-root "$OUTPUT_ROOT"

echo "=== Independently reconstructing outputs ==="
"$PYTHON_BIN" "$SCRIPT_DIR/verify_outputs.py" \
  --output-root "$OUTPUT_ROOT" \
  --expected-codebert "$EXPECTED_CODEBERT" \
  --expected-graphcodebert "$EXPECTED_GRAPHCODEBERT"

echo "=== RQ3 calibration complete and verified ==="
