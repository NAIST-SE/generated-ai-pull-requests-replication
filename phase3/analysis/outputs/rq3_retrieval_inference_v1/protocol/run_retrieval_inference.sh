#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CALIBRATION_ROOT="$ANALYSIS_DIR/outputs/rq3_mismatched_calibration_v1"
OUTPUT_ROOT="$ANALYSIS_DIR/outputs/rq3_retrieval_inference_v1"
PYTHON_BIN="${RQ3_RETRIEVAL_PYTHON:-python3}"

echo "=== RQ3 paired retrieval-inference preflight ==="
echo "Host: $(hostname -s)"
echo "Analysis directory: $ANALYSIS_DIR"
"$PYTHON_BIN" -c 'import numpy,pandas,scipy,statsmodels; print("Required numerical packages import")'
"$PYTHON_BIN" "$SCRIPT_DIR/retrieval_inference.py" \
  --preflight \
  --calibration-root "$CALIBRATION_ROOT"

if [ "${1:-}" = "--preflight" ]; then
  echo "=== Preflight complete; no retrieval statistic was written ==="
  exit 0
elif [ -n "${1:-}" ]; then
  echo "Usage: bash rq3_mismatched_calibration/run_retrieval_inference.sh [--preflight]" >&2
  exit 2
fi

if [ -e "$OUTPUT_ROOT" ]; then
  echo "ERROR: output namespace already exists; refusing to overwrite: $OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT/protocol"
cp \
  "$SCRIPT_DIR/RETRIEVAL_INFERENCE_PROTOCOL.md" \
  "$SCRIPT_DIR/retrieval_inference.py" \
  "$SCRIPT_DIR/verify_retrieval_inference.py" \
  "$SCRIPT_DIR/run_retrieval_inference.sh" \
  "$ANALYSIS_DIR/paired_stats.py" \
  "$OUTPUT_ROOT/protocol/"
exec > >(tee -a "$OUTPUT_ROOT/pipeline.log") 2>&1

echo "=== Running paired retrieval inference ==="
"$PYTHON_BIN" "$SCRIPT_DIR/retrieval_inference.py" \
  --calibration-root "$CALIBRATION_ROOT" \
  --output-root "$OUTPUT_ROOT"

echo "=== Independently reconstructing retrieval outputs ==="
"$PYTHON_BIN" "$SCRIPT_DIR/verify_retrieval_inference.py" \
  --calibration-root "$CALIBRATION_ROOT" \
  --output-root "$OUTPUT_ROOT"

echo "=== RQ3 paired retrieval inference complete and verified ==="
