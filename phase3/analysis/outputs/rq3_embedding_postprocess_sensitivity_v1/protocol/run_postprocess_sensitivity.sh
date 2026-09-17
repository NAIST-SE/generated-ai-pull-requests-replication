#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CALIBRATION_ROOT="$ANALYSIS_DIR/outputs/rq3_mismatched_calibration_v1"
OUTPUT_ROOT="$ANALYSIS_DIR/outputs/rq3_embedding_postprocess_sensitivity_v1"
PYTHON_BIN="${RQ3_CALIBRATION_PYTHON:-/home/ayo-b/.pyenv/versions/3.12.9/bin/python}"

if [ "$(hostname -s)" != "nokogiri" ]; then
  echo "ERROR: the embedding sensitivity must run on nokogiri." >&2
  exit 1
fi

echo "=== RQ3 embedding post-processing sensitivity preflight ==="
echo "Host: $(hostname -s)"
echo "Analysis directory: $ANALYSIS_DIR"
"$PYTHON_BIN" -c 'import numpy,pandas,scipy,statsmodels; print("✓ Required numerical packages import")'
"$PYTHON_BIN" "$SCRIPT_DIR/postprocess_sensitivity.py" \
  --preflight \
  --calibration-root "$CALIBRATION_ROOT"

if [ "${1:-}" = "--preflight" ]; then
  echo "=== Preflight complete; no sensitivity metric was computed ==="
  exit 0
elif [ -n "${1:-}" ]; then
  echo "Usage: bash rq3_mismatched_calibration/run_postprocess_sensitivity.sh [--preflight]" >&2
  exit 2
fi

if [ -e "$OUTPUT_ROOT" ]; then
  echo "ERROR: output namespace already exists; refusing to overwrite: $OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT/protocol"
cp \
  "$SCRIPT_DIR/POSTPROCESSING_PROTOCOL.md" \
  "$SCRIPT_DIR/analysis.py" \
  "$SCRIPT_DIR/verify_outputs.py" \
  "$SCRIPT_DIR/postprocess_sensitivity.py" \
  "$SCRIPT_DIR/verify_postprocess.py" \
  "$SCRIPT_DIR/run_postprocess_sensitivity.sh" \
  "$OUTPUT_ROOT/protocol/"
exec > >(tee -a "$OUTPUT_ROOT/pipeline.log") 2>&1

echo "=== Running frozen post-processing variants ==="
"$PYTHON_BIN" "$SCRIPT_DIR/postprocess_sensitivity.py" \
  --calibration-root "$CALIBRATION_ROOT" \
  --output-root "$OUTPUT_ROOT"

echo "=== Independently reconstructing sensitivity outputs ==="
"$PYTHON_BIN" "$SCRIPT_DIR/verify_postprocess.py" \
  --calibration-root "$CALIBRATION_ROOT" \
  --output-root "$OUTPUT_ROOT"

echo "=== RQ3 embedding sensitivity complete and verified ==="
