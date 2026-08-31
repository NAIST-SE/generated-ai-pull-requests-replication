#!/usr/bin/env bash
# Rebuild Preliminary-RQ and RQ1 from the finalized corrected detector corpus.

set -euo pipefail

HOST_SHORT="$(hostname -s)"
if [[ "$HOST_SHORT" != "nokogiri" ]]; then
  echo "ERROR: this pipeline is server-only; current host is '$HOST_SHORT'." >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/mnt/ext-hdd1/ayo/llm-in-prs}"
MAIN_PYTHON="${MAIN_PYTHON:-/home/ayo-b/.pyenv/versions/3.12.9/bin/python}"
INPUT="${INPUT:-$REPO_ROOT/phase3/datasets/temperature_1/run_3_corrected_detection_v1/detection_results.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/phase3/analysis/outputs/corrected_detection_v1}"

cd "$REPO_ROOT"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "ERROR: corrected output directory already exists at $OUTPUT_ROOT." >&2
  echo "Use a new OUTPUT_ROOT or review the existing result before replacing it." >&2
  exit 3
fi

mkdir -p "$OUTPUT_ROOT"

echo "=== Corrected detector input ==="
sha256sum "$INPUT"

echo "=== Preliminary Question: paired tests ==="
"$MAIN_PYTHON" phase3/analysis/deep_audit/analysis_prelim_paired.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_ROOT/prelim"

echo "=== Preliminary Question and RQ1: descriptive metrics ==="
"$MAIN_PYTHON" phase3/analysis/deep_audit/analysis_detection_descriptive.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_ROOT/detection_descriptive"

echo "=== RQ1: pooled Friedman and model-specific paired contrasts ==="
"$MAIN_PYTHON" phase3/analysis/deep_audit/analysis_rq1_cleaned.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_ROOT/rq1"

echo "=== RQ1: probability figure from calculated contrasts ==="
"$MAIN_PYTHON" phase3/analysis/deep_audit/plot_rq1_probability_boxplots.py \
  --input "$INPUT" \
  --pairwise "$OUTPUT_ROOT/rq1/rq1_model_context_pairwise.csv" \
  --output-dir "$OUTPUT_ROOT/figures"

echo "=== Independent corrected-output verification ==="
"$MAIN_PYTHON" phase3/analysis/deep_audit/verify_corrected_detection_outputs.py \
  --run-dir "$REPO_ROOT/phase3/datasets/temperature_1/run_3_corrected_detection_v1" \
  --output-root "$OUTPUT_ROOT"

echo "=== Corrected Preliminary-RQ/RQ1 pipeline PASS ==="
echo "Review $OUTPUT_ROOT/verification_report.json before changing manuscript values."
