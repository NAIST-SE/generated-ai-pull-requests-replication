#!/usr/bin/env bash
# Execute the deep-audit remediation pipeline on nokogiri only.

set -euo pipefail

HOST_SHORT="$(hostname -s)"
if [[ "$HOST_SHORT" != "nokogiri" ]]; then
  echo "ERROR: this pipeline is server-only; current host is '$HOST_SHORT'." >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/mnt/ext-hdd1/ayo/llm-in-prs}"
MAIN_PYTHON="${MAIN_PYTHON:-/home/ayo-b/.pyenv/versions/3.12.9/bin/python}"
JUPYTER="${JUPYTER:-jupyter}"
MAIN_KERNEL="${MAIN_KERNEL:-iwadi-main}"
PY312_KERNEL="${PY312_KERNEL:-iwadi-py312}"
CLEAN_DIR="$REPO_ROOT/phase3/datasets/temperature_1/run_3_cleaned"

cd "$REPO_ROOT"

if [[ -e "$CLEAN_DIR" ]]; then
  echo "ERROR: cleaned derivative already exists at $CLEAN_DIR." >&2
  echo "Review it or rerun prepare_cleaned_run3.py explicitly with --overwrite." >&2
  exit 3
fi

echo "=== Preparing immutable cleaned run-3 derivative ==="
"$MAIN_PYTHON" phase3/analysis/deep_audit/prepare_cleaned_run3.py

echo "=== Running paired Preliminary Question analysis ==="
"$MAIN_PYTHON" phase3/analysis/deep_audit/analysis_prelim_paired.py

echo "=== Regenerating cleaned descriptive detector metrics ==="
"$MAIN_PYTHON" phase3/analysis/deep_audit/analysis_detection_descriptive.py

echo "=== Running cleaned RQ1 analysis ==="
"$MAIN_PYTHON" phase3/analysis/deep_audit/analysis_rq1_cleaned.py

echo "=== Executing RQ2/RQ3 notebooks with mandated kernels ==="
JUPYTER="$JUPYTER" \
MAIN_KERNEL="$MAIN_KERNEL" \
PY312_KERNEL="$PY312_KERNEL" \
ASSEMBLE_PYTHON="$MAIN_PYTHON" \
bash phase3/analysis/run_figure_notebooks.sh

echo "=== Deep-audit server pipeline completed ==="
echo "Review phase3/datasets/temperature_1/run_3_cleaned/integrity_summary.json"
echo "Review phase3/analysis/outputs/deep_audit/ before changing manuscript numbers"
