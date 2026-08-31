#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; printf "VISUAL_REFRESH_EXIT=%s\n" "$rc"' EXIT

if [[ "$(hostname -s)" != "nokogiri" ]]; then
  echo "ERROR: visual refresh must run on nokogiri" >&2
  exit 2
fi

analysis_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$analysis_dir/../.." && pwd)"
analysis_python="${ANALYSIS_PYTHON:-/home/ayo-b/.pyenv/versions/3.12.9/bin/python}"
cuda_preload="${CUDA_PRELOAD:-/usr/lib/x86_64-linux-gnu/libcuda.so.1}"

cd "$analysis_dir"

run_notebook() {
  local kernel="$1"
  local notebook="$2"
  echo "RUN notebook=$notebook kernel=$kernel"
  LD_PRELOAD="$cuda_preload" "$analysis_python" -m jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.kernel_name="$kernel" \
    --ExecutePreprocessor.timeout=-1 \
    "$notebook"
  echo "PASS notebook=$notebook"
}

run_notebook iwadi-main rq1_probability_boxplots.ipynb
run_notebook iwadi-py312 analysis_rq2_lexical_density.ipynb
run_notebook iwadi-main analysis_rq3_codebert.ipynb
run_notebook iwadi-main analysis_rq3_graphcodebert.ipynb

echo "RUN composite assembly"
"$analysis_python" assemble_textual_features_pdf.py
echo "PASS composite assembly"

echo "RUN output verification"
"$analysis_python" deep_audit/verify_outputs.py \
  --clean-dir "$repo_root/phase3/datasets/temperature_1/run_3_cleaned" \
  --analysis-dir "$analysis_dir"
echo "PASS output verification"
