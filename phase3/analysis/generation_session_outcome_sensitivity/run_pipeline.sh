#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname -s)" != "nokogiri" ]]; then
  echo "Refusing to run outside Nokogiri" >&2
  exit 2
fi

repo_root="/mnt/ext-hdd1/ayo/llm-in-prs"
script_dir="$repo_root/phase3/analysis/generation_session_outcome_sensitivity"
main_python="/home/ayo-b/.pyenv/versions/3.12.9/bin/python"
py312_python="/mnt/ext-hdd1/ayo/py312-analysis/bin/python"
cuda_library="/usr/lib/x86_64-linux-gnu/libcuda.so.1"

cd "$repo_root"
export PYTHONPATH="$script_dir:$repo_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[1/7] Model-free tests"
"$main_python" -m unittest discover -s "$script_dir" -p 'test_*.py' -v

echo "[2/7] Main RQ2 metrics and historical validation"
"$main_python" "$script_dir/score_main.py"

echo "[3/7] TAALED/spaCy metrics and historical validation"
"$py312_python" "$script_dir/score_py312.py"

echo "[4/7] Pinned BERTScore and historical validation"
LD_PRELOAD="$cuda_library" "$main_python" "$script_dir/score_bertscore.py"

echo "[5/7] Pinned local RoBERTa detectors"
LD_PRELOAD="$cuda_library" "$main_python" "$script_dir/score_detectors.py"

echo "[6/7] Paired sensitivity analysis"
"$main_python" "$script_dir/analyze.py"

echo "[7/7] Final integrity verification"
"$main_python" "$script_dir/verify.py"

echo "Outcome-sensitivity pipeline completed successfully"
