#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname -s)" != "nokogiri" ]]; then
  echo "Refusing to run outside Nokogiri" >&2
  exit 2
fi

repo_root="/mnt/ext-hdd1/ayo/llm-in-prs"
script_dir="$repo_root/phase3/analysis/generation_session_outcome_sensitivity"
output_dir="$repo_root/phase3/analysis/outputs/tier_b/generation-session-outcome-sensitivity-v1"
main_python="/home/ayo-b/.pyenv/versions/3.12.9/bin/python"
cuda_library="/usr/lib/x86_64-linux-gnu/libcuda.so.1"

test -f "$output_dir/main_metrics_metadata.json"
test -f "$output_dir/py312_metrics_metadata.json"

cd "$repo_root"
export PYTHONPATH="$script_dir:$repo_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[resume 1/5] Model-free tests"
"$main_python" -m unittest discover -s "$script_dir" -p 'test_*.py' -v

echo "[resume 2/5] Pinned BERTScore and historical validation"
LD_PRELOAD="$cuda_library" "$main_python" "$script_dir/score_bertscore.py"

echo "[resume 3/5] Pinned local RoBERTa detectors"
LD_PRELOAD="$cuda_library" "$main_python" "$script_dir/score_detectors.py"

echo "[resume 4/5] Paired sensitivity analysis"
"$main_python" "$script_dir/analyze.py"

echo "[resume 5/5] Final integrity verification"
"$main_python" "$script_dir/verify.py"

echo "Outcome-sensitivity resume completed successfully"
