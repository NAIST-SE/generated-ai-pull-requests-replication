#!/usr/bin/env bash
set -euo pipefail

repo_root="/mnt/ext-hdd1/ayo/llm-in-prs"
analysis_dir="$repo_root/phase3/analysis/generation_session_six_context_bridge"
python_main="/home/ayo-b/.pyenv/versions/3.12.9/bin/python"

if [[ "$(hostname)" != *nokogiri* ]]; then
  echo "Refusing to run outside Nokogiri" >&2
  exit 2
fi

cd "$analysis_dir"
"$python_main" -m unittest -v test_bridge.py
"$python_main" analyze.py
"$python_main" verify.py
