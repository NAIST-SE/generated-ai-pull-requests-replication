#!/usr/bin/env bash
set -u

repo_root="/mnt/ext-hdd1/ayo/llm-in-prs"
analysis_dir="$repo_root/phase3/analysis/generation_session_six_context_bridge"
output_dir="$repo_root/phase3/analysis/outputs/tier_b/generation-session-six-context-bridge-v1"

mkdir -p "$output_dir"
set +e
bash "$analysis_dir/run.sh" >"$output_dir/run.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" >"$output_dir/run.exit"
exit "$status"
