#!/usr/bin/env bash
set -euo pipefail

study_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$study_repo_root"

approved_server_python="/home/ayo-b/.pyenv/versions/3.12.9/bin/python"
if [[ -n "${DETECTION_PYTHON:-}" ]]; then
  detection_python="$DETECTION_PYTHON"
elif [[ -x "$approved_server_python" ]]; then
  detection_python="$approved_server_python"
else
  detection_python="python3"
fi
action="${1:-}"
tokens_file="${2:-}"

if [[ "$action" == "preflight" || "$action" == "run-local" ]]; then
  export HF_HUB_OFFLINE=1
  export HF_HUB_DISABLE_PROGRESS_BARS=1
  server_libcuda="/usr/lib/x86_64-linux-gnu/libcuda.so.1"
  if [[ -f "$server_libcuda" ]]; then
    export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${server_libcuda}"
  fi
fi

case "$action" in
  build)
    "$detection_python" -m phase3.detection.corrected_rerun.build_corpus
    ;;
  preflight)
    if [[ -z "$tokens_file" ]]; then
      echo "Usage: $0 preflight /absolute/path/to/zerogpt-tokens.txt" >&2
      exit 64
    fi
    "$detection_python" -m phase3.detection.corrected_rerun.run_detector \
      --detector roberta --preflight-only
    "$detection_python" -m phase3.detection.corrected_rerun.run_detector \
      --detector roberta-qa --preflight-only
    "$detection_python" -m phase3.detection.corrected_rerun.run_detector \
      --detector zerogpt --preflight-only --tokens-file "$tokens_file"
    ;;
  run-local)
    "$detection_python" -m phase3.detection.corrected_rerun.run_detector \
      --detector roberta --device cuda
    "$detection_python" -m phase3.detection.corrected_rerun.run_detector \
      --detector roberta-qa --device cuda
    ;;
  run-zerogpt)
    if [[ -z "$tokens_file" ]]; then
      echo "Usage: $0 run-zerogpt /absolute/path/to/zerogpt-tokens.txt" >&2
      exit 64
    fi
    "$detection_python" -m phase3.detection.corrected_rerun.run_detector \
      --detector zerogpt --tokens-file "$tokens_file"
    ;;
  finalize)
    "$detection_python" -m phase3.detection.corrected_rerun.finalize finalize
    "$detection_python" -m phase3.detection.corrected_rerun.finalize verify
    ;;
  verify)
    "$detection_python" -m phase3.detection.corrected_rerun.finalize verify
    ;;
  *)
    echo "Usage: $0 {build|preflight|run-local|run-zerogpt|finalize|verify} [tokens-file]" >&2
    exit 64
    ;;
esac
