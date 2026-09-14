#!/usr/bin/env bash
# Portable command dispatcher for B1. Detector jobs are deliberately sequential.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
B1_RUNTIME_ROOT="${B1_RUNTIME_ROOT:-}"
B1_OUTPUT_ROOT="${B1_OUTPUT_ROOT:-$REPO_ROOT/phase3/analysis/outputs/tier_b/b1-modern-detectors-v1}"

command_name="${1:-help}"
shift || true

if [[ "$command_name" == "help" ]]; then
  echo "Usage: bash $SCRIPT_DIR/run_b1.sh COMMAND"
  echo "Commands: probe, setup, models, prepare, preflight, smoke DETECTOR REPEAT, verify-smoke DETECTOR,"
  echo "          reference-parity DETECTOR, approve DETECTOR ..., full DETECTOR, import-binoculars,"
  echo "          verify-full, analyse, verify-analysis, test"
  exit 0
fi

if [[ "$command_name" == "probe" ]]; then
  exec bash "$SCRIPT_DIR/probe_host.sh" "$@"
fi

if [[ "$command_name" == "setup" ]]; then
  if [[ -z "$B1_RUNTIME_ROOT" ]]; then
    echo "ERROR: set B1_RUNTIME_ROOT to the absolute lab runtime directory." >&2
    exit 2
  fi
  exec bash "$SCRIPT_DIR/setup_runtime.sh" "$@"
fi

case "$command_name" in
  prepare|import-binoculars|verify-full|analyse|verify-analysis|test)
    B1_PYTHON="${B1_PYTHON:-$(command -v python3 || true)}"
    ;;
  *)
    if [[ -z "$B1_RUNTIME_ROOT" ]]; then
      echo "ERROR: set B1_RUNTIME_ROOT to the absolute lab runtime directory." >&2
      exit 2
    fi
    B1_PYTHON="${B1_PYTHON:-$B1_RUNTIME_ROOT/venv/bin/python}"
    ;;
esac

if [[ ! -x "$B1_PYTHON" ]]; then
  echo "ERROR: B1 Python is unavailable at $B1_PYTHON." >&2
  exit 2
fi

export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
if [[ -n "$B1_RUNTIME_ROOT" ]]; then
  export HF_HOME="$B1_RUNTIME_ROOT/hf-cache"
  export MPLCONFIGDIR="$B1_RUNTIME_ROOT/matplotlib-cache"
  mkdir -p "$MPLCONFIGDIR"
fi
cd "$REPO_ROOT"

case "$command_name" in
  models)
    exec "$B1_PYTHON" -m phase3.detection.b1_modern.download_models \
      --runtime-root "$B1_RUNTIME_ROOT" "$@"
    ;;
  prepare)
    exec "$B1_PYTHON" -m phase3.detection.b1_modern.prepare_manifest \
      --repo-root "$REPO_ROOT" --output-root "$B1_OUTPUT_ROOT" "$@"
    ;;
  preflight)
    exec "$B1_PYTHON" -m phase3.detection.b1_modern.preflight \
      --repo-root "$REPO_ROOT" --runtime-root "$B1_RUNTIME_ROOT" \
      --output-root "$B1_OUTPUT_ROOT" "$@"
    ;;
  smoke)
    detector="${1:?usage: run_b1.sh smoke DETECTOR REPEAT}"
    repeat="${2:?usage: run_b1.sh smoke DETECTOR REPEAT}"
    exec "$B1_PYTHON" -m phase3.detection.b1_modern.run_detector \
      --detector "$detector" --mode smoke --repeat "$repeat" \
      --repo-root "$REPO_ROOT" --runtime-root "$B1_RUNTIME_ROOT" \
      --output-root "$B1_OUTPUT_ROOT" "${@:3}"
    ;;
  verify-smoke)
    detector="${1:?usage: run_b1.sh verify-smoke DETECTOR}"
    exec "$B1_PYTHON" -m phase3.detection.b1_modern.verify_smoke \
      --detector "$detector" --repo-root "$REPO_ROOT" \
      --output-root "$B1_OUTPUT_ROOT" "${@:2}"
    ;;
  reference-parity)
    detector="${1:?usage: run_b1.sh reference-parity DETECTOR}"
    exec "$B1_PYTHON" -m phase3.detection.b1_modern.verify_reference_parity \
      --detector "$detector" --repo-root "$REPO_ROOT" \
      --runtime-root "$B1_RUNTIME_ROOT" --output-root "$B1_OUTPUT_ROOT" "${@:2}"
    ;;
  approve)
    detector="${1:?usage: run_b1.sh approve DETECTOR [approval arguments]}"
    exec "$B1_PYTHON" -m phase3.detection.b1_modern.approve_full_run \
      --detector "$detector" --repo-root "$REPO_ROOT" \
      --output-root "$B1_OUTPUT_ROOT" "${@:2}"
    ;;
  full)
    detector="${1:?usage: run_b1.sh full DETECTOR}"
    exec "$B1_PYTHON" -m phase3.detection.b1_modern.run_detector \
      --detector "$detector" --mode full --confirm-full-run \
      --repo-root "$REPO_ROOT" --runtime-root "$B1_RUNTIME_ROOT" \
      --output-root "$B1_OUTPUT_ROOT" "${@:2}"
    ;;
  import-binoculars)
    exec "$B1_PYTHON" -m phase3.detection.b1_modern.import_external_binoculars \
      --output-root "$B1_OUTPUT_ROOT" "$@"
    ;;
  verify-full)
    exec "$B1_PYTHON" -m phase3.detection.b1_modern.verify_full \
      --repo-root "$REPO_ROOT" --output-root "$B1_OUTPUT_ROOT" "$@"
    ;;
  analyse)
    exec "$B1_PYTHON" -m phase3.detection.b1_modern.analyse \
      --repo-root "$REPO_ROOT" --output-root "$B1_OUTPUT_ROOT" "$@"
    ;;
  verify-analysis)
    exec "$B1_PYTHON" -m phase3.detection.b1_modern.verify_analysis \
      --repo-root "$REPO_ROOT" --output-root "$B1_OUTPUT_ROOT" "$@"
    ;;
  test)
    exec "$B1_PYTHON" -m pytest -q "$SCRIPT_DIR/test_b1_modern.py" "$@"
    ;;
  *)
    echo "Usage: bash $SCRIPT_DIR/run_b1.sh COMMAND"
    echo "Commands: probe, setup, models, prepare, preflight, smoke DETECTOR REPEAT, verify-smoke DETECTOR,"
    echo "          reference-parity DETECTOR, approve DETECTOR ..., full DETECTOR, import-binoculars,"
    echo "          verify-full, analyse, verify-analysis, test"
    ;;
esac
