#!/usr/bin/env bash
# Run all RQ3 figure notebooks then assemble textual_features_boxplots.pdf.
#
# Usage:
#   bash run_figure_notebooks.sh
#
# Set JUPYTER env var to override the jupyter command, e.g.:
#   JUPYTER=/path/to/venv/bin/jupyter bash run_figure_notebooks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# PyTorch's bundled CUDA runtime doesn't find libcuda.so.1 automatically on this
# system; preloading the real driver library makes torch.cuda.is_available() work.
LIBCUDA="/usr/lib/x86_64-linux-gnu/libcuda.so.1"
if [ -f "$LIBCUDA" ]; then
  export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${LIBCUDA}"
fi

# Default to the venv's jupyter if it exists and no override is set
VENV_JUPYTER="$SCRIPT_DIR/../../.venv/bin/jupyter"
if [ -z "${JUPYTER:-}" ] && [ -x "$VENV_JUPYTER" ]; then
  JUPYTER="$(realpath "$VENV_JUPYTER")"
else
  JUPYTER="${JUPYTER:-jupyter}"
fi

# Resolve jupyter to its full path so we can derive the venv bin directory
JUPYTER_PATH="$(command -v "$JUPYTER" 2>/dev/null || echo "$JUPYTER")"
VENV_BIN="$(dirname "$JUPYTER_PATH")"
VENV_PIP="${VENV_BIN}/pip"
VENV_PYTHON="${VENV_BIN}/python3"
[ -x "$VENV_PIP" ]    || VENV_PIP="pip"
[ -x "$VENV_PYTHON" ] || VENV_PYTHON="python3"

echo "=== Checking PyTorch CUDA ==="
TORCH_VER="$(timeout 15 "$VENV_PYTHON" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
if timeout 15 "$VENV_PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "   ✓ CUDA available — torch ${TORCH_VER}"
else
  echo "   WARNING: CUDA unavailable — torch ${TORCH_VER}"
  echo "   Heavy notebooks (bertscore, codebert, graphcodebert) will run on CPU and be slow."
fi

echo "=== Starting notebook runs ==="

# Light notebooks first, heavy GPU notebooks last
LIGHT=(
  analysis_rq3_word_sentence_count.ipynb
  analysis_rq3_reading_ease.ipynb
  analysis_rq3_ttr.ipynb
  analysis_rq3_mtld.ipynb
  analysis_rq3_lexical_density.ipynb
  analysis_rq3_rouge.ipynb
  analysis_rq3_bm25.ipynb
)

# BERTScore and CodeBERT/GraphCodeBERT all load large transformer models
HEAVY=(
  analysis_rq3_bertscore.ipynb
  analysis_rq3_codebert.ipynb
  analysis_rq3_graphcodebert.ipynb
)

PASS=0
FAIL=0
FAILED_NOTEBOOKS=()

run_notebook() {
  local nb="$1"
  local timeout="$2"
  echo ""
  echo "▶  $nb"
  if "$JUPYTER" nbconvert \
      --to notebook \
      --execute \
      --inplace \
      --ExecutePreprocessor.timeout="$timeout" \
      --ExecutePreprocessor.kernel_name=python3 \
      "$nb"; then
    echo "   ✓ done"
    ((PASS++)) || true
  else
    echo "   ✗ FAILED"
    FAILED_NOTEBOOKS+=("$nb")
    ((FAIL++)) || true
  fi
}

echo "=== Running light notebooks (timeout: 10 min each) ==="
for nb in "${LIGHT[@]}"; do
  run_notebook "$nb" 600
done

echo ""
echo "=== Running heavy notebooks (timeout: 2 h each) ==="
for nb in "${HEAVY[@]}"; do
  run_notebook "$nb" 7200
done

echo ""
echo "=== Summary: $PASS passed, $FAIL failed ==="
if [ ${#FAILED_NOTEBOOKS[@]} -gt 0 ]; then
  echo "Failed:"
  for nb in "${FAILED_NOTEBOOKS[@]}"; do
    echo "  - $nb"
  done
  echo ""
  echo "Fix the failed notebooks before assembling the PDF."
  exit 1
fi

echo ""
echo "=== Assembling textual_features_boxplots.pdf ==="
"$VENV_PYTHON" assemble_textual_features_pdf.py
