#!/usr/bin/env bash
# Run all RQ2/RQ3 figure notebooks then assemble textual_features_boxplots.pdf.
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

# Frontend jupyter (any working nbconvert is fine — it only drives execution; the KERNEL
# is what needs the packages). Override with JUPYTER=... if needed.
JUPYTER="${JUPYTER:-jupyter}"

# Named kernels selected explicitly per notebook so we never depend on which "python3"
# kernelspec happens to win. Register once (see run instructions):
#   python -m ipykernel install --user --name iwadi-main          # main analysis env (torch, matplotlib, ...)
#   <py312-env>/bin/python -m ipykernel install --user --name iwadi-py312   # taaled + spaCy env
MAIN_KERNEL="${MAIN_KERNEL:-iwadi-main}"     # word/sentence, ttr, reading_ease, rouge, bm25, bertscore, codebert, graphcodebert
PY312_KERNEL="${PY312_KERNEL:-iwadi-py312}"  # mtld, lexical_density (taaled/spaCy)

# Python used only to assemble the final PDF (needs matplotlib/pypdf) — the MAIN kernel's env.
ASSEMBLE_PYTHON="${ASSEMBLE_PYTHON:-python}"

echo "=== Checking kernels & CUDA ==="
"$JUPYTER" kernelspec list 2>/dev/null | grep -E "iwadi-main|iwadi-py312" || {
  echo "   ERROR: expected kernels not registered. Register them first:"
  echo "     python -m ipykernel install --user --name ${MAIN_KERNEL}"
  echo "     /mnt/ext-hdd1/ayo/py312-analysis/bin/python -m ipykernel install --user --name ${PY312_KERNEL}"
  exit 1
}
TORCH_VER="$(timeout 15 "$ASSEMBLE_PYTHON" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
if timeout 15 "$ASSEMBLE_PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "   ✓ CUDA available — torch ${TORCH_VER} (main kernel: ${MAIN_KERNEL})"
else
  echo "   WARNING: CUDA unavailable — torch ${TORCH_VER}"
  echo "   Heavy notebooks (bertscore, codebert, graphcodebert) will run on CPU and be slow."
fi

echo "=== Starting notebook runs ==="

# Light notebooks first, heavy GPU notebooks last
LIGHT=(
  analysis_rq2_word_sentence_count.ipynb
  analysis_rq2_reading_ease.ipynb
  analysis_rq2_ttr.ipynb
  analysis_rq2_mtld.ipynb
  analysis_rq2_lexical_density.ipynb
  analysis_rq2_rouge.ipynb
  analysis_rq2_bm25.ipynb
)

# BERTScore and CodeBERT/GraphCodeBERT all load large transformer models
HEAVY=(
  analysis_rq2_bertscore.ipynb
  analysis_rq3_codebert.ipynb
  analysis_rq3_graphcodebert.ipynb
)

PASS=0
FAIL=0
FAILED_NOTEBOOKS=()

run_notebook() {
  local nb="$1"
  local timeout="$2"
  local kernel="${3:-$MAIN_KERNEL}"   # kernel name (iwadi-main / iwadi-py312)
  echo ""
  echo "▶  $nb   [kernel: $kernel]"
  if "$JUPYTER" nbconvert \
      --to notebook \
      --execute \
      --inplace \
      --ExecutePreprocessor.timeout="$timeout" \
      --ExecutePreprocessor.kernel_name="$kernel" \
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
  case "$nb" in
    analysis_rq2_mtld.ipynb|analysis_rq2_lexical_density.ipynb)
      run_notebook "$nb" 600 "$PY312_KERNEL" ;;
    *)
      run_notebook "$nb" 600 "$MAIN_KERNEL" ;;
  esac
done

echo ""
echo "=== Running heavy notebooks (timeout: 2 h each) ==="
for nb in "${HEAVY[@]}"; do
  run_notebook "$nb" 7200 "$MAIN_KERNEL"
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
"$ASSEMBLE_PYTHON" assemble_textual_features_pdf.py
