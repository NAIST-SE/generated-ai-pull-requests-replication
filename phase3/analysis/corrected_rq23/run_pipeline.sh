#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ANALYSIS_DIR/../.." && pwd)"

CORPUS="$REPO_ROOT/phase3/datasets/temperature_1/run_3_corrected_detection_v1/corpus.csv"
DIFFS="$REPO_ROOT/phase3/datasets/selected_prs_with_diffs_and_file_contents.csv"
OUTPUT_ROOT="$ANALYSIS_DIR/outputs/corrected_rq23_v1"
NOTEBOOK_DIR="$OUTPUT_ROOT/notebooks"
ARTIFACT_DIR="$NOTEBOOK_DIR/outputs"

CORPUS_SHA256="07136384785d6d02d17762535ab51a2007a7a47b8c41b77fbb311ce9c8780306"
DIFFS_SHA256="45c71d0170fc9343d999998ad6515e9b5b8298f63b66215f4110b3911f681d0c"

MAIN_PYTHON="${RQ23_MAIN_PYTHON:-/home/ayo-b/.pyenv/versions/3.12.9/bin/python}"
JUPYTER_BIN="${RQ23_JUPYTER:-jupyter}"
MAIN_KERNEL="${RQ23_MAIN_KERNEL:-iwadi-main}"
PY312_KERNEL="${RQ23_PY312_KERNEL:-iwadi-py312}"

export RQ23_DEBERTA_REVISION="5b07a9086c1dbb79981ff7b05b4d1ad83b3af51c"
export RQ23_CODEBERT_REVISION="3b0952feddeffad0063f274080e3c23d75e7eb39"
export RQ23_GRAPHCODEBERT_REVISION="2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d"
export RQ23_DEBERTA_SNAPSHOT="/home/ayo-b/.cache/huggingface/hub/models--microsoft--deberta-xlarge-mnli/snapshots/$RQ23_DEBERTA_REVISION"
export RQ23_BERTSCORE_BASELINE="/home/ayo-b/.pyenv/versions/3.12.9/lib/python3.12/site-packages/bert_score/rescale_baseline/en/microsoft/deberta-xlarge-mnli.tsv"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LIBCUDA="/usr/lib/x86_64-linux-gnu/libcuda.so.1"
if [ -f "$LIBCUDA" ]; then
  export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}${LIBCUDA}"
fi

MODE="run"
if [ "${1:-}" = "--preflight" ]; then
  MODE="preflight"
elif [ -n "${1:-}" ]; then
  echo "Usage: bash corrected_rq23/run_pipeline.sh [--preflight]" >&2
  exit 2
fi

if [ "$(hostname -s)" != "nokogiri" ]; then
  echo "ERROR: corrected RQ2/RQ3 analyses must run on nokogiri." >&2
  exit 1
fi

check_sha() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "$path" | awk '{print $1}')"
  if [ "$observed" != "$expected" ]; then
    echo "ERROR: SHA-256 mismatch for $path" >&2
    echo "  expected: $expected" >&2
    echo "  observed: $observed" >&2
    exit 1
  fi
}

echo "=== Corrected RQ2/RQ3 preflight ==="
echo "Host: $(hostname -s)"
echo "Repository: $REPO_ROOT"
echo "Commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
check_sha "$CORPUS" "$CORPUS_SHA256"
check_sha "$DIFFS" "$DIFFS_SHA256"
echo "✓ Frozen corpus and diff hashes match"

"$JUPYTER_BIN" kernelspec list 2>/dev/null | grep -q "$MAIN_KERNEL"
"$JUPYTER_BIN" kernelspec list 2>/dev/null | grep -q "$PY312_KERNEL"
echo "✓ Kernels registered: $MAIN_KERNEL, $PY312_KERNEL"

if ! "$MAIN_PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "ERROR: CUDA is unavailable in $MAIN_PYTHON" >&2
  exit 1
fi
"$MAIN_PYTHON" -c 'import torch; print(f"✓ CUDA: {torch.__version__} / {torch.cuda.get_device_name(0)}")'

for required_path in \
  "$RQ23_DEBERTA_SNAPSHOT/config.json" \
  "$RQ23_BERTSCORE_BASELINE" \
  "/home/ayo-b/.cache/huggingface/hub/models--microsoft--codebert-base/snapshots/$RQ23_CODEBERT_REVISION/config.json" \
  "/home/ayo-b/.cache/huggingface/hub/models--microsoft--graphcodebert-base/snapshots/$RQ23_GRAPHCODEBERT_REVISION/config.json"; do
  if [ ! -f "$required_path" ]; then
    echo "ERROR: missing pinned offline model artifact: $required_path" >&2
    exit 1
  fi
done
echo "✓ Pinned transformer snapshots and BERTScore baseline are present"

"$MAIN_PYTHON" "$SCRIPT_DIR/verify_outputs.py" preflight \
  --corpus "$CORPUS" \
  --diffs "$DIFFS"

if [ "$MODE" = "preflight" ]; then
  PREFLIGHT_DIR="$(mktemp -d /tmp/corrected-rq23-preflight.XXXXXX)"
  "$MAIN_PYTHON" "$SCRIPT_DIR/prepare_notebooks.py" \
    --analysis-dir "$ANALYSIS_DIR" \
    --output-dir "$PREFLIGHT_DIR" \
    --corpus "$CORPUS" \
    --diffs "$DIFFS"
  echo "✓ Prepared notebook copies validated in $PREFLIGHT_DIR"
  echo "=== Preflight complete; no study metric was computed ==="
  exit 0
fi

if [ -f "$OUTPUT_ROOT/completion.json" ]; then
  echo "ERROR: completed output namespace already exists: $OUTPUT_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/pipeline.log") 2>&1

echo "=== Preparing isolated notebook copies ==="
"$MAIN_PYTHON" "$SCRIPT_DIR/prepare_notebooks.py" \
  --analysis-dir "$ANALYSIS_DIR" \
  --output-dir "$NOTEBOOK_DIR" \
  --corpus "$CORPUS" \
  --diffs "$DIFFS"

run_notebook() {
  local notebook="$1"
  local timeout_seconds="$2"
  local kernel="$3"
  echo
  echo "▶ $notebook [kernel=$kernel, timeout=$timeout_seconds]"
  (
    cd "$NOTEBOOK_DIR"
    "$JUPYTER_BIN" nbconvert \
      --to notebook \
      --execute \
      --inplace \
      --ExecutePreprocessor.timeout="$timeout_seconds" \
      --ExecutePreprocessor.kernel_name="$kernel" \
      "$notebook"
  )
  echo "✓ $notebook"
}

echo "=== Running light RQ2 notebooks ==="
run_notebook analysis_rq2_word_sentence_count.ipynb 900 "$MAIN_KERNEL"
run_notebook analysis_rq2_reading_ease.ipynb 900 "$MAIN_KERNEL"
run_notebook analysis_rq2_ttr.ipynb 900 "$MAIN_KERNEL"
run_notebook analysis_rq2_mtld.ipynb 1200 "$PY312_KERNEL"
run_notebook analysis_rq2_lexical_density.ipynb 1800 "$PY312_KERNEL"
run_notebook analysis_rq2_rouge.ipynb 1200 "$MAIN_KERNEL"

echo "=== Running redesigned fixed-corpus BM25 analysis ==="
"$MAIN_PYTHON" "$ANALYSIS_DIR/deep_audit/analysis_rq2_bm25_retrieval.py" \
  --input "$CORPUS" \
  --output-dir "$ARTIFACT_DIR/bm25_retrieval" \
  --figure-output "$ARTIFACT_DIR/figures_pdf/bm25_retrieval.pdf"

echo "=== Running transformer-based analyses ==="
run_notebook analysis_rq2_bertscore.ipynb 7200 "$MAIN_KERNEL"
run_notebook analysis_rq3_codebert.ipynb 7200 "$MAIN_KERNEL"
run_notebook analysis_rq3_graphcodebert.ipynb 7200 "$MAIN_KERNEL"

echo "=== Assembling corrected textual-feature figure PDF ==="
(cd "$NOTEBOOK_DIR" && "$MAIN_PYTHON" assemble_textual_features_pdf.py)

echo "=== Independently reconstructing output statistics ==="
"$MAIN_PYTHON" "$SCRIPT_DIR/verify_outputs.py" verify \
  --corpus "$CORPUS" \
  --diffs "$DIFFS" \
  --output-root "$OUTPUT_ROOT" \
  --artifacts "$ARTIFACT_DIR"

echo "=== Corrected RQ2/RQ3 run complete and verified ==="
