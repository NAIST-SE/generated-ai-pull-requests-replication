#!/usr/bin/env bash
# Create an isolated B1 environment and pin the two official repositories.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
B1_RUNTIME_ROOT="${B1_RUNTIME_ROOT:-}"
B1_BASE_PYTHON="${B1_BASE_PYTHON:-python3}"
B1_MIN_FREE_GIB="${B1_MIN_FREE_GIB:-100}"

if [[ -z "$B1_RUNTIME_ROOT" ]]; then
  echo "ERROR: set B1_RUNTIME_ROOT to an absolute lab path with at least 100 GB free." >&2
  exit 2
fi
if [[ "$B1_RUNTIME_ROOT" != /* || "$B1_RUNTIME_ROOT" == "/" ]]; then
  echo "ERROR: B1_RUNTIME_ROOT must be a specific absolute directory, not /." >&2
  exit 2
fi

mkdir -p "$B1_RUNTIME_ROOT/external" "$B1_RUNTIME_ROOT/hf-cache" "$B1_RUNTIME_ROOT/environment"

"$B1_BASE_PYTHON" -c '
import pathlib, shutil, sys
root = pathlib.Path(sys.argv[1])
minimum = int(sys.argv[2]) * 1024**3
free = shutil.disk_usage(root).free
if free < minimum:
    raise SystemExit(f"B1 runtime has only {free / 1024**3:.1f} GiB free; require {int(sys.argv[2])} GiB")
' "$B1_RUNTIME_ROOT" "$B1_MIN_FREE_GIB"

"$B1_BASE_PYTHON" -c '
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit(f"B1 requires Python 3.10--3.12; found {sys.version.split()[0]}")
'

if [[ ! -x "$B1_RUNTIME_ROOT/venv/bin/python" ]]; then
  "$B1_BASE_PYTHON" -m venv "$B1_RUNTIME_ROOT/venv"
fi
B1_PYTHON="$B1_RUNTIME_ROOT/venv/bin/python"
"$B1_PYTHON" -m pip install --upgrade pip
"$B1_PYTHON" -m pip install --requirement "$SCRIPT_DIR/requirements.txt"

clone_at_commit() {
  local repository_url="$1"
  local expected_commit="$2"
  local destination="$3"
  if [[ ! -d "$destination/.git" ]]; then
    git clone --no-checkout "$repository_url" "$destination"
  fi
  git -C "$destination" fetch --depth 1 origin "$expected_commit"
  git -C "$destination" checkout --detach "$expected_commit"
  local actual_commit
  actual_commit="$(git -C "$destination" rev-parse HEAD)"
  if [[ "$actual_commit" != "$expected_commit" ]]; then
    echo "ERROR: checkout mismatch for $destination" >&2
    exit 3
  fi
  if [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
    echo "ERROR: official checkout has local modifications: $destination" >&2
    exit 3
  fi
}

clone_at_commit \
  "https://github.com/ahans30/Binoculars.git" \
  "c8ae2f90d50ee696418bc71d8d9e5020e5f9d7b8" \
  "$B1_RUNTIME_ROOT/external/Binoculars"
clone_at_commit \
  "https://github.com/baoguangsheng/fast-detect-gpt.git" \
  "971b05202bac2bb504d60c0ac0812fea7a8f7c82" \
  "$B1_RUNTIME_ROOT/external/fast-detect-gpt"

"$B1_PYTHON" -m pip freeze > "$B1_RUNTIME_ROOT/environment/pip-freeze.txt"
cd "$REPO_ROOT"
"$B1_PYTHON" -m pytest -q "$SCRIPT_DIR/test_b1_modern.py"

echo "B1 runtime prepared at $B1_RUNTIME_ROOT"
echo "Next: accept the Meta Llama 3 licence, export HF_TOKEN without writing it to disk, then run:"
echo "  bash $SCRIPT_DIR/run_b1.sh models"
