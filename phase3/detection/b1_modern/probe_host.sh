#!/usr/bin/env bash
# Read-only lab host probe. It intentionally prints no environment variables or credentials.

set -euo pipefail

probe_path="${B1_RUNTIME_ROOT:-.}"
probe_status=0

echo "B1 lab host probe"
uname -a
echo

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi \
    --query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap \
    --format=csv; then
    nvidia-smi \
      --query-gpu=index,name,memory.total,memory.free,driver_version \
      --format=csv
  fi
else
  echo "ERROR: nvidia-smi is unavailable."
  probe_status=1
fi
echo

if command -v free >/dev/null 2>&1; then
  free -h
fi
df -h "$probe_path"
echo

python3 --version
git --version

if ! python3 -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)'; then
  echo "ERROR: B1 requires Python 3.10--3.12."
  probe_status=1
fi

exit "$probe_status"
