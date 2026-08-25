#!/bin/bash

# setup_env.sh
# Sets up a clean virtual environment for Llama 3.1 generation on the server.
# Usage: source setup_env.sh

echo "Setting up virtual environment..."

# 1. Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created .venv"
else
    echo ".venv already exists"
fi

# 2. Activate venv
source .venv/bin/activate

# 3. Upgrade pip (crucial for modern wheels)
pip install --upgrade pip

# 4. Install dependencies
# Using --no-cache-dir to ensure fresh pulls and avoid corruption
echo "Installing dependencies from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

# 4b. MTLD dependency (taaled) — analysis only, Python 3.12 ONLY.
# taaled's build breaks under modern setuptools and pip build-isolation ignores a
# pin in requirements.txt, so it must be installed separately, against an older
# setuptools and with --no-build-isolation. Skipped automatically outside 3.12.
setup_taaled() {
    PYVER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    if [ "$PYVER" != "3.12" ]; then
        echo "Skipping taaled: needs Python 3.12 (found $PYVER). Run the mtld notebook under a 3.12 env."
        return 0
    fi
    echo "Installing taaled (MTLD) against setuptools==69.5.1, no build isolation..."
    pip install "setuptools==69.5.1" wheel
    pip install --no-build-isolation taaled==0.32 pylats==0.64
}
setup_taaled

# 5. Verify installation
echo "Verifying installation..."
python3 -c "import transformers; import accelerate; import bitsandbytes; print(f'Transformers: {transformers.__version__}, Accelerate: {accelerate.__version__}, BitsAndBytes: {bitsandbytes.__version__}')"

echo "Setup complete. Virtual environment is active."
