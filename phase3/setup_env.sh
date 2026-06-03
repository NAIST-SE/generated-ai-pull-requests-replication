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

# 5. Verify installation
echo "Verifying installation..."
python3 -c "import transformers; import accelerate; import bitsandbytes; print(f'Transformers: {transformers.__version__}, Accelerate: {accelerate.__version__}, BitsAndBytes: {bitsandbytes.__version__}')"

echo "Setup complete. Virtual environment is active."
