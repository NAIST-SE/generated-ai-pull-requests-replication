**HPC GPU Containers Guide (v3)**

# 🚀 Running GPU Jobs with Slurm + Singularity (Setup-First Guide)

This guide shows how to:

1. **Build a container environment**
2. **Run it on GPUs with Slurm**
3. **Handle Hugging Face + caches correctly**

Works for:

- PyTorch / TensorFlow / JAX
- CUDA programs
- Hugging Face LLMs
- Any containerized workload

---

## 🧠 Mental model (read once)

- **Slurm** = gives you GPUs + time
- **Singularity** = runs software environment
- **Container (.sif)** = frozen Linux + Python + libraries
- **Your script** = workload
- **Cache dirs** = must be writable

---

## 1️⃣ Check available modules

```bash
module avail
```

You should see:

- `singularity` or `apptainer`
- `cuda`
- `slurm`

---

## 2️⃣ Write a container definition file (llm.def)

This defines your software environment.

Example: `llm.def`

```bash
Bootstrap: docker
From: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

%post
    apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

    pip install --no-cache-dir \
        transformers==4.39.3 \
        accelerate \
        sentencepiece \
        huggingface_hub

%environment
    # Keep torch cache small and writable
    export TORCH_HOME=/tmp/torch

%runscript
    exec python "$@"
```

This container provides:

- Python
- CUDA
- PyTorch
- Transformers
- HF Hub

---

## 3️⃣ Build the container (one time)

On most clusters (no root):

```bash
module load singularity
singularity build --fakeroot llm.sif llm.def
```

or (if fakeroot not allowed):

```bash
singularity build --remote llm.sif llm.def
```

This creates: `llm.sif`

---

## 4️⃣ Write your program

Example: `llm_test.py`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "Write a short haiku about GPUs and Slurm."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 5️⃣ Create writable cache directories

HPC filesystems are often read-only in containers.

```bash
mkdir -p /work/username/.hf_cache
```

---

## 6️⃣ Write the Slurm job script

`run_llm.sh`

```bash
#!/bin/bash
#SBATCH --job-name=llama8b_test
#SBATCH --partition=gpu_short
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=llm.out
#SBATCH --error=llm.err

module load singularity

mkdir -p /work/username/.hf_cache

export SINGULARITYENV_HF_HOME=/work/username/.hf_cache
export SINGULARITYENV_TRANSFORMERS_CACHE=/work/username/.hf_cache
export SINGULARITYENV_HF_TOKEN=hf_xxxxxxxxx

singularity exec --nv llm.sif python llm_test.py
```

Why:

- `SINGULARITYENV_` injects env vars into container
- avoids Read-only file system: `/.cache`
- avoids HF login prompts

---

## 7️⃣ Submit the job

```bash
sbatch run_llm.sh
```

Check status:

```bash
squeue -p gpu_short
```

---

## 8️⃣ View output

```bash
cat llm.out
cat llm.err
```

---

## 9️⃣ Mount external data (optional)

```bash
singularity exec --nv \
  -B /work/username/data:/data \
  llm.sif python llm_test.py
```

In Python:

```python
open("/data/file.txt")
```

---

## 🔟 Multiple GPUs

Job script:

```bash
#SBATCH --gres=gpu:2
```

Python:

```python
import torch
print(torch.cuda.device_count())
```

---

## ⚠️ Common errors & fixes

### ❌ Read-only file system: /.cache

Fix:

```bash
export SINGULARITYENV_HF_HOME=/work/username/.hf_cache
```

### ❌ ModuleNotFoundError

You used system Python. Fix:

```bash
singularity exec llm.sif python script.py
```

### ❌ CUDA available: False

Forgot: `--nv`

### ❌ You must be root to build

Use:

```bash
singularity build --fakeroot
```

or:

```bash
singularity build --remote
```

---

## ✅ Minimal working template

```bash
#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --partition=gpu_short
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=job.out
#SBATCH --error=job.err

module load singularity

singularity exec --nv myimage.sif python my_script.py
```

---

---

# Interactive GPU jobs (minimal command summary)

```bash
# on laptop
code .
# connect to HPC via SSH extension

# in VS Code terminal
srun --partition=gpu_intr --gres=gpu:1 --pty bash # enter interactive session
module load singularity
singularity exec --nv llm.sif bash # enter container
# now in container
mkdir -p /work/username/.hf_cache
export SINGULARITYENV_HF_HOME=/work/username/.hf_cache
export SINGULARITYENV_TRANSFORMERS_CACHE=/work/username/.hf_cache
export SINGULARITYENV_HF_TOKEN=hf_xxxxxxxxx

# start python script
python llm_test.py
# or start jupyter
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
## then select jupyter server in VS Code
```

---

## 🧩 Summary

- `.def` → defines environment
- `.sif` → built container
- `sbatch` → schedules job
- `--nv` → enables GPU
- `SINGULARITYENV_*` → fixes caches & tokens

---

## Quick command reference

```bash
# Check GPU nodes and resources
scontrol show node cc21isgpu00 | grep -E "Gres|AllocTRES"
scontrol show partition gpu_short | grep -E "Gres|AllocTRES"



export OPENROUTER_API_KEY="sk-or-..."

pyenv exec python phase3/generation/generate_openai.py \
  --base-url "https://router.huggingface.co/v1" \
  --api-key "$HF_TOKEN" \
  --model "Qwen/Qwen2.5-72B-Instruct" \
  --builder qwen \
  --router huggingface \
  --quantization bf16 \
  --provider novita \
  --limit 308 \
  --variations ALL


pyenv exec python phase3/generation/generate_openai.py \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key "$OPENROUTER_API_KEY" \
  --model "qwen/qwen-2.5-72b-instruct" \
  --builder qwen \
  --router openrouter \
  --quantization bf16 \
  --provider hyperbolic \
  --limit 2 \
  --variations P-4 \
  --resume phase3/datasets/Qwen_Qwen2.5-72B-Instruct_1771217443.csv

pyenv exec python phase3/generation/generate_openai.py \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key "$OPENROUTER_API_KEY" \
  --model "meta-llama/llama-3.1-70b-instruct" \
  --quantization bf16 \
  --provider "deepinfra" \
  --limit 308 \
  --variations ALL \
  --resume phase3/datasets/meta-llama_llama-3.1-70b-instruct_1771215436.csv

pyenv exec python phase3/generation/generate_openai.py \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key "$OPENROUTER_API_KEY" \
  --model "meta-llama/llama-3.1-70b-instruct" \
  --quantization bf16 \
  --provider "deepinfra" \
  --limit 1 \
  --variations P-1

export OPENAI_API_KEY=<your-openai-api-key>

pyenv exec python phase3/generation/generate_chatgpt.py --limit 308 --variation ALL

export HF_TOKEN=<your-hf-token>

 .venv/bin/python3.12 phase3/detection/run_detection.py --detectors zerogpt roberta roberta-qa
```
