#!/bin/bash

# Configuration
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 1. Run LLAMA Generation
echo "Starting LLAMA Generation..."
pyenv exec python generation/generate_openai.py \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key "sk-or-v1-xx" \
  --model "meta-llama/llama-3.1-70b-instruct" \
  --quantization bf16 \
  --provider "deepinfra" \
  --limit 308 \
  --variations ALL \
  --workers 5 \
  > "$LOG_DIR/llama_run.log" 2>&1 &
PID_LLAMA=$!
echo "LLAMA PID: $PID_LLAMA"

# 2. Run QWEN Generation
echo "Starting QWEN Generation..."
pyenv exec python generation/generate_openai.py \
  --base-url "https://router.huggingface.co/v1" \
  --api-key "hf_xx" \
  --model "Qwen/Qwen2.5-72B-Instruct" \
  --builder qwen \
  --router huggingface \
  --quantization bf16 \
  --provider novita \
  --limit 308 \
  --variations ALL \
  --workers 5 \
  > "$LOG_DIR/qwen_run.log" 2>&1 &
PID_QWEN=$!
echo "QWEN PID: $PID_QWEN"

# 3. Run CHATGPT Generation
echo "Starting CHATGPT Generation..."
pyenv exec python generation/generate_chatgpt.py \
  --limit 308 \
  --variation ALL \
  --workers 5 \
  > "$LOG_DIR/chatgpt_run.log" 2>&1 &
PID_GPT=$!
echo "CHATGPT PID: $PID_GPT"

# Wait for ALL processes to finish
echo "Waiting for all processes..."
wait $PID_LLAMA
wait $PID_QWEN
wait $PID_GPT

echo "All processes finished."

