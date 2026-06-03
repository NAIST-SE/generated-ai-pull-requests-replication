#!/bin/bash

# Configuration
LOG_DIR="detection/logs"
mkdir -p "$LOG_DIR"

echo "Step 1: Preparing Data..."
# Merges the 3 result files into datasets/generated_descriptions.csv
python3 detection/prepare_data.py

echo "Step 2: Running Detection..."
# Detectors: ZeroGPT, Roberta, Roberta-QA
# Using --resume to allow stopping/starting
# Using --workers 5 for parallel processing

python3 detection/run_detection.py \
    --detectors zerogpt roberta roberta-qa \
    --workers 5 \
    > "$LOG_DIR/detection_run.log" 2>&1 &

PID=$!
echo "Detection started with PID: $PID"
echo "Monitor logs: tail -f $LOG_DIR/detection_run.log"

wait $PID
echo "Detection pipeline finished."
