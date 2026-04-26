#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ======== 配置区 ========
MODEL_PATH="<YOUR_WARMUP_CHECKPOINT_PATH>"  # Stage ① 输出的 LoRA checkpoint
BASE_PORT=8003
GPUS=(0)                # smoke: single GPU
MAX_WORKERS=8
PER_DEVICE_CONCURRENCY=1
MAX_QUEUE=32
HOST="0.0.0.0"
LOG_DIR="./logs_batch"
APP="$SCRIPT_DIR/../training/influence/optimizer_batch_med.py"
# ========================

mkdir -p "$LOG_DIR"

for idx in "${!GPUS[@]}"; do
  GPU_ID="${GPUS[$idx]}"
  PORT=$((BASE_PORT + idx))
  LOG="$LOG_DIR/gpu${GPU_ID}.port${PORT}.log"

  echo "[LAUNCH] GPU:$GPU_ID  PORT:$PORT  LOG:$LOG"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  HOST="$HOST" PORT="$PORT" \
  MAX_WORKERS="$MAX_WORKERS" PER_DEVICE_CONCURRENCY="$PER_DEVICE_CONCURRENCY" MAX_QUEUE="$MAX_QUEUE" \
  TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="$SCRIPT_DIR/../training/influence" \
  nohup python "$APP" --model-path "$MODEL_PATH" >"$LOG" 2>&1 &
done

echo "All started. Check logs under $LOG_DIR"
