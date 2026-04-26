#!/usr/bin/env bash
set -e

# ======== 配置区 ========
MODEL_NAME="<YOUR_TEACHER_MODEL_PATH>"  # e.g. /path/to/Qwen3-32B  或  Qwen/Qwen3-32B
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
HOST="0.0.0.0"
PORT="8000"
TP_SIZE=8
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.90
# ========================

export CUDA_VISIBLE_DEVICES

echo "Starting vLLM server: ${MODEL_NAME}  →  http://${HOST}:${PORT}"

vllm serve "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  "$@"
