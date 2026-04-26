#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ======== 配置区 ========
MODEL_PATH="<YOUR_VERL_CHECKPOINT_DIR>"  # verl 输出目录（含 global_step_* 子目录）
PROJECT_NAME="<YOUR_PROJECT_NAME>"
STEP=40
# ========================

cd "$SCRIPT_DIR/../training/rl_prompter/verl"

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir  "${MODEL_PATH}/${PROJECT_NAME}/global_step_${STEP}/actor" \
    --target_dir "${MODEL_PATH}/${PROJECT_NAME}/global_step_${STEP}/actor/huggingface"
