#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ======== 配置区 ========
MODEL_PATH="<YOUR_TARGET_MODEL_PATH>"
DATASET_NAME="<YOUR_DATASET_NAME>"   # 需提前注册到 training/sft/LLaMA-Factory/data/dataset_info.json
LR=1e-5
EPOCHS=3.0
OUTPUT_DIR="./out/${DATASET_NAME}"
YAML_CONFIG="$SCRIPT_DIR/../training/sft/training_target_model.yaml"
# ========================

BATCH_SIZE=1
NUM_WORKERS=16
GRAD_ACCUM=1
SEED=42
LOGGING_STEPS=50
SAVE_STEPS=200

llamafactory-cli train "${YAML_CONFIG}" \
    model_name_or_path="$MODEL_PATH" \
    dataset="$DATASET_NAME" \
    per_device_train_batch_size="$BATCH_SIZE" \
    learning_rate="$LR" \
    num_train_epochs="$EPOCHS" \
    output_dir="$OUTPUT_DIR" \
    seed="$SEED" \
    preprocessing_num_workers="$NUM_WORKERS" \
    gradient_accumulation_steps="$GRAD_ACCUM" \
    logging_steps="$LOGGING_STEPS" \
    save_steps="$SAVE_STEPS"
