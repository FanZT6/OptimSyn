#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/base_training_args.sh"

export WANDB_DISABLED=true

# ======== 配置区 ========
MODEL_PATH="<YOUR_TARGET_MODEL_PATH>"       # e.g. /path/to/Qwen3-8B-Base
TRAIN_DATA_PATH="<YOUR_WARMUP_DATA_PATH>"  # e.g. ./data/warmup/qa_pairs.jsonl
data_dir="$SCRIPT_DIR/../data/warmup"
percentage=1.0
data_seed=6
job_name="warmup-lora-seed${data_seed}"
# ========================

output_dir=./out/${job_name}
mkdir -p "$output_dir"

train_files=("$TRAIN_DATA_PATH")

training_args="$base_training_args \
--model_name_or_path $MODEL_PATH \
--output_dir $output_dir \
--percentage $percentage \
--data_seed $data_seed \
--train_files ${train_files[@]} 2>&1 | tee $output_dir/train.log"

eval "$header" "$training_args"
