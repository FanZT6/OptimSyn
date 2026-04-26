#!/usr/bin/env bash
# Usage: bash generate_rubrics_med.sh <CUDA_VISIBLE_DEVICES> <STEP>
# e.g.:  bash generate_rubrics_med.sh 0,1,2,3,4,5,6,7 40

export CUDA_VISIBLE_DEVICES=$1
STEP=$2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

# ======== 配置区 ========
MODEL_PATH="<YOUR_VERL_CHECKPOINT_DIR>"
PROJECT_NAME="<YOUR_PROJECT_NAME>"
DOMAIN="Medical and Health"
DOMAIN_x="medical"
DATA_FILE="<YOUR_SEED_DATA_FILE>"         # e.g. ./data/seed/medical_seed.jsonl
TEACHER_MODEL="<YOUR_TEACHER_MODEL>"      # closed API 模型名，或 vLLM 服务的 model id
# export DASHSCOPE_API_KEY="<YOUR_API_KEY>"   # 使用 Qwen API 时取消注释
# export OPENAI_API_KEY="<YOUR_API_KEY>"      # 使用 OpenAI API 时取消注释
# ========================

PROMPTER_PATH="${MODEL_PATH}/${PROJECT_NAME}/global_step_${STEP}/actor/huggingface"
RUBRIC_OUT="$ROOT/data/rubrics/${DOMAIN_x}/${PROJECT_NAME}/step_${STEP}.jsonl"
QA_OUT="$ROOT/data/qa_pairs/${DOMAIN_x}/${PROJECT_NAME}/step_${STEP}.jsonl"

python "$ROOT/data_synthesis/generate_rubrics.py" \
  --data_file "$DATA_FILE" \
  --prompt_file "$ROOT/data_synthesis/prompts/generate_rubrics.txt" \
  --system_prompt_file "$ROOT/data_synthesis/system_prompts/generate_rubrics.txt" \
  --domain "$DOMAIN" \
  --output_file "$RUBRIC_OUT" \
  --model_path "$PROMPTER_PATH" \
  --max_tokens 8192 \
  --temperature 0.7 \
  --top_p 0.9 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.9 \
  --max_samples 0.99

python "$ROOT/data_synthesis/generate_qa.py" \
  --instruction_file "$RUBRIC_OUT" \
  --prompt_file "$ROOT/data_synthesis/prompts/generate_qa.txt" \
  --domain "$DOMAIN" \
  --output_file "$QA_OUT" \
  --model "$TEACHER_MODEL" \
  --num_proc 32 \
  --max_samples 26000 \
  --max_retries 3
