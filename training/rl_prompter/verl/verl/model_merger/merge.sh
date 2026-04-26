MODEL_PATH="/xboxtrain/share/xboxtrain/zhiting/models/rl/qwen_3"
PROJECT_NAME=HSS_1.0_Qwen3_235b
EXPERIMENT_NAME=qwen3_8b_function_rm_Med
STEP=65

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ${MODEL_PATH}/${PROJECT_NAME}/global_step_${STEP}/actor \
    --target_dir ${MODEL_PATH}/${PROJECT_NAME}/global_step_${STEP}/actor/huggingface