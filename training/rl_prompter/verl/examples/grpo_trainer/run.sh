# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.

set -x
export WANDB_MODE="offline"
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
export TENSORBOARD_DIR="<YOUR_TENSORBOARD_LOG_DIR>"

# ======== 配置区 ========
PROMPTER_MODEL="<YOUR_PROMPTER_MODEL_PATH>"    # e.g. /path/to/Qwen3-8B
TRAIN_FILES="<YOUR_TRAIN_DATA_PATH>"           # .parquet 训练集
TEST_FILES="<YOUR_TEST_DATA_PATH>"             # .parquet 测试集
SAVE_PATH="<YOUR_CHECKPOINT_SAVE_DIR>"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REWARD_FUNCTION="$SCRIPT_DIR/../../verl/utils/reward_score/syn_batch.py"

TRAIN_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=64
LR=1e-6
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=8192
ROLLOUT_N=15
ROLLOUT_TEMPERATURE=1.5
ROLLOUT_GPU_UTIL=0.7
PROJECT_NAME="<YOUR_PROJECT_NAME>"
EXPERIMENT_NAME="<YOUR_EXPERIMENT_NAME>"
NNODES=1
N_GPUS_PER_NODE=8
SAVE_FREQ=10
TEST_FREQ=5
TOTAL_EPOCHS=1
# ========================

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$TEST_FILES \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$PROMPTER_MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS_PER_NODE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$REWARD_FUNCTION \
    trainer.critic_warmup=0 \
    trainer.logger='[tensorboard,wandb]' \
    trainer.log_val_generations=10 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir=$SAVE_PATH
