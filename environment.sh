SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/training/influence"
pip install -r requirements.txt
pip install -e .

pip install vllm==0.8.5.post1
pip install vllm-flash-attn==2.6.2
pip install torch==2.6.0
pip install flash_attn==2.6.2

pip install transformers==4.51.0
pip install peft==0.15.2
pip install triton==3.2.0
pip uninstall flash_attn

cd "$SCRIPT_DIR/evaluation/evalscope"

pip install -e '.[all]'

cd "$SCRIPT_DIR/training/sft/LLaMA-Factory"
pip install -e ".[torch,metrics]" --no-build-isolation

pip install ray==2.49.0
