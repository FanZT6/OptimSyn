#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
from tqdm import tqdm


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """读取 jsonl 文件，一行一个 dict。"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error in {file_path}: {e} | line[:80]= {line[:80]!r}")
    return data


def main():
    parser = argparse.ArgumentParser()
    # === I/O ===
    parser.add_argument("--data_file", required=True, help="输入 jsonl 文件（每行一个样本，含 clean_text 等字段）")
    parser.add_argument("--output_file", required=True, help="生成 rubrics 后输出的 jsonl 文件路径")

    # === prompt 模板 ===
    parser.add_argument("--prompt_file", required=True, help="user prompt 模板文件（generate_rubrics.txt）")
    parser.add_argument("--system_prompt_file", default=None,
                        help="system prompt 模板文件（generate_rubrics.txt 的 system 版）")
    parser.add_argument("--domain", required=True, help="领域名，用于替换模板中的 {domain}")

    # === vLLM / 模型相关 ===
    parser.add_argument("--model_path", required=True, help="vLLM 加载的 HF 模型路径")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # === 采样数量控制 ===
    parser.add_argument(
        "--max_samples",
        type=float,
        default=1.0,
        help="如果 <=1 视为比例；>1 视为最大条数（默认 1.0 表示用全部样本）",
    )

    args = parser.parse_args()

    # 1. 读取模板
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        user_prompt_template = f.read()

    system_prompt_template = ""
    if args.system_prompt_file is not None:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt_template = f.read()

    # 2. 读入原始 jsonl 数据
    print(f"[INFO] Loading data from {args.data_file} ...")
    samples = load_jsonl(args.data_file)
    total_n = len(samples)
    print(f"[INFO] Loaded {total_n} samples.")

    # 3. 采样子集（可选）
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be > 0")

    if args.max_samples < 1.0:
        use_n = int(total_n * args.max_samples)
    else:
        use_n = min(int(args.max_samples), total_n)

    samples = samples[:use_n]
    print(f"[INFO] Using {len(samples)} samples for generation.")

    # 4. 加载 vLLM 模型
    print(f"[INFO] Loading vLLM model from {args.model_path} ...")
    llm = LLM(
        model=args.model_path, \
        tensor_parallel_size=args.tensor_parallel_size, \
        gpu_memory_utilization=args.gpu_memory_utilization, \
        trust_remote_code=True
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # 5. 构造 prompts（仿照你给的 preprocess 脚本逻辑）
    print("[INFO] Building prompts from clean_text ...")
    prompts: List[str] = []
    meta_infos: List[Dict[str, Any]] = []

    for idx, ex in enumerate(samples):
        # 1) 取 seed 文本：优先 clean_text，其次 text，再次 raw_text
        content = ex.get("clean_text") or ex.get("text") or ex.get("raw_text") or ""
        if not isinstance(content, str):
            content = str(content)

        # 2) 使用模板构建 user/system prompt
        user_prompt = (
            user_prompt_template
            .replace("{document}", content)
            .replace("{domain}", args.domain)
        )
        if system_prompt_template:
            system_prompt = system_prompt_template.replace("{domain}", args.domain)
        else:
            system_prompt = ""

        # 3) 构造 chat messages
        chat_msgs = []
        if system_prompt:
            chat_msgs.append({"role": "system", "content": system_prompt})
        chat_msgs.append({"role": "user", "content": user_prompt})

        # 4) 用 tokenizer 的 chat_template 拼成最终字符串 prompt
        prompt_text = tokenizer.apply_chat_template(
            chat_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompts.append(prompt_text)
        meta_infos.append(
            {
                "idx": idx,
                "content": content,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "chat_msgs": chat_msgs,
            }
        )

    # 6. 调用 vLLM 生成
    print(f"[INFO] Generating rubrics for {len(prompts)} samples ...")
    outputs = llm.generate(prompts, sampling_params)

    # 7. 写回到 output_file：在原始样本基础上附加 prompt/completion 等字段
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    print(f"[INFO] Saving results to {args.output_file} ...")

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for meta, gen_out in zip(meta_infos, outputs):
            ex = samples[meta["idx"]]

            # vLLM 生成的文本（rubrics）
            gen_text = gen_out.outputs[0].text.strip()

            # 按你 preprocess 脚本的结构补充字段
            ex["data_source"] = args.data_file
            ex["prompt"] = meta["chat_msgs"]  # system + user
            ex["ability"] = args.domain
            ex["reward_model"] = {
                "style": "rule",
                "ground_truth": ""
            }
            ex["extra_info"] = {
                "seed_data": meta["content"],
                "domain": args.domain
            }
            # 生成结果放在 completion 字段（后续 generate_qa 或其它脚本直接用）
            ex["completion"] = gen_text

            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("[DONE] Rubrics generation finished.")


if __name__ == "__main__":
    main()