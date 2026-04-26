#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import requests
import time
import logging
import random
from functools import partial
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

from dashscope import Generation
import dashscope

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# 如果你要用新加坡地域，可以解除下面这行注释
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"


# =====================
# 工具函数
# =====================

def load_jsonl(path: str):
    """Load JSONL file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_prompt_template(path: str) -> str:
    """Load the prompt template text."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_json_from_text(text: str):
    """从模型返回文本中抽取第一个 JSON 对象。"""
    if not text:
        return None
    try:
        # 如果有 <think>...</think> 这类前缀，先切掉
        text_data = text.split("</think>")[-1].strip()
        match = re.search(r"\{.*\}", text_data, re.S)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        logging.warning(f"JSON extraction failed: {e}")
    return None


def build_chat_prompt(prompt_template: str,
                      domain: str,
                      doc: str,
                      std_q: str,
                      std_a: str):
    """Builds a chat prompt from template and variables."""
    final_prompt_text = (
        prompt_template
        .replace("{doc}", doc)
        .replace("{Prompt-related}", std_q)
        .replace("{Response-related}", std_a)
    )
    system_prompt = (
        f"You are an expert in the field of {domain}. "
        "I will now provide you with a document. "
        "Based on this document, you need to synthesize high-quality question-answer (QA) data, "
        "which will be used for downstream task training."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_prompt_text}
    ]
    return messages


# =====================
# GPT 服务（如果你会用到）
# =====================

def call_gpt_service(messages, model: str = "gpt-4.1-2025-04-14") -> Optional[str]:
    """Call GPT-like service via HTTP.

    注意：这里的 url / api_key 是占位符，改成你自己的。
    """
    url = "https://xxx/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-xxx",  # 建议改成 os.getenv("GPT_API_KEY")
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 2048,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]["content"]
         # 只返回文本，方便后续统一处理
        return message.strip()
    except Exception as e:
        logging.error(f"GPT request failed: {e}")
        return None


# =====================
# Qwen 调用（dashscope 官方方式）
# =====================

def call_QWEN_service(messages, model: str = "qwen-plus") -> Optional[str]:
    """调用百炼 Qwen 模型（使用 dashscope.Generation.call）。"""
    try:
        response = Generation.call(
            # 建议在环境变量中配置 DASHSCOPE_API_KEY
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=model,
            messages=messages,
            result_format="message",
        )

        if response.status_code == 200:
            # 正常返回
            return response.output.choices[0].message.content
        else:
            # 非 200 的错误信息
            logging.error(
                f"DashScope error: http_status={response.status_code}, "
                f"code={response.code}, message={response.message}"
            )
            return None
    except Exception as e:
        logging.error(f"QWEN request exception: {e}")
        return None


# =====================
# 单条 & 批量查询
# =====================

def single_query(content, model: str,
                 max_retries: int = 3,
                 min_delay: float = 0.5,
                 max_delay: float = 2.0) -> Optional[str]:
    """Send a single query to the model service, with limited retries."""
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            # 重试之间稍微 sleep 一下，避免打爆 API
            time.sleep(random.uniform(min_delay, max_delay))

        if model.startswith("gpt"):
            output = call_gpt_service(content, model)
        elif model.startswith("qwen"):
            output = call_QWEN_service(content, model)
        else:
            logging.error(f"Unsupported model type: {model}")
            return None

        if output is not None:
            return output

        logging.warning(f"{model}: attempt {attempt}/{max_retries} failed, retrying...")

    logging.error(f"{model}: max retries exceeded for one sample, giving up.")
    return None


def batch_query(contents, model: str, num_proc: int = 64,
                max_retries: int = 3):
    """Batch query using multi-threading."""
    query_func = partial(single_query, model=model, max_retries=max_retries)
    results = []

    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        for res in tqdm(executor.map(query_func, contents),
                        total=len(contents),
                        desc=f"Querying {model}"):
            results.append(res)
    return results


# =====================
# 主函数
# =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction_file", required=True,
                        help="Path to instruction JSONL file")
    parser.add_argument("--prompt_file", required=True,
                        help="Path to prompt template file")
    parser.add_argument("--domain", required=True,
                        help="Domain string to insert into prompt")
    parser.add_argument("--output_file", required=True,
                        help="Where to save extracted JSONL outputs")
    parser.add_argument("--model", required=True,
                        help="Model name to use for generation (e.g., 'qwen-plus')")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit the number of samples processed")
    parser.add_argument("--num_proc", type=int, default=64,
                        help="Number of parallel threads")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retries per sample")
    args = parser.parse_args()

    # 1. 读入指令
    instructions = load_jsonl(args.instruction_file)
    if args.max_samples is not None:
        instructions = instructions[:args.max_samples]

    # 2. 读入模板
    prompt_template = load_prompt_template(args.prompt_file)

    # 3. 构造所有 messages
    contents = []
    for data in tqdm(instructions, desc="Building chat prompts"):
        domain_doc = data.get("clean_text", "")  # 或 data.get("text", "")
        std_q = data.get("Prompt-related", "")
        std_a = data.get("Response-related", "")
        messages = build_chat_prompt(
            prompt_template,
            args.domain,
            domain_doc,
            std_q,
            std_a
        )
        contents.append(messages)

    # 4. 并发调用模型
    outputs = batch_query(
        contents,
        args.model,
        num_proc=args.num_proc,
        max_retries=args.max_retries
    )

    # 5. 解析 JSON 并写出
    success_cnt = 0
    total = len(outputs)
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for i, output in enumerate(outputs):
            json_obj = extract_json_from_text(output) if output else None
            if json_obj:
                out_f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                success_cnt += 1
            else:
                logging.warning(f"Could not extract JSON for index {i}.")

    logging.info(f"Done. Parsed JSON for {success_cnt}/{total} samples. "
                 f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()