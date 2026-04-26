#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from annotated_types import Len
import requests
import time
import logging
import random
from functools import partial
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset, load_from_disk


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_jsonl(path):
    """Load JSONL file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_prompt_template(path):
    """Load the prompt template text."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_json_from_text(text):
    """Extract the first JSON object from the text."""
    try:
        text_data = text.split("</think>")[-1].strip()
        match = re.search(r"\{.*\}", text_data, re.S)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        logging.warning(f"JSON extraction failed: {e}")
    return None


def build_chat_prompt(prompt_template, domain, doc, std_q, std_a, std_qa, trainability):
    """Builds a chat prompt from template and variables."""

    system_prompt = f"You are an expert in the field of {domain}. I will now provide you with a document. Based on this document, you need to synthesize high‑quality question‑answer (QA) data, which will be used for downstream task training."
    def safe_to_str(obj):
        """将 dict/list 转成 JSON 字符串，其他类型转成 str。"""
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, ensure_ascii=False)
        return str(obj) if obj is not None else ""

    final_prompt_text = (
        prompt_template
        .replace("{doc}", safe_to_str(doc))
        .replace("{Prompt-related}", safe_to_str(std_q))
        .replace("{Response-related}", safe_to_str(std_a))
        .replace("{Prompt-Response alignment}", safe_to_str(std_qa))
        .replace("{Technical/trainability aspects}", safe_to_str(trainability))
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_prompt_text}
    ]
    return messages


def call_gpt_service(messages, model="gpt-4.1-2025-04-14"):
    """Call GPT service."""
    url = "https://xxx/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-xxx"
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]["content"]
        completion_tokens = data["usage"].get("completion_tokens", None)
        return message.strip(), completion_tokens
    except Exception as e:
        logging.error(f"Request failed: {e}")
        return None, None

def single_query(content, model):
    """Send a single query to the model service."""
    output = None
    while output is None:
        time.sleep(random.uniform(1, 30))  # Avoid tight loop
        if model.startswith("gpt"):
            output, tokens = call_gpt_service(content, model)
        else:
            logging.error(f"Unsupported model type: {model}")
            break
    return output


def batch_query(contents, model, num_proc=64):
    """Batch query using multi-threading."""
    query_func = partial(single_query, model=model)
    results = []

    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        for res in tqdm(executor.map(query_func, contents), total=len(contents), desc=f"Querying {model}"):
            results.append(res)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction_file", required=True, help="Path to instruction JSONL file")
    parser.add_argument("--prompt_file", required=True, help="Path to prompt template file")
    parser.add_argument("--domain", required=True, help="Domain string to insert into prompt")
    parser.add_argument("--output_file", required=True, help="Where to save extracted JSONL outputs")
    parser.add_argument("--model", required=True, help="Model name to use for generation")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit the number of samples processed")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of parallel threads")
    args = parser.parse_args()

    instructions = load_jsonl(args.instruction_file)

    prompt_template = load_prompt_template(args.prompt_file)

    contents = []
    for data in tqdm(instructions):
        domain_doc = data.get("content", "")

        std_q = data.get("Prompt-related", "")
        std_a = data.get("Response-related", "")
        std_qa = data.get("Prompt-Response alignment", "")
        trainable = data.get("Technical/trainability aspects", "")
        prompt = build_chat_prompt(prompt_template, args.domain, domain_doc, std_q, std_a, std_qa, trainable)

        contents.append(prompt)

    outputs = batch_query(contents, args.model, num_proc=args.num_proc)

    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for i, output in enumerate(outputs):
            json_obj = extract_json_from_text(output) if output else None
            if json_obj:
                out_f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
            else:
                logging.warning(f"Could not extract JSON for index {i}.")


if __name__ == "__main__":
    main()
