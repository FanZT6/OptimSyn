import os
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import requests
import re
import time
import json
import logging
import random
from functools import partial
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def api(inputs: list) -> dict:

    host_ip = os.getenv("REWARD_SERVICE_HOST", "127.0.0.1")
    urls = [
        f"http://{host_ip}:8801/gradients/sync",
    ]
    
    primary_url = urls[0]
    backup_url = primary_url

    def call_api(url):
        resp = requests.post(url, json={"inputs": inputs}, timeout=1200)
        resp.raise_for_status()
        result = resp.json()
        grad_info = result.get('result', {})
        if not grad_info:
            print(f"No grad_info in result from {url}:", result)
            return None
        
        return grad_info

    try:
        scores = call_api(primary_url)
        if scores is not None:
            return scores
        print(f"Primary API {primary_url} returned no usable data, trying backup API {backup_url}")
    except Exception as e:
        print(f"Error calling primary API {primary_url}: {e}")

    try:
        scores = call_api(backup_url)
        if scores is not None:
            return scores
        print(f"Backup API {backup_url} returned no usable data.")
    except Exception as e:
        print(f"Error calling backup API {backup_url}: {e}")

    return {}


def build_chat_prompt(prompt_template, domain, doc, std_q, std_a, std_qa, std_trainability):
    """Builds a chat prompt from template and variables."""
    final_prompt_text = (
        prompt_template
        .replace("{doc}", doc)
        .replace("{Prompt-related}", std_q)
        .replace("{Response-related}", std_a)
        .replace("{Prompt-Response alignment}", std_qa)
        .replace("{Technical/trainability aspects}", std_trainability)
    )
    system_prompt = f"You are an expert in the field of {domain}. I will now provide you with a document. Based on this document, you need to synthesize high‑quality question‑answer (QA) data, which will be used for downstream task training."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_prompt_text}
    ]
    # print(messages)
    # exit()
    return messages

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

def call_gemini_service(messages, model="gemini-2.5-pro-preview-06-05"):

    url = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('API_KEY', '<YOUR_API_KEY>')}"
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
        completion_tokens = data.get("usage", {}).get("completion_tokens", None)
        return message.strip(), completion_tokens
    except Exception as e:
        logging.error(f"Request failed: {e}")
        return None, None

def call_gpt_service(messages, model="gpt-4.1-2025-04-14"):
    """Call GPT service."""
    url = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('API_KEY', '<YOUR_API_KEY>')}"
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


def call_qwen_service(messages, model="qwen3-235b-instruct"):
    """
    Call locally deployed Qwen model API (vLLM, OpenAI-compatible).
    Set VLLM_BASE_URL to point at your own vLLM server (see scripts/vllm.sh).
    """
    url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000") + "/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0, 
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=1200)
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]["content"]
        completion_tokens = data.get("usage", {}).get("completion_tokens", None)
        return message.strip(), completion_tokens
    except Exception as e:
        logging.error(f"Local Qwen request failed: {e}")
        print(messages)
        return None, None


def single_query(content, model, max_retries=3):
    """
    Send a single query to the model service with format checking and retries.

    Args:
        content (str): The input query to send to the model.
        model (str): The model name (e.g., "gpt-3.5", "sglang-test").
        max_retries (int): Maximum number of retries if format check fails.

    Returns:
        dict | None: Parsed JSON result if successful, otherwise None.
    """
    if content is None or content == "":
        return {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ""}
            ]
        }, 0.0
    attempt = 0
    inputs = None

    # print(content)
    while attempt < max_retries:
        attempt += 1

        # Random delay to prevent sending too many concurrent requests
        time.sleep(random.uniform(1, 3))

        output = None
        if model.startswith("gemini"):
            output, _ = call_gemini_service(content, model)
        if model.startswith("gpt"):
            output, _ = call_gpt_service(content, model)
        elif model.lower().startswith("qwen3"):
            output, _ = call_qwen_service(content, model)
        else:
            logging.error(f"Unsupported model type: {model}")
            break

        # Try to parse JSON from model output
        inputs = extract_json_from_text(output)

        if inputs is not None:
            return inputs, 1.0
        else:
            print(f"[WARNING] Attempt {attempt}: Failed to extract JSON from output, retrying...")

    print("[ERROR] Failed to extract valid JSON after retries in single_query.")
    
    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""}
        ]
    }, 0.0


def batch_query(contents, model, num_proc=8):
    """
    Query a list of contents in parallel using multiple threads.

    Args:
        contents (list[str]): A list of query strings.
        model (str): Model name to query.
        num_proc (int): Number of parallel threads.

    Returns:
        list[dict | None]: A list of parsed JSON objects or None for failures.
    """
    query_func = partial(single_query, model=model)
    results, format_reward = [], []

    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        for res, if_correct in tqdm(executor.map(query_func, contents), total=len(contents), desc=f"Querying {model}"):
            results.append(res)
            format_reward.append(if_correct)

    # print(results)
    return results, format_reward


from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm


def normalize_scores(scores_dict):
    """
    Normalize the scores in the scores_dict such that
    their values are in the range [0, 1].

    Args:
        scores_dict (dict[str, list[float]]): A dictionary containing
                                               scores for different tasks.

    Returns:
        dict[str, list[float]]: A dictionary with normalized scores.
    """
    normalized_scores = {}
    
    for task, scores in scores_dict.items():
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score > min_score: 
            normalized_scores[task] = [(score - min_score) / (max_score - min_score) for score in scores]
        else:
            normalized_scores[task] = [0.0 for score in scores] 
    
    return normalized_scores

def get_as_str(obj, key):
    """获取json_obj[key] 的字符串形式"""
    value = obj.get(key, "")
    # 如果是 dict 或 list，则序列化为字符串
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    # 其他类型直接转成字符串（避免 int/float 报错）
    return str(value)

def compute_score(solution_str: List[str], extra_info: List[Dict[str, Any]]) -> List[float]:
    """
    Args:
        solution_str: List of solution strings (decoded model outputs for each sample)
        extra_info:   List of dict for each sample, may contain seed_data/domain etc.
    Returns:
        List[float]: Average score per sample across tasks
    """
    prompt_file = os.path.join(os.path.dirname(__file__), "../../../../../../data_synthesis/prompts/generate_qa_v3.txt")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # --------- 2. Build chat prompt messages ----------
    messages = []
    for prompt, seed in zip(solution_str, extra_info):
        if not isinstance(seed, dict):
            seed = {}
        seed_data = seed.get("seed_data", {})
        domain = seed.get("domain", "Humanity and Social Sciences")

        json_obj = extract_json_from_text(prompt) if prompt else None
        if not json_obj:
            # instead of returning 0.0, produce a placeholder for this sample
            messages.append(None)
            continue

        std_q = get_as_str(json_obj, "Prompt-related")
        std_a = get_as_str(json_obj, "Response-related")
        std_qa = get_as_str(json_obj, "Prompt-Response alignment")
        std_trainability = get_as_str(json_obj, "Technical/trainability aspects")

        try:
            message = build_chat_prompt(prompt_template, domain, seed_data, std_q, std_a, std_qa, std_trainability)
            messages.append(message)
        except Exception as e:
            print(f"[ERROR] Failed to build prompt for sample: {e}")
            messages.append(None)
            print(f"Question Quality Standard: {std_q}")
            print(f"answer Quality Standard: {std_a}")

    batch_inputs = [m if m is not None else "" for m in messages]

    api_results, format_rewards = batch_query(batch_inputs, model="qwen3-235b-instruct")

    scores_dict = api(api_results)

    num_samples = len(solution_str)
    tasks = list(scores_dict.keys())

    scores_dict = normalize_scores(scores_dict)
    # if len(tasks) == 1:
    task = tasks[0]
    average_scores = np.array(scores_dict[task], dtype=np.float32) * np.array(format_rewards, dtype=np.float32)
    print(f"scores in scores_dict: {average_scores}")
    return average_scores.tolist()

    # average_scores = []
    # for i in range(num_samples):
    #     # 收集当前样本的各任务分数
    #     task_scores = [scores_dict[task][i] for task in tasks if i < len(scores_dict[task])]
    #     if task_scores:
    #         mean_score = sum(task_scores) / len(task_scores)
    #     else:
    #         mean_score = 0.0
    #     average_scores.append(mean_score)

    # # 转为 numpy 并逐元素乘以奖励系数
    # average_scores = np.array(average_scores, dtype=np.float32) * np.array(format_rewards, dtype=np.float32)

    # return average_scores.tolist()


if __name__ == "__main__":

    inputs = 10 * ["""{"Prompt-related": ["Domain Relevance: Ensure questions are deeply rooted in the economic theory of worker safety, focusing on Adam Smith's compensating wage differentials and market mechanisms.", "Clarity and Unambiguity: Questions must be precise to avoid misinterpretation of complex concepts like marginal cost curves or risk-dollar tradeoffs.", "Bias Avoidance: Questions should not assume subjective stances (e.g., smoker vs. non-smoker attitudes) but instead test objective analysis of the text's examples."], "Response-related": ["Factual Accuracy: Answers must correctly reference the document's claims, such as the comparison of occupational deaths to choking or motor vehicle fatalities.", "Reasoning Depth: Answers should explain how worker preferences and firm incentives interact, as detailed in the marginal cost curve analysis.", "Information Density: Answers must balance conciseness with coverage of key elements (e.g., assumptions about risk awareness, safety equipment efficacy)."], "Prompt-Response alignment": ["Consistency: Questions and answers must align in scope, e.g., if the question references Figure 23.2, the answer must explicitly address the marginal cost curve's role.", "Format Compliance: For short-answer questions, answers should avoid listing bullet points and instead use structured paragraphs to mirror academic writing.", "Logical Linkage: Answers must directly connect to the question's focus (e.g., explaining how \\\"diminishing incremental value\\\" affects safety decisions)."], "Technical/trainability aspects": ["Parsability: QA pairs should avoid markdown formatting and use plain text for seamless processing by downstream models.", "Length Suitability: Answers should be concise enough to fit within typical context windows while retaining critical details like the \\\"no-risk level of safety\\\" concept.", "Copyright Compliance: Avoid reproducing verbatim text from the document, ensuring answers are paraphrased and original."]}"""]
    seed = 10 * [{"seed_data": "In the year of Our Lord 1527, when the frost lingered late into the spring, the merchants of the Eastern Marches gathered in the old guildhall of Brackenford to discuss matters of trade and security. Reports had reached them of increased raids along the riverways, carried out by bands of marauders who claimed no allegiance to any lord or crown.", "domain": "Humanities and Social Sciences"}]
    avg_score = compute_score(inputs, seed)
    print("Average score:", avg_score)
