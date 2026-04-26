# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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

from collections import defaultdict
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", num_workers: int | None = None) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            num_workers: Max worker threads for parallel scoring/decoding. Defaults to min(32, os.cpu_count()).
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.num_workers = num_workers or min(32, (os.cpu_count() or 8))

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # --------- 多线程处理开始 ---------
        def _process_one(i: int):
            """子线程内处理单条样本：截取有效 token、解码、打分。"""
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            attn = data_item.batch["attention_mask"]

            # 计算有效长度
            valid_prompt_length = int(attn[:prompt_length].sum().item())
            valid_response_length = int(attn[prompt_length:].sum().item())

            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_ids = response_ids[:valid_response_length]

            # 解码
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # 取标注与元信息
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            # 计算分数
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            # 兼容两种返回：标量 或 字典
            if isinstance(score, dict):
                reward = score["score"]
                score_dict = score
                score_is_dict = True
            else:
                reward = score
                score_dict = {"score": score}  # 仅用于打印
                score_is_dict = False

            return {
                "i": i,
                "reward": float(reward),
                "valid_response_length": valid_response_length,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "score_dict": score_dict,
                "score_is_dict": score_is_dict,
                "data_source": data_source,
            }

        results = [None] * len(data)
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = [ex.submit(_process_one, i) for i in range(len(data))]
            for fut in as_completed(futures):
                res = fut.result()
                results[res["i"]] = res

        # 主线程合并结果（保持与原实现相同的顺序与打印逻辑）
        for res in results:
            i = res["i"]
            vrl = res["valid_response_length"]
            reward_tensor[i, vrl - 1] = res["reward"]

            if res["score_is_dict"]:
                for key, value in res["score_dict"].items():
                    reward_extra_info[key].append(value)

            ds = res["data_source"]
            if ds not in already_print_data_sources:
                already_print_data_sources[ds] = 0

            if already_print_data_sources[ds] < self.num_examine:
                already_print_data_sources[ds] += 1
                print("[prompt]", res["prompt_str"])
                print("[response]", res["response_str"])
                print("[ground_truth]", res["ground_truth"])
                if res["score_is_dict"]:
                    for key, value in res["score_dict"].items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", res["score_dict"]["score"])
        # --------- 多线程处理结束 ---------

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor