#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient Service - Qwen2.5
- 支持同步/异步两种调用
- 基于线程池做排队与并发调度
- 用信号量限制占用 GPU 的并发度（避免 OOM）
- 提供健康检查 /healthz 与简易监控 /metrics
"""

import os
import sys
import uuid
import time
import threading
from typing import Any, Dict, Optional, Literal, List
import importlib.util
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np

import os, sys
os.environ.setdefault("DISABLE_TRITON", "1")  # 给你自己作标记用，可选
# 清理之前的脏状态
sys.modules.pop("triton", None)
# 告诉 transformers 自己判定“不可用”
try:
    from transformers.utils import import_utils as _iu
    _iu.is_triton_available = lambda: False
except Exception:
    pass

# os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# 若你的环境会因 torchao / triton 冲突报错，可保留以下两行禁用
# if importlib.util.find_spec("torchao") is not None:
#     print("[INFO] 'torchao' detected, disabling it to avoid Triton import errors...")
#     sys.modules["torchao"] = None
if importlib.util.find_spec("triton") is not None:
    print("[INFO] Disabling triton module...")
    sys.modules["triton"] = None

import argparse
import torch
from torch.nn.functional import normalize
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# === 业务相关：请确保以下模块在你的工程里可导入 ===
# from collect_grad_reps import collect_grads
from get_training_dataset import get_training_dataset
from get_validation_dataset import get_dataloader

import json
import os
from hashlib import md5
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType


def prepare_batch(batch, device=torch.device("cuda:0")):
    """ Move the batch to the device. """
    for key in batch:
        batch[key] = batch[key].to(device)


def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    """ 
    Retrieve the highest index for which the data (either representation or gradients) has been stored. 

    Args:
        output_dir (str): The output directory.
        prefix (str, optional): The prefix of the files, [reps | grads]. Defaults to "reps".

    Returns:
        int: The maximum representation index, or -1 if no index is found.
    """

    files = [file for file in os.listdir(
        output_dir) if file.startswith(prefix)]
    index = [int(file.split(".")[0].split("-")[1])
             for file in files]  # e.g., output_dir/reps-100.pt
    return max(index) if len(index) > 0 else -1


def get_output(model,
               weights: Iterable[Tensor],
               buffers: Iterable[Tensor],
               input_ids=None,
               attention_mask=None,
               labels=None,
               ) -> Tensor:
    logits = model(weights, buffers, *(input_ids.unsqueeze(0),
                    attention_mask.unsqueeze(0))).logits
    labels = labels.unsqueeze(0)
    loss_fct = F.cross_entropy
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
    return loss


def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        # print("Using BasicProjector")
    return projector


def obtain_gradients(model, batch):
    """ obtain gradients. """
    loss = model(**batch).loss
    loss.backward()
    vectorized_grads = torch.cat(
        [p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    return vectorized_grads


def obtain_sign_gradients(model, batch):
    """ obtain gradients with sign. """
    loss = model(**batch).loss
    loss.backward()

    # Instead of concatenating the gradients, concatenate their signs
    vectorized_grad_signs = torch.cat(
        [torch.sign(p.grad).view(-1) for p in model.parameters() if p.grad is not None])

    return vectorized_grad_signs


def obtain_gradients_with_adam(model, batch, avg, avg_sq):
    """ obtain gradients with adam optimizer states. """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    loss = model(**batch).loss
    loss.backward()

    vectorized_grads = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])

    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

    return vectorized_grads


def prepare_optimizer_state(model, optimizer_state, device):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                       for n in names])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq


import torch
from typing import List, Optional, Dict

# @torch.no_grad()
def collect_grads(
    dataloader,
    model,
    proj_dim: List[int] = [8192],
    adam_optimizer_state: Optional[dict] = None,
    gradient_type: str = "adam",
    max_samples: Optional[int] = None,
    project_interval: int = 16   # 每多少个 batch 统一投影一次
) -> Dict[int, torch.Tensor]:
    """
    按批量聚合的方式收集梯度，并进行随机投影
    """
    model.eval()

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Adam 状态提前准备好
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    # 获取有效 grad_dim（只对需要梯度的参数）
    grad_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 创建 projector
    projector_class = get_trak_projector(device)
    projectors = [
        projector_class(
            grad_dim=grad_dim,
            proj_dim=dim,
            seed=0,
            proj_type=ProjectionType.rademacher,
            device=device,
            dtype=dtype,
            block_size=128,
        )
        for dim in proj_dim
    ]

    # 缓存两个字典
    full_grads = []                     # 存多个 batch 的梯度向量
    projected_grads = {dim: [] for dim in proj_dim}

    sample_count = 0

    for batch in tqdm(dataloader):
        prepare_batch(batch)  

        # ============ normal 或 adam 获取梯度 ============
        if gradient_type == "adam":
            grads = obtain_gradients_with_adam(model, batch, m, v)
        elif gradient_type == "sign":
            grads = obtain_sign_gradients(model, batch)
        else:
            grads = obtain_gradients(model, batch)

        model.zero_grad()

        full_grads.append(grads)
        sample_count += 1

        # 满 project_interval 一次批量投影
        if len(full_grads) >= project_interval:
            stacked = torch.stack(full_grads).to(torch.float16)
            for i, proj in enumerate(projectors):
                proj_out = proj.project(stacked, model_id=0)
                projected_grads[proj_dim[i]].append(proj_out.cpu())
            full_grads.clear()

        if max_samples is not None and sample_count >= max_samples:
            break

    # 如果还有不足 project_interval 的剩余梯度，最后也投影
    if len(full_grads) > 0:
        stacked = torch.stack(full_grads).to(torch.float16)
        for i, proj in enumerate(projectors):
            proj_out = proj.project(stacked, model_id=0)
            projected_grads[proj_dim[i]].append(proj_out.cpu())
        full_grads.clear()

    # concat 每个 proj_dim 的所有结果
    for dim in proj_dim:
        if len(projected_grads[dim]) > 0:
            projected_grads[dim] = torch.cat(projected_grads[dim], dim=0)
        else:
            projected_grads[dim] = torch.empty(0)

    return projected_grads

# ------------------ 配置（可用环境变量覆盖） ------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8006"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# 负责排队/调度的线程池大小（CPU 侧并发）：
MAX_WORKERS = int(os.getenv("MAX_WORKERS", str(os.cpu_count() or 8)))
# 真正占 GPU 的并发度（强烈建议 1~2）：
PER_DEVICE_CONCURRENCY = int(os.getenv("PER_DEVICE_CONCURRENCY", "1"))
# 队列允许的最大排队任务数（超过则 429）：
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "256"))

DEVICE = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

# ------------------ CLI 参数 ------------------
parser = argparse.ArgumentParser(description="Gradient Service for Qwen2.5")
parser.add_argument("--model-path", type=str, required=True,
                    help="本地模型目录或HF模型ID（支持LoRA适配器路径）")
parser.add_argument("--learning-rate", type=float, default=1e-5,
                    help="学习率（如优化/继续训练需要）")
parser.add_argument("--optimizer-ckpt", type=str, default=None,
                    help="（可选）Adam优化器状态文件路径，如未给则默认尝试 <model_path>/optimizer.bin")
args, _ = parser.parse_known_args()

MODEL_PATH = args.model_path
OPTIM_CKPT = args.optimizer_ckpt or os.path.join(MODEL_PATH, "optimizer.bin")

# ------------------ 模型加载 ------------------
def load_model(model_name_or_path: str, torch_dtype: Any = torch.bfloat16) -> Any:
    """
    兼容 LoRA/PEFT 与原始基座模型的加载。
    """
    is_peft = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    if is_peft:
        # 加载PEFT配置，先载入基座，再套LoRA权重（device_map="auto"）
        config = LoraConfig.from_pretrained(model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, torch_dtype=torch_dtype, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_name_or_path,  device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map="auto"
        )

    # 仅对 LoRA 参数开启梯度（若后续训练/算梯度需要）
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
    return model

print(f"[BOOT] Loading model from: {MODEL_PATH}")
try:
    dtype = torch.float16  # 你也可以换成 bfloat16
    model = load_model(MODEL_PATH, dtype)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


print("[BOOT] Loading tokenizer ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer from {MODEL_PATH}: {e}")

adam_optimizer_state = None
if OPTIM_CKPT and os.path.exists(OPTIM_CKPT):
    print(f"[BOOT] Loading optimizer state from: {OPTIM_CKPT}")
    adam_optimizer_state = torch.load(OPTIM_CKPT, map_location="cpu").get("state", None)
else:
    print("[BOOT] Optimizer state not found or not provided; continuing without it.")

device = next(model.parameters()).device
m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

import glob
from functools import lru_cache

DIM = 8192
CKPTS = [4]
TARGET_TASKS = ["medqa"]
N_SUBTASKS = {"mmlu": 2, "bbh": 27, "tydiqa": 9, "medqa": 1}
VALIDATION_GRADS = {}  # {task: [tensor_per_ckpt]}
CHECKPOINT_WEIGHTS = [1.0]  # 与 CKPTS 对齐
CHECKPOINT_WEIGHTS = [w / sum(CHECKPOINT_WEIGHTS) for w in CHECKPOINT_WEIGHTS]

def _load_tensor(path):
    t = torch.load(path, map_location="cpu")
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    # 预归一化 + dtype 压到半精（VRAM 紧张就用 float16）
    return torch.nn.functional.normalize(
        t.to(torch.float16 if DEVICE.type == "cuda" else torch.float32), dim=-1
    )

def preload_validation_grads():
    base = os.getenv(
        "VALIDATION_GRAD_DIR",
        "<YOUR_GRAD_DIR>/{}-ckpt{}-sgd/dim{}"
    )
    for task in TARGET_TASKS:
        VALIDATION_GRADS[task] = []
        for ckpt in CKPTS:
            vpath = base.format(task, ckpt, DIM)
            if os.path.isdir(vpath):
                vpath = os.path.join(vpath, "all_orig.pt")
            vt = _load_tensor(vpath)
            vt = vt.to(DEVICE, non_blocking=True)
            VALIDATION_GRADS[task].append(vt)

preload_validation_grads()

# ------------------ 请求/响应结构 ------------------

class GradRequest(BaseModel):
    inputs: List[Dict[str, Any]]  # 允许传入list[dict]，每个dict是一条样本
    train_step: bool = False

class AsyncSubmitResp(BaseModel):
    job_id: str

class AsyncQueryResp(BaseModel):
    status: Literal["PENDING", "RUNNING", "DONE", "ERROR"]
    result: Optional[Dict] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

# ------------------ 并发控制与任务队列 ------------------
grad_sema = threading.Semaphore(PER_DEVICE_CONCURRENCY)
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)
JOBS: Dict[str, Dict] = {}  # {job_id: {future, error, started_at, finished_at}}

def _queue_depth() -> int:
    # 统计未完成且未错误的任务数量
    return sum(1 for m in JOBS.values() if m["error"] is None and not m["future"].done())

# ------------------ 业务核心：计算梯度/影响分 ------------------

def compute_gradients(inputs) -> Dict:
    with grad_sema:
        dataset = Dataset.from_list(inputs)
        dataset = get_training_dataset(
            dataset,
            tokenizer=tokenizer,
            max_seq_length=8192,
            processing_num_workers=16
        )

        # Remove unused columns
        columns = deepcopy(dataset.column_names)
        for col in ["input_ids", "labels", "attention_mask"]:
            if col in columns:
                columns.remove(col)
        dataset = dataset.remove_columns(columns)

        dataloader = get_dataloader(dataset, tokenizer=tokenizer)

        # 1. Compute training gradients in memory
        training_proj_grads = collect_grads(
            dataloader,
            model,
            proj_dim=[8192],
            adam_optimizer_state=adam_optimizer_state,
            gradient_type="adam"
        )
        training_info = training_proj_grads[8192]
        training_info = normalize(training_info.float(), dim=-1).to(DEVICE)

        # 2. Load validation gradients from disk
        dim = 8192
        ckpts = [4]
        checkpoint_weights = [1.0]
        s = sum(checkpoint_weights)
        checkpoint_weights = [w / s for w in checkpoint_weights]
        target_task_names = ["medqa"]
        validation_gradient_path = os.getenv(
            "VALIDATION_GRAD_DIR",
            "<YOUR_GRAD_DIR>/{}-ckpt{}-sgd/dim{}"
        )
        N_SUBTASKS = {"mmlu": 2, "bbh": 27, "tydiqa": 9, "medqa": 1}

        influence_score_list = {}

        for task in target_task_names:
            influence_score = 0
            for i, ckpt in enumerate(ckpts):
                vpath = validation_gradient_path.format(task, ckpt, dim)
                if os.path.isdir(vpath):
                    vpath = os.path.join(vpath, "all_orig.pt")
                validation_info = torch.load(vpath)
                if not torch.is_tensor(validation_info):
                    validation_info = torch.tensor(validation_info)
                validation_info = validation_info.to(DEVICE).float()
                # validation_info = validation_info.to(DEVICE).half()

                influence_score += checkpoint_weights[i] * torch.matmul(
                    training_info, validation_info.transpose(0, 1)
                )

            # Reshape per subtask
            influence_score = influence_score.reshape(
                influence_score.shape[0], N_SUBTASKS[task], -1
            ).mean(-1).max(-1)[0]


            influence_score_list[task] = influence_score.detach().cpu().tolist()
            print(influence_score_list[task])
            print(len(influence_score_list[task]))

        return influence_score_list


# ------------------ FastAPI 应用与路由 ------------------
app = FastAPI(title="Gradient Service - Qwen2.5", version="1.0")

def _submit_job(req: GradRequest) -> str:
    if _queue_depth() >= MAX_QUEUE:
        raise HTTPException(status_code=429, detail=f"queue is full (>{MAX_QUEUE})")

    job_id = str(uuid.uuid4())
    fut: Future = EXECUTOR.submit(compute_gradients, req.inputs)

    JOBS[job_id] = {
        "future": fut,
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
    }

    def _cb(f: Future, jid=job_id):
        try:
            _ = f.result()
            JOBS[jid]["finished_at"] = time.time()
        except Exception as e:
            JOBS[jid]["finished_at"] = time.time()
            JOBS[jid]["error"] = repr(e)

    fut.add_done_callback(_cb)
    return job_id

@app.get("/healthz")
def healthz():
    return {"ok": True, "device": str(DEVICE)}

@app.get("/metrics")
def metrics():
    return {
        "queue_depth": _queue_depth(),
        "max_queue": MAX_QUEUE,
        "max_workers": MAX_WORKERS,
        "per_device_concurrency": PER_DEVICE_CONCURRENCY,
        "jobs_total": len(JOBS),
    }

@app.post("/gradients/sync")
def gradients_sync(req: GradRequest):
    try:
        return {"status": "OK", "result": compute_gradients(req.inputs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"compute failed: {e}")

@app.post("/gradients/async/submit", response_model=AsyncSubmitResp)
def gradients_async_submit(req: GradRequest):
    job_id = _submit_job(req)
    return AsyncSubmitResp(job_id=job_id)

@app.get("/gradients/async/{job_id}", response_model=AsyncQueryResp)
def gradients_async_result(job_id: str):
    meta = JOBS.get(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="job_id not found")
    fut: Future = meta["future"]
    if meta["error"] is not None:
        return AsyncQueryResp(status="ERROR", error=meta["error"],
                              started_at=meta["started_at"], finished_at=meta["finished_at"])
    if fut.running():
        return AsyncQueryResp(status="RUNNING", started_at=meta["started_at"])
    if not fut.done():
        return AsyncQueryResp(status="PENDING", started_at=meta["started_at"])
    try:
        res = fut.result()
        return AsyncQueryResp(status="DONE", result=res,
                              started_at=meta["started_at"], finished_at=meta["finished_at"])
    except Exception as e:
        return AsyncQueryResp(status="ERROR", error=repr(e),
                              started_at=meta["started_at"], finished_at=meta["finished_at"])

# 既可脚本直跑，也可被 `python -m uvicorn serve_gradient_api:app` 调起
if __name__ == "__main__":
    print(f"[BOOT] Serving on {HOST}:{PORT} (log_level={LOG_LEVEL})")
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL)

    # inputs = {"messages": [{"role": "system", "content": "you are a helpful assistant."}, {"role": "user", "content": "How did the interplay of internal political dynamics and external economic pressures shape African state-building processes from the nineteenth century to post-independence, and what examples illustrate the persistence of localized governance structures despite colonial interventions?"}, {"role": "assistant", "content": "The evolution of African state-building from the nineteenth century to post-independence was profoundly shaped by the tension between internal political dynamics and external economic pressures, with localized governance structures often persisting despite colonial attempts to impose centralized authority. One key example lies in the West African savannah, where precolonial states like the Sokoto Caliphate and the Asante Empire grappled with the challenge of integrating fragmented local communities (kafu). These miniature states emphasized local autonomy, forcing empire-builders to oscillate between military domination and pragmatic alliances. Colonial administrators, rather than erasing this pattern, adapted it through systems like indirect rule, which co-opted existing local elites to manage rural populations—a strategy that inadvertently reinforced the very segmentation it sought to control. This continuity highlights how colonial governance was not a clean break from precolonial realities but a negotiation with entrenched socio-political structures.\n\nA second example emerges from the demographic and ecological constraints that defined governance across underpopulated regions. The text underscores how states in the nineteenth century often collapsed when discontented groups migrated beyond the reach of authority, destabilizing polities. This phenomenon persisted into the colonial era, as seen in the Mau Mau uprising in Kenya (1950s) and rural-urban migrations that undermined colonial administrative boundaries. The failure of colonial regimes to resolve these tensions reflects deeper continuities in African political history: the difficulty of constructing supra-local identities in regions where ecological and demographic factors incentivized localized loyalty over centralized allegiance.\n\nCritically, external economic pressures—such as the Atlantic and trans-Saharan slave trades—prefigured colonial extractive economies but did not wholly dictate African political outcomes. For instance, the integration of African rulers into global trade networks often allowed them to consolidate power through control of people (e.g., polygamy as a status symbol) rather than land, a dynamic that persisted in modified forms under colonial cash-crop economies. Post-independence states inherited these structural legacies, struggling to reconcile centralized governance ideals with the enduring influence of lineage-based authority and local economic networks. Thus, while colonialism intensified external exploitation, it remained enmeshed in indigenous political frameworks, illustrating how African state-building has been a process of adaptation rather than imposition."}]}
    # g = compute_gradients(inputs)
    # inputs_1 = 100 * [{"messages": [{"role": "system", "content": "you are a helpful assistant."}, {"role": "user", "content": "How did the interplay of internal political dynamics and external economic pressures shape African state-building processes from the nineteenth century to post-independence, and what examples illustrate the persistence of localized governance structures despite colonial interventions?"}, {"role": "assistant", "content": "The evolution of African state-building from the nineteenth century to post-independence was profoundly shaped by the tension between internal political dynamics and external economic pressures, with localized governance structures often persisting despite colonial attempts to impose centralized authority. One key example lies in the West African savannah, where precolonial states like the Sokoto Caliphate and the Asante Empire grappled with the challenge of integrating fragmented local communities (kafu). These miniature states emphasized local autonomy, forcing empire-builders to oscillate between military domination and pragmatic alliances. Colonial administrators, rather than erasing this pattern, adapted it through systems like indirect rule, which co-opted existing local elites to manage rural populations—a strategy that inadvertently reinforced the very segmentation it sought to control. This continuity highlights how colonial governance was not a clean break from precolonial realities but a negotiation with entrenched socio-political structures.\n\nA second example emerges from the demographic and ecological constraints that defined governance across underpopulated regions. The text underscores how states in the nineteenth century often collapsed when discontented groups migrated beyond the reach of authority, destabilizing polities. This phenomenon persisted into the colonial era, as seen in the Mau Mau uprising in Kenya (1950s) and rural-urban migrations that undermined colonial administrative boundaries. The failure of colonial regimes to resolve these tensions reflects deeper continuities in African political history: the difficulty of constructing supra-local identities in regions where ecological and demographic factors incentivized localized loyalty over centralized allegiance.\n\nCritically, external economic pressures—such as the Atlantic and trans-Saharan slave trades—prefigured colonial extractive economies but did not wholly dictate African political outcomes. For instance, the integration of African rulers into global trade networks often allowed them to consolidate power through control of people (e.g., polygamy as a status symbol) rather than land, a dynamic that persisted in modified forms under colonial cash-crop economies. Post-independence states inherited these structural legacies, struggling to reconcile centralized governance ideals with the enduring influence of lineage-based authority and local economic networks. Thus, while colonialism intensified external exploitation, it remained enmeshed in indigenous political frameworks, illustrating how African state-building has been a process of adaptation rather than imposition."}]}] * 3

    # # # 第一次
    # start = time.perf_counter()
    # g = compute_gradients(inputs_1)
    # end = time.perf_counter()
    # print(f"[inputs_1] compute_gradients耗时: {end - start:.4f} 秒")


"""
python training/influence/optimizer_batch_med.py --model-path <YOUR_WARMUP_CHECKPOINT_PATH>
"""