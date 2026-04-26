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


# def get_number_of_params(model):
#     """ Make sure that only lora parameters require gradients in peft models. """
#     if isinstance(model, PeftModel):
#         names = [n for n, p in model.named_parameters(
#         ) if p.requires_grad and "lora" not in n]
#         assert len(names) == 0
#     num_params = sum([p.numel()
#                      for p in model.parameters() if p.requires_grad])
#     print(f"Total number of parameters that require gradients: {num_params}")
#     return num_params


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


def collect_grads(dataloader,
                  model,
                  proj_dim: List[int] = [8192],
                  adam_optimizer_state: Optional[dict] = None):
    """
    Collects gradients from a batch using the model, returns projected gradients.

    Args:
        batch (dict): input batch (already numericalized, on CPU).
        model (torch.nn.Module): model from which gradients will be collected.
        proj_dim (List[int]): projection output dimensions.
        adam_optimizer_state (dict): Adam optimizer state dict.
    Returns:
        projected_grads (List[Tensor]): list of projected gradients, each shaped [proj_dim]
    """

    block_size = 128
    torch.random.manual_seed(0)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Adam avg / avg_sq
    m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    projector = get_trak_projector(device)
    # number_of_params = get_number_of_params(model)

    # Total number of parameters that require gradients for qwen2.5-7b: 80740352 
    projectors = []
    for dim in proj_dim:
        proj = projector(grad_dim=80740352,
                         proj_dim=dim,
                         seed=0,
                         proj_type=ProjectionType.rademacher,
                         device=device,
                         dtype=dtype,
                         block_size=block_size,)
        projectors.append(proj)

    batch = next(iter(dataloader))
    prepare_batch(batch)
    vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)

    # 现在每次只处理一条数据所以需要加一个维度
    if vectorized_grads.dim() == 1:
        vectorized_grads = vectorized_grads.unsqueeze(0)

    model.zero_grad()

    # if vectorized_grads.dim() == 1:
    #     vectorized_grads = vectorized_grads.unsqueeze(0)

    projected_grads = [proj.project(vectorized_grads, model_id=0) for proj in projectors]

    print("Finished")

    return projected_grads
