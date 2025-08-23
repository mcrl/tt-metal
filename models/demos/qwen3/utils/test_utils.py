# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import json
import os
from itertools import product
from pathlib import Path
from typing import Any, Literal, Sequence, Tuple

import safetensors.torch
import torch
from loguru import logger

from models.demos.qwen3.utils.abstract_module import AbstractModule
from models.demos.qwen3.utils.config_helpers import dequantize
from models.utility_functions import comp_pcc


def load_state_dict(model_path: Path, module_path: str):
    if not not module_path:
        module_path += "."  # So that the later matches include the separating dot

    weight_paths = json.load(open(model_path / "model.safetensors.index.json", "r"))["weight_map"]
    per_safetensor_weights = {}

    for weight_name in weight_paths.keys():
        if not weight_name.startswith(module_path):
            continue
        per_safetensor_weights.setdefault(weight_paths[weight_name], []).append(weight_name)

    return {
        weight_name[len(module_path):]: safetensor_state_dict[weight_name]
        for safetensor_file_path, weight_names in per_safetensor_weights.items()
        for safetensor_state_dict in [safetensors.torch.load_file(model_path / safetensor_file_path)]
        for weight_name in weight_names
    }


def add_inv_scale_to_state_dict(
    state_dict: dict[str, torch.Tensor],
    block_shape: Sequence[int],
    weight_names: list[str] = ["up_proj", "down_proj", "gate_proj"],
) -> dict[str, torch.Tensor]:
    """
    Quantizes specified weights in state_dict and adds inverse scale tensors.

    Args:
        state_dict: original model weights
        block_shape: shape of quantization blocks (e.g., [128, 128])
        weight_names: list of substrings to match in parameter names

    Returns:
        new state_dict with quantized weights and _scale_inv tensors
    """
    output_state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if weight_names and not any(name.endswith(weight_name + ".weight") for weight_name in weight_names):
            output_state_dict[name] = tensor
            continue

        dequant_scale = torch.randn(
            (
                *tensor.shape[: -len(block_shape)],
                *(
                    (tensor.shape[-len(block_shape) + idx] + block_dim - 1) // block_dim
                    for idx, block_dim in enumerate(block_shape)
                ),
            ),
            dtype=torch.float32,
        )

        tensor_quant = dequantize(tensor.to(torch.float8_e4m3fn), 1.0 / dequant_scale, block_shape)
        output_state_dict[name] = tensor_quant.to(torch.float8_e4m3fn)
        output_state_dict[name + "_scale_inv"] = dequant_scale

    return output_state_dict


SEQ_LEN_DIM_IDX = 2


def pad_or_trim_seq_len(tensor: torch.Tensor, mode: Literal["prefill", "decode"], seq_len: int) -> torch.Tensor:
    """Changes the tensor's sequence length to match the given seq_len, adding padding if necessary."""
    assert mode in ["prefill", "decode"], f"Unsupported mode: {mode}"

    tensor_seq_len = tensor.shape[SEQ_LEN_DIM_IDX]
    if tensor_seq_len == seq_len:
        return tensor.clone()

    padded_tensor_shape = list(tensor.shape)
    padded_tensor_shape[SEQ_LEN_DIM_IDX] = seq_len
    padded_tensor = torch.zeros(padded_tensor_shape, dtype=tensor.dtype, device=tensor.device)

    padded_tensor_ranges = tuple(
        slice(None) if idx != SEQ_LEN_DIM_IDX else slice(None, min(seq_len, tensor_seq_len))
        for idx in range(tensor.ndim)
    )
    padded_tensor[padded_tensor_ranges] = tensor[padded_tensor_ranges]

    return padded_tensor


def get_model_config(ModuleClass: type[AbstractModule], mode: Literal["prefill", "decode"], *args, **kwargs) -> Any:
    """Get the module config for the given mode and kwargs."""
    if mode == "prefill":
        return ModuleClass.prefill_model_config(*args, **kwargs)
    elif mode == "decode":
        return ModuleClass.decode_model_config(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'prefill' and 'decode'.")


def run_module_forward(ModuleClass: type[AbstractModule], mode: Literal["prefill", "decode"], *args, **kwargs) -> Any:
    """Run the module forward pass for the given mode and kwargs."""
    if mode == "prefill":
        return ModuleClass.forward_prefill(*args, **kwargs)
    elif mode == "decode":
        return ModuleClass.forward_decode(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'prefill' and 'decode'.")


def compare_tensor_pcc(
    tt_output_torch: torch.Tensor, reference_output: torch.Tensor, pcc_required: float = 0.98,
    msg="", assert_mode=False
) -> float:
    assert (
        tt_output_torch.ndim == reference_output.ndim
    ), f"Both model and reference outputs must have the same number of dimensions; got {tt_output_torch.ndim}D and {reference_output.ndim}D instead"
    for dim in range(tt_output_torch.ndim):
        assert (
            tt_output_torch.shape[dim] == reference_output.shape[dim]
        ), f"Model and reference output shape mismatch on dim {dim} ({tt_output_torch.shape[dim]} != {reference_output.shape[dim]})"

    passing, pcc = comp_pcc(tt_output_torch, reference_output, pcc_required)
    logger.info(f"[{msg}] Tensor shape: {tt_output_torch.shape}, PCC: {pcc}")
    if assert_mode:
        assert passing, f"Pearson Correlation Coefficient {pcc} is below required {pcc_required}."
    return pcc
