import json
import os
from itertools import product
from pathlib import Path
from typing import Any, Literal, Sequence, Tuple

import safetensors.torch
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc


def load_state_dict(model_path: Path, module_path: str):
    if not not module_path:
        module_path += "."
        
    weight_paths = json.load(open(model_path / "model.safetensors.index.json", "r"))["weight_map"]
    per_safetensor_weights = {}

    for weight_name in weight_paths.keys():
        if not weight_name.startswith(module_path):
            continue
        per_safetensor_weights.setdefault(weight_paths[weight_name], []).append(weight_name)

    return {
        weight_name[len(module_path) :]: safetensor_state_dict[weight_name]
        for safetensor_file_path, weight_names in per_safetensor_weights.items()
        for safetensor_state_dict in [safetensors.torch.load_file(model_path / safetensor_file_path)]
        for weight_name in weight_names
    }


def compare_tensor_pcc(
    tt_output_torch: torch.Tensor, reference_output: torch.Tensor, pcc_required: float = 0.98, msg="", assert_mode=False
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
