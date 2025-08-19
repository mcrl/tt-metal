# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.qwen3.utils.abstract_module import AbstractModule
from models.demos.qwen3.utils.config_dataclass import FromWeightConfig, LinearConfig, MeshDeviceStub, RMSNormConfig
from models.demos.qwen3.utils.config_helpers import COMPUTE_KERNEL_CONFIG_LOFI, save_and_get_path
from models.demos.qwen3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class Qwen3MoeAttentionTT(AbstractModule):
    """Simplified TT implementation of Qwen3MoeAttention (prefill path).

    Notes:
    - Implements separate q/k/v/o projections and per-head RMSNorm on q and k.
    - Uses standard SDPA without rotary for minimal functional parity on seq_len=1 tests.
    - Decode path is left unimplemented for now.
    """

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        assert (
            len(state_dicts) == 1 and state_dicts[0] is not None
        ), f"Attention expects exactly one non-padding state dict, got {len(state_dicts)}"
        (state_dict,) = state_dicts  # type: ignore

        def save_linear_weight(weight: torch.Tensor, name: str):
            tt_weight = ttnn.as_tensor(
                weight.T,  # linear expects input_tensor_b shape [in, out]
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            return save_and_get_path(output_path / f"{name}.input_tensor_b", tt_weight)

        def save_vector_weight(weight: torch.Tensor, name: str):
            tt_weight = ttnn.as_tensor(
                weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            return save_and_get_path(output_path / f"{name}.weight", tt_weight)

        return {
            "wq": {"input_tensor_b": save_linear_weight(state_dict["q_proj.weight"], "wq")},
            "wk": {"input_tensor_b": save_linear_weight(state_dict["k_proj.weight"], "wk")},
            "wv": {"input_tensor_b": save_linear_weight(state_dict["v_proj.weight"], "wv")},
            "wo": {"input_tensor_b": save_linear_weight(state_dict["o_proj.weight"], "wo")},
            "q_norm": {"weight": save_vector_weight(state_dict["q_norm.weight"], "q_norm")},
            "k_norm": {"weight": save_vector_weight(state_dict["k_norm.weight"], "k_norm")},
        }

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        return {
            "wq": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "wk": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "wv": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "wo": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "q_norm": RMSNormConfig(
                epsilon=hf_config.rms_norm_eps,
                weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "k_norm": RMSNormConfig(
                epsilon=hf_config.rms_norm_eps,
                weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "num_heads": hf_config.num_attention_heads,
            "head_dim": hf_config.head_dim,
            "input_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "output_memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelDecodeConfig:
        # Not implemented for now
        return cls.prefill_model_config(hf_config, mesh_device)

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        assert x.memory_config() == cfg["input_memory_config"], f"{x.memory_config()} != {cfg['input_memory_config']}"

        # Projections
        q = ttnn.linear(x, **cfg["wq"])  # [1, 1, S, H]
        k = ttnn.linear(x, **cfg["wk"])  # [1, 1, S, H]
        v = ttnn.linear(x, **cfg["wv"])  # [1, 1, S, H]

        # Reshape to heads: [1, heads, S, head_dim]
        _, _, seq_len, _ = q.shape
        num_heads = cfg["num_heads"]
        head_dim = cfg["head_dim"]
        qh = ttnn.reshape(q, [1, num_heads, seq_len, head_dim])
        kh = ttnn.reshape(k, [1, num_heads, seq_len, head_dim])
        vh = ttnn.reshape(v, [1, num_heads, seq_len, head_dim])
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        # Note: keep v until after SDPA

        # Per-head RMSNorm for q and k
        qh = ttnn.rms_norm(qh, **cfg["q_norm"])  # weight is shared across heads
        kh = ttnn.rms_norm(kh, **cfg["k_norm"])  # weight is shared across heads

        # SDPA (causal)
        attn = ttnn.transformer.scaled_dot_product_attention(qh, kh, vh, is_causal=True)  # [1, heads, S, head_dim]
        ttnn.deallocate(qh)
        ttnn.deallocate(kh)
        ttnn.deallocate(vh)

        # Concat heads and output projection
        attn_cat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,S,H]
        ttnn.deallocate(attn)
        out = ttnn.linear(attn_cat, **cfg["wo"])  # [1,1,S,H]
        ttnn.deallocate(attn_cat)

        assert out.memory_config() == cfg["output_memory_config"]
        return out

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        raise NotImplementedError("Decode not implemented for Qwen3MoeAttentionTT")
