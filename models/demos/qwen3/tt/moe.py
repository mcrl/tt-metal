import torch
from torch import nn
import ttnn
from pathlib import Path

from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig
from models.demos.qwen3.tt.ccl_1d import CCL1D
from models.demos.qwen3.utils.timer import profile_time
from models.demos.qwen3.utils.profiler import profile_trace, Profiler


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, intermediate_size: int, mesh_device: ttnn.Device):
        super().__init__()
        self.mesh_device = mesh_device
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        assert config.hidden_act == "silu"
        self.is_tt_setup = False

    def setup_tt(self):
        if self.is_tt_setup:
            return
        self.is_tt_setup = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.mul(self.act_fn(self.gate_proj(x)), self.up_proj(x)))


class Qwen3MoeSparseMoeBlock(nn.Module):

    @profile_trace("create-layer", level=3, args={"class": "Qwen3MoeSparseMoeBlock"})
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, mesh_device: ttnn.Device):
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size, mesh_device=mesh_device)
                for _ in range(self.num_experts)
            ]
        )

        self.layer_idx = layer_idx
        self.is_tt_setup = False

    @profile_trace("setup-tt", level=3, args={"class": "Qwen3MoeSparseMoeBlock"})
    def setup_tt(self):
        if self.is_tt_setup:
            return

        with Profiler().trace_with_timer("CCL", level=4):
            self.ccl = CCL1D(self.mesh_device)

        with Profiler().trace_with_timer("gate_weight_tt", level=4):
            self.gate_weight = ttnn.as_tensor(
                self.gate.weight.transpose(0, 1),
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=Path.home() / ".cache/weights" / f"decoder_{self.layer_idx}_moe_gate",
            )

        self.num_devices = self.mesh_device.get_num_devices()
        self.num_experts_per_device = self.num_experts // self.num_devices

        with Profiler().trace_with_timer("expert_mapping_tensors_tt", level=4):
            self.expert_mapping_tensors = ttnn.from_torch(
                torch.eye(self.num_devices, dtype=torch.int32)
                .repeat_interleave(self.num_experts_per_device, dim=0)
                .unsqueeze(0)
                .unsqueeze(0),
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                dtype=ttnn.uint16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        gate_proj = []
        up_proj = []
        down_proj = []

        with Profiler().trace_with_timer("proj prepare", level=4):
            for expert_idx in range(self.num_experts):
                gate_proj.append(self.experts[expert_idx].gate_proj.weight)
                up_proj.append(self.experts[expert_idx].up_proj.weight)
                down_proj.append(self.experts[expert_idx].down_proj.weight)

            gate_proj = torch.stack(gate_proj, dim=0).permute(0, 2, 1).unsqueeze(0)
            up_proj = torch.stack(up_proj, dim=0).permute(0, 2, 1).unsqueeze(0)
            down_proj = torch.stack(down_proj, dim=0).permute(0, 2, 1).unsqueeze(0)

        with Profiler().trace_with_timer("proj upload", level=4):
            self.gate_proj = ttnn.as_tensor(
                gate_proj,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=Path.home() / ".cache/weights" / f"gate_proj_{self.layer_idx}",
            )

            self.up_proj = ttnn.as_tensor(
                up_proj,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=Path.home() / ".cache/weights" / f"up_proj_{self.layer_idx}",
            )

            self.down_proj = ttnn.as_tensor(
                down_proj,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=Path.home() / ".cache/weights" / f"down_proj_{self.layer_idx}",
            )
        self.is_tt_setup = True

    CACHED_KEY = None
    CACHED_TENSORS = [None, None, None]

    @classmethod
    def moe_tensor_caches(cls, batch_size, sequence_length, hidden_dim, top_k, mesh_device):
        key = (batch_size, sequence_length, hidden_dim, top_k)

        if key == cls.CACHED_KEY:
            for i in range(len(cls.CACHED_TENSORS)):
                ttnn.full_like(cls.CACHED_TENSORS[i], 0, optional_tensor=cls.CACHED_TENSORS[i])
            return cls.CACHED_TENSORS

        cls.CACHED_KEY = key
        cls.CACHED_TENSORS[0] = ttnn.zeros(
            (1, batch_size, sequence_length, hidden_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        cls.CACHED_TENSORS[1] = ttnn.zeros(
            (1, batch_size, sequence_length, top_k),
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        cls.CACHED_TENSORS[2] = ttnn.zeros(
            (top_k, batch_size, sequence_length, hidden_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return cls.moe_tensor_caches(batch_size, sequence_length, hidden_dim, top_k, mesh_device)

    @profile_trace("Qwen3MoeSparseMoeBlock", level=3)
    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        with Profiler().trace_with_timer("reshape", level=4):
            hidden_states = ttnn.reshape(hidden_states, (-1, hidden_dim), memory_config=ttnn.L1_MEMORY_CONFIG)

        with Profiler().trace_with_timer("moe-router", level=4):
            router_logits = ttnn.linear(
                hidden_states, self.gate_weight, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )

        with Profiler().trace_with_timer("softmax-topk-div", level=4):
            routing_weights = ttnn.softmax(router_logits, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
            routing_weights, selected_experts = ttnn.topk(
                routing_weights, self.top_k, dim=1, largest=True, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            if self.norm_topk_prob:
                routing_weights = ttnn.div(
                    routing_weights,
                    ttnn.sum(routing_weights, dim=1, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG),
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )

        with Profiler().trace_with_timer("to-layout", level=4):
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
            hidden_states = ttnn.reshape(
                hidden_states, (batch_size, 1, sequence_length, hidden_dim), memory_config=ttnn.L1_MEMORY_CONFIG
            )

            selected_experts = ttnn.to_layout(
                selected_experts, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            selected_experts = ttnn.reshape(
                selected_experts, (batch_size, 1, sequence_length, self.top_k), memory_config=ttnn.L1_MEMORY_CONFIG
            )

        with Profiler().trace_with_timer("prepare-all-to-all", level=4):
            (
                all_to_all_dispatch_output_tensors,
                all_to_all_dispatch_metadata_tensors,
                all_to_all_combine_output_tensors,
            ) = self.moe_tensor_caches(batch_size, sequence_length, hidden_dim, self.top_k, self.mesh_device)

        with Profiler().trace_with_timer("all-to-all-dispatch", level=4):
            ttnn.all_to_all_dispatch(
                hidden_states,
                selected_experts,
                self.expert_mapping_tensors,
                output_tensors=[all_to_all_dispatch_output_tensors, all_to_all_dispatch_metadata_tensors],
                cluster_axis=0,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                num_links=1,
                topology=ttnn.Topology.Linear,
                global_semaphore=self.ccl.get_semaphore(0),
                init_semaphore=self.ccl.get_semaphore(0),
            )

            post_all_to_all_dispatch_output = ttnn.reshape(
                all_to_all_dispatch_output_tensors,
                shape=(1, 1, batch_size * sequence_length, hidden_dim),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            post_all_to_all_dispatch_output = ttnn.repeat(
                post_all_to_all_dispatch_output,
                ttnn.Shape((1, self.num_experts_per_device, 1, 1)),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            post_all_to_all_dispatch_output = ttnn.to_layout(
                post_all_to_all_dispatch_output, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
            )

        with Profiler().trace_with_timer("expert-compute", level=4):
            gate_proj_output = ttnn.matmul(
                post_all_to_all_dispatch_output, self.gate_proj, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            up_proj_output = ttnn.matmul(
                post_all_to_all_dispatch_output, self.up_proj, memory_config=ttnn.L1_MEMORY_CONFIG
            )

            glu_output = ttnn.mul(
                gate_proj_output,
                up_proj_output,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            )

            experts_output = ttnn.matmul(glu_output, self.down_proj, memory_config=ttnn.L1_MEMORY_CONFIG)

            experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
            experts_output = ttnn.reshape(
                experts_output,
                (self.num_experts_per_device, batch_size, sequence_length, hidden_dim),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        with Profiler().trace_with_timer("all-to-all-combine", level=4):
            ttnn.all_to_all_combine(
                experts_output,
                self.expert_mapping_tensors,
                all_to_all_dispatch_metadata_tensors,
                optional_output_tensor=all_to_all_combine_output_tensors,
                axis=0,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                num_links=1,
                global_semaphore=self.ccl.get_semaphore(0),
                init_semaphore=self.ccl.get_semaphore(0),
            )
            ttnn.synchronize_device(self.mesh_device)

            post_combine_output_tensor = ttnn.reshape(
                all_to_all_combine_output_tensors,
                shape=(self.top_k, 1, batch_size * sequence_length, hidden_dim),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            # post_combine_output_tensor = ttnn.to_layout(
            #     post_combine_output_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
            # )

        with Profiler().trace_with_timer("to-layout", level=4):
            routing_weights_rm = ttnn.to_layout(
                routing_weights, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            routing_weights_rm = ttnn.repeat(
                routing_weights_rm, ttnn.Shape((hidden_dim, 1, 1, 1)), memory_config=ttnn.L1_MEMORY_CONFIG
            )
            routing_weights_rm = ttnn.permute(routing_weights_rm, (3, 1, 2, 0), memory_config=ttnn.L1_MEMORY_CONFIG)
            routing_weights = ttnn.to_layout(routing_weights_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        with Profiler().trace_with_timer("moe-post", level=4):
            post_combine_output_tensor = ttnn.mul(
                post_combine_output_tensor, routing_weights, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            post_combine_output_tensor = ttnn.sum(
                post_combine_output_tensor, dim=0, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG
            )

        with Profiler().trace_with_timer("all-reduce", level=4):
            post_combine_output_tensor = ttnn.reshape(
                post_combine_output_tensor,
                shape=(1, 1, batch_size * sequence_length * hidden_dim // 256, 256),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            post_combine_output_tensor_g = ttnn.experimental.all_reduce(
                post_combine_output_tensor,
                math_op=ttnn.ReduceType.Sum,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
            )
            ttnn.synchronize_device(self.mesh_device)

        with Profiler().trace_with_timer("reshape", level=4):
            final_hidden_states = ttnn.reshape(
                post_combine_output_tensor_g,
                (batch_size, sequence_length, hidden_dim),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        return final_hidden_states


__all__ = ["Qwen3MoeMLP", "Qwen3MoeSparseMoeBlock"]
