import torch
from torch import nn
import ttnn

from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig
from models.demos.qwen3.tt.ccl_1d import CCL1D
from models.demos.qwen3.tt.timer import profile_time, start_timer, stop_timer


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

    def setup_tt(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.mul(self.act_fn(self.gate_proj(x)), self.up_proj(x)))


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, mesh_device: ttnn.Device):
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size, mesh_device=mesh_device) for _ in range(self.num_experts)]
        )

        self.layer_idx = layer_idx

    def setup_tt(self):
        self.ccl = CCL1D(self.mesh_device)
        self.gate_weight_tt = ttnn.from_torch(
            self.gate.weight.transpose(0, 1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.gate_weight_tt = ttnn.to_layout(self.gate_weight_tt, ttnn.TILE_LAYOUT)

        self.num_devices = self.mesh_device.get_num_devices()
        self.num_experts_per_device = self.num_experts // self.num_devices
        self.expert_mapping_tensors_tt = ttnn.from_torch(
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

        for expert_idx in range(self.num_experts):
            gate_proj.append(self.experts[expert_idx].gate_proj.weight)
            up_proj.append(self.experts[expert_idx].up_proj.weight)
            down_proj.append(self.experts[expert_idx].down_proj.weight)

        gate_proj = torch.stack(gate_proj, dim=0).permute(0, 2, 1).unsqueeze(0)
        up_proj = torch.stack(up_proj, dim=0).permute(0, 2, 1).unsqueeze(0)
        down_proj = torch.stack(down_proj, dim=0).permute(0, 2, 1).unsqueeze(0)

        self.gate_proj_tt = ttnn.from_torch(gate_proj,
                                            device=self.mesh_device,
                                            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                                            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.gate_proj_tt = ttnn.to_layout(self.gate_proj_tt, ttnn.TILE_LAYOUT)

        self.up_proj_tt = ttnn.from_torch(up_proj,
                                          device=self.mesh_device,
                                          mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                                          dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.up_proj_tt = ttnn.to_layout(self.up_proj_tt, ttnn.TILE_LAYOUT)

        self.down_proj_tt = ttnn.from_torch(down_proj,
                                            device=self.mesh_device,
                                            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                                            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.down_proj_tt = ttnn.to_layout(self.down_proj_tt, ttnn.TILE_LAYOUT)

    @profile_time()
    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = ttnn.reshape(hidden_states, (-1, hidden_dim))

        start_timer("moe-router", device=self.mesh_device)
        router_logits_tt = ttnn.linear(hidden_states, self.gate_weight_tt, dtype=ttnn.bfloat16)
        routing_weights_tt = ttnn.softmax(router_logits_tt, dim=1)
        routing_weights_tt, selected_experts = ttnn.topk(routing_weights_tt, self.top_k, dim=1, largest=True)

        if self.norm_topk_prob:
            routing_weights_tt = ttnn.div(routing_weights_tt,
                                          ttnn.sum(routing_weights_tt, dim=1, keepdim=True))

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, (batch_size, sequence_length, 1, hidden_dim))

        selected_experts = ttnn.to_layout(selected_experts, ttnn.ROW_MAJOR_LAYOUT)
        selected_experts = ttnn.reshape(selected_experts, (batch_size, sequence_length, 1, self.top_k))
        stop_timer("moe-router", device=self.mesh_device)

        start_timer("moe-prepare-tensors", device=self.mesh_device)
        all_to_all_dispatch_output_tensors = ttnn.from_torch(
            torch.zeros([1, batch_size, sequence_length, hidden_dim]),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        all_to_all_dispatch_metadata_tensors = ttnn.from_torch(
            torch.zeros([1, batch_size, sequence_length, self.top_k], dtype=torch.int32),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        all_to_all_combine_output_tensors = ttnn.from_torch(
            torch.zeros([self.top_k, batch_size, sequence_length, hidden_dim]),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        stop_timer("moe-prepare-tensors", device=self.mesh_device)

        start_timer("moe-all-to-all-dispatch", device=self.mesh_device)
        ttnn.all_to_all_dispatch(
            hidden_states,
            selected_experts,
            self.expert_mapping_tensors_tt,
            output_tensors=[
                all_to_all_dispatch_output_tensors,
                all_to_all_dispatch_metadata_tensors
            ],
            cluster_axis=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_links=1,
            topology=ttnn.Topology.Linear,
            global_semaphore=self.ccl.get_semaphore(0),
            init_semaphore=self.ccl.get_semaphore(0)
        )
        post_all_to_all_dispatch_output = ttnn.reshape(
            all_to_all_dispatch_output_tensors, shape=(1, 1, batch_size * sequence_length, hidden_dim)
        )
        post_all_to_all_dispatch_output = ttnn.repeat(post_all_to_all_dispatch_output, ttnn.Shape((1, self.num_experts_per_device, 1, 1)))
        post_all_to_all_dispatch_output = ttnn.to_layout(post_all_to_all_dispatch_output, ttnn.TILE_LAYOUT)
        stop_timer("moe-all-to-all-dispatch", device=self.mesh_device)

        start_timer("moe-expert-compute", device=self.mesh_device)
        gate_proj_output_tt = ttnn.matmul(post_all_to_all_dispatch_output, self.gate_proj_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up_proj_output_tt = ttnn.matmul(post_all_to_all_dispatch_output, self.up_proj_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        glu_output_tt = ttnn.mul(gate_proj_output_tt, up_proj_output_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                 input_tensor_a_activations=[ttnn.UnaryOpType.SILU])

        experts_output_tt = ttnn.matmul(glu_output_tt, self.down_proj_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        experts_output_tt = ttnn.to_layout(experts_output_tt, ttnn.ROW_MAJOR_LAYOUT)
        experts_output_tt = ttnn.reshape(experts_output_tt, (self.num_experts_per_device, batch_size, sequence_length, hidden_dim))
        stop_timer("moe-expert-compute", device=self.mesh_device)

        start_timer("moe-all-to-all-combine", device=self.mesh_device)
        ttnn.all_to_all_combine(
            experts_output_tt,
            self.expert_mapping_tensors_tt,
            all_to_all_dispatch_metadata_tensors,
            optional_output_tensor=all_to_all_combine_output_tensors,
            axis=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_links=1,
            global_semaphore=self.ccl.get_semaphore(0),
            init_semaphore=self.ccl.get_semaphore(0)
        )
        post_combine_output_tensor = ttnn.reshape(all_to_all_combine_output_tensors, shape=(self.top_k, 1, batch_size * sequence_length, hidden_dim))
        post_combine_output_tensor = ttnn.to_layout(post_combine_output_tensor, ttnn.TILE_LAYOUT)
        stop_timer("moe-all-to-all-combine", device=self.mesh_device)

        routing_weights_tt_rm = ttnn.to_layout(routing_weights_tt, ttnn.ROW_MAJOR_LAYOUT)
        routing_weights_tt_rm = ttnn.repeat(routing_weights_tt_rm, ttnn.Shape((hidden_dim, 1, 1, 1)))
        routing_weights_tt_rm = ttnn.permute(routing_weights_tt_rm, (3, 1, 2, 0))
        routing_weights_tt = ttnn.to_layout(routing_weights_tt_rm, ttnn.TILE_LAYOUT)

        start_timer("moe-post", device=self.mesh_device)
        post_combine_output_tensor = ttnn.mul(post_combine_output_tensor, routing_weights_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        post_combine_output_tensor = ttnn.sum(post_combine_output_tensor, dim=0, keepdim=True)
        stop_timer("moe-post", device=self.mesh_device)

        start_timer("moe-reduce-scatter", device=self.mesh_device)
        post_combine_output_tensor_rs = ttnn.reduce_scatter(
            post_combine_output_tensor,
            dim=3,
            math_op=ttnn.ReduceType.Sum,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear
        )
        ttnn.synchronize_device(self.mesh_device)
        stop_timer("moe-reduce-scatter", device=self.mesh_device)

        start_timer("moe-all-gather", device=self.mesh_device)
        post_combine_output_tensor_g = ttnn.all_gather(
            post_combine_output_tensor_rs,
            dim=3,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            mesh_device=self.mesh_device
        )
        ttnn.synchronize_device(self.mesh_device)
        stop_timer("moe-all-gather", device=self.mesh_device)

        final_hidden_states_tt = ttnn.reshape(post_combine_output_tensor_g, (batch_size, sequence_length, hidden_dim))

        return final_hidden_states_tt


__all__ = ["Qwen3MoeMLP", "Qwen3MoeSparseMoeBlock"]
