import torch
from torch import nn
import ttnn

from models.demos.qwen3.common.configuration_qwen3_moe import Qwen3MoeConfig
from models.demos.qwen3.tt.ccl_1d import CCL1D
from models.tt_transformers.tt.ccl import TT_CCL

from models.demos.qwen3.utils.profiler import profile_trace, Profiler
from models.demos.qwen3.tt.model_cache import ttnn_model_cache_path
from models.demos.qwen3.common.configuration_qwen3_moe import InferenceMode


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
            self.ccl = TT_CCL(self.mesh_device)

        with Profiler().trace_with_timer("gate_weight_tt", level=4):
            self.gate_weight = ttnn.as_tensor(
                self.gate.weight.transpose(0, 1),
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=ttnn_model_cache_path(f"decoder_{self.layer_idx}_moe_gate"),
            )

        self.num_devices = self.mesh_device.get_num_devices()
        self.num_experts_per_device = self.num_experts // self.num_devices

        with Profiler().trace_with_timer("expert_mapping_tensors_tt", level=4):
            self.device_expert_mapping_legacy = ttnn.from_torch(
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
                cache_file_name=ttnn_model_cache_path(f"gate_proj_{self.layer_idx}"),
            )

            self.up_proj = ttnn.as_tensor(
                up_proj,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=ttnn_model_cache_path(f"up_proj_{self.layer_idx}"),
            )

            self.down_proj = ttnn.as_tensor(
                down_proj,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=ttnn_model_cache_path(f"down_proj_{self.layer_idx}"),
            )
        self.is_tt_setup = True

    @profile_trace("Qwen3MoeSparseMoeBlock", level=3)
    def forward_v1(self, hidden_states: ttnn.Tensor, mode: InferenceMode = InferenceMode.PREFILL) -> ttnn.Tensor:
        if mode == InferenceMode.PREFILL:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        elif mode == InferenceMode.DECODE:
            _, sequence_length, batch_size, hidden_dim = hidden_states.shape
            mem_cfg = ttnn.L1_MEMORY_CONFIG

        with Profiler().trace_with_timer("reshape", level=4):
            hidden_states = ttnn.reshape(hidden_states, (-1, hidden_dim), memory_config=mem_cfg)

        with Profiler().trace_with_timer("moe-router", level=4):
            router_logits = ttnn.linear(hidden_states, self.gate_weight, dtype=ttnn.bfloat16, memory_config=mem_cfg)

        with Profiler().trace_with_timer("softmax-topk-div", level=4):
            routing_weights = ttnn.softmax(router_logits, dim=1, memory_config=mem_cfg)
            routing_weights, selected_experts = ttnn.topk(
                routing_weights, self.top_k, dim=1, largest=True, memory_config=mem_cfg
            )
            if self.norm_topk_prob:
                routing_weights = ttnn.div(
                    routing_weights,
                    ttnn.sum(routing_weights, dim=1, keepdim=True, memory_config=mem_cfg),
                    memory_config=mem_cfg,
                )

        with Profiler().trace_with_timer("to-layout", level=4):
            selected_experts = ttnn.to_layout(
                selected_experts, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            selected_experts = ttnn.reshape(
                selected_experts, (1, batch_size, sequence_length, self.top_k), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        # with Profiler().trace_with_timer("prepare-all-to-all", level=4):
        #     all_to_all_combine_output_tensors = ttnn.zeros(
        #         (self.top_k, batch_size, sequence_length, hidden_dim),
        #         dtype=ttnn.bfloat16,
        #         layout=ttnn.ROW_MAJOR_LAYOUT,
        #         device=self.mesh_device,
        #         memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     )

        with Profiler().trace_with_timer("repeat-hidden", level=4):
            hidden_states = ttnn.reshape(
                hidden_states,
                shape=(1, 1, batch_size * sequence_length, hidden_dim),
                memory_config=mem_cfg,
            )
            hidden_states = ttnn.repeat(
                hidden_states,
                ttnn.Shape((1, self.num_experts_per_device, 1, 1)),
                memory_config=mem_cfg,
            )
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=mem_cfg)

        with Profiler().trace_with_timer("expert-compute", level=4):
            gate_proj_output = ttnn.matmul(hidden_states, self.gate_proj, memory_config=mem_cfg)
            up_proj_output = ttnn.matmul(hidden_states, self.up_proj, memory_config=mem_cfg)
            ttnn.deallocate(hidden_states)

            glu_output = ttnn.mul(
                gate_proj_output,
                up_proj_output,
                memory_config=mem_cfg,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            )
            ttnn.deallocate(gate_proj_output)
            ttnn.deallocate(up_proj_output)

            experts_output = ttnn.matmul(glu_output, self.down_proj, memory_config=mem_cfg)
            ttnn.deallocate(glu_output)

            experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem_cfg)
            experts_output = ttnn.reshape(
                experts_output,
                (self.num_experts_per_device, batch_size, sequence_length, hidden_dim),
                memory_config=mem_cfg,
            )

        with Profiler().trace_with_timer("all-to-all-combine", level=4):
            all_to_all_combine_output_tensors = ttnn.all_to_all_combine(
                experts_output,
                self.device_expert_mapping_legacy,
                selected_experts,
                axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                num_links=1,
                global_semaphore=self.ccl.get_semaphore(0),
                init_semaphore=self.ccl.get_semaphore(0),
            )

            all_to_all_combine_output_tensors = ttnn.experimental.all_gather_async(
                all_to_all_combine_output_tensors,
                dim=1,
                cluster_axis=1,
                mesh_device=self.mesh_device,
                topology=ttnn.Topology.Linear,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                num_links=1,
                multi_device_global_semaphore=self.ccl.get_semaphore(0),
                barrier_semaphore=self.ccl.get_semaphore(1),
            )
            ttnn.synchronize_device(self.mesh_device)
            combined_output = ttnn.reshape(
                all_to_all_combine_output_tensors,
                shape=(self.top_k, 1, batch_size * sequence_length, hidden_dim),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        with Profiler().trace_with_timer("to-layout", level=4):
            routing_weights = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem_cfg)
            routing_weights = ttnn.repeat(routing_weights, ttnn.Shape((hidden_dim, 1, 1, 1)), memory_config=mem_cfg)
            routing_weights = ttnn.permute(routing_weights, (3, 1, 2, 0), memory_config=mem_cfg)
            routing_weights = ttnn.to_layout(routing_weights, ttnn.TILE_LAYOUT, memory_config=mem_cfg)

        with Profiler().trace_with_timer("moe-post", level=4):
            combined_output_2 = ttnn.mul(combined_output, routing_weights, memory_config=mem_cfg)
            ttnn.deallocate(routing_weights)
            ttnn.deallocate(combined_output)

            combined_output_3 = ttnn.sum(combined_output_2, dim=0, keepdim=True, memory_config=mem_cfg)
            ttnn.deallocate(combined_output_2)

        with Profiler().trace_with_timer("reshape", level=4):
            if mode == InferenceMode.PREFILL:
                final_hidden_states = ttnn.reshape(
                    combined_output_3,
                    (batch_size, sequence_length, hidden_dim),
                    memory_config=mem_cfg,
                )
            elif mode == InferenceMode.DECODE:
                final_hidden_states = ttnn.reshape(
                    combined_output_3,
                    (1, 1, batch_size, hidden_dim),
                    memory_config=mem_cfg,
                )

        return final_hidden_states

    @profile_trace("Qwen3MoeSparseMoeBlock", level=3)
    def forward_v2(self, hidden_states: ttnn.Tensor, mode: InferenceMode = InferenceMode.PREFILL) -> ttnn.Tensor:
        if mode == InferenceMode.PREFILL:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        elif mode == InferenceMode.DECODE:
            _, sequence_length, batch_size, hidden_dim = hidden_states.shape
            mem_cfg = ttnn.L1_MEMORY_CONFIG

        with Profiler().trace_with_timer("reshape", level=4):
            hidden_states = ttnn.reshape(hidden_states, (-1, hidden_dim), memory_config=mem_cfg)

        with Profiler().trace_with_timer("moe-router", level=4):
            router_logits = ttnn.linear(hidden_states, self.gate_weight, dtype=ttnn.bfloat16, memory_config=mem_cfg)

        with Profiler().trace_with_timer("softmax-topk-div", level=4):
            routing_weights = ttnn.softmax(router_logits, dim=1, memory_config=mem_cfg)
            routing_weights, selected_experts = ttnn.topk(
                routing_weights, self.top_k, dim=1, largest=True, memory_config=mem_cfg
            )
            if self.norm_topk_prob:
                routing_weights = ttnn.div(
                    routing_weights,
                    ttnn.sum(routing_weights, dim=1, keepdim=True, memory_config=mem_cfg),
                    memory_config=mem_cfg,
                )

        # Convert selected_experts to UINT32 and ROW_MAJOR for routing tensor preparation
        with Profiler().trace_with_timer("typecast-selected-experts", level=4):
            # typecast requires TILE layout
            selected_experts = ttnn.typecast(selected_experts, ttnn.uint32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # Convert to ROW_MAJOR after typecast
            selected_experts = ttnn.to_layout(
                selected_experts, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        # Convert routing_weights to ROW_MAJOR
        with Profiler().trace_with_timer("to-layout-routing-weights", level=4):
            routing_weights = ttnn.to_layout(
                routing_weights, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        # Create device-expert mapping (uniform partitioning)
        # Each device gets contiguous range of experts
        with Profiler().trace_with_timer("device-expert-mapping", level=4):
            device_expert_mappings = []
            for device_id in range(self.num_devices):
                mapping = torch.arange(
                    device_id * self.num_experts_per_device,
                    (device_id + 1) * self.num_experts_per_device,
                    dtype=torch.int32
                )
                device_expert_mappings.append(mapping)

            device_expert_mapping_torch = torch.stack(device_expert_mappings, dim=0)  # (D, E/D)
            device_expert_mapping = ttnn.from_torch(
                device_expert_mapping_torch,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            )

        # Step 1: Prepare routing tensors with device-expert mapping
        with Profiler().trace_with_timer("prepare-routing-tensors", level=4):
            num_routed, routed_tokens, routed_weights, token_idx_map = ttnn.prepare_moe_routing_tensors(
                selected_experts, routing_weights, device_expert_mapping, self.num_experts
            )

        # Prepare expert weights - convert to ROW_MAJOR and squeeze first dimension from (1, E/D, H, H') to (E/D, H, H')
        with Profiler().trace_with_timer("prepare-expert-weights", level=4):
            # Gate projection: convert to ROW_MAJOR and squeeze first dimension
            gate_proj_rm = ttnn.to_layout(self.gate_proj, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gate_proj_rm = ttnn.squeeze(gate_proj_rm, dim=0)

            # Up projection: convert to ROW_MAJOR and squeeze first dimension
            up_proj_rm = ttnn.to_layout(self.up_proj, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            up_proj_rm = ttnn.squeeze(up_proj_rm, dim=0)

            # Down projection: convert to ROW_MAJOR and squeeze first dimension
            down_proj_rm = ttnn.to_layout(self.down_proj, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            down_proj_rm = ttnn.squeeze(down_proj_rm, dim=0)

        # Convert hidden_states to ROW_MAJOR for expert projection
        with Profiler().trace_with_timer("hidden-to-row-major", level=4):
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Step 2: Gate & Up Projection (batched matmul for all experts)
        with Profiler().trace_with_timer("gate-projection", level=4):
            gate_output = ttnn.projection_to_intermediate(
                hidden_states,
                routed_tokens,
                num_routed,
                gate_proj_rm,
                self.top_k
            )

        with Profiler().trace_with_timer("up-projection", level=4):
            up_output = ttnn.projection_to_intermediate(
                hidden_states,
                routed_tokens,
                num_routed,
                up_proj_rm,
                self.top_k
            )

        # Step 3: SiLU(gate) * up
        with Profiler().trace_with_timer("silu-multiply", level=4):
            # Convert to TILE for SiLU (unary ops require TILE layout)
            gate_output = ttnn.to_layout(gate_output, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            up_output = ttnn.to_layout(up_output, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # Apply SiLU to gate_output: silu(x) = x * sigmoid(x)
            gate_silu = ttnn.silu(gate_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # Elementwise multiply
            combined_activations = ttnn.mul(gate_silu, up_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # Convert back to ROW_MAJOR for down projection
            combined_activations = ttnn.to_layout(
                combined_activations, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        # Step 4: Down Projection with routing weights and accumulation
        with Profiler().trace_with_timer("down-projection", level=4):
            num_tokens = batch_size * sequence_length
            moe_output = ttnn.projection_to_output(
                combined_activations,
                token_idx_map,
                routed_tokens,
                num_routed,
                routed_weights,
                down_proj_rm,
                num_tokens,
                self.top_k
            )
        
        # Step 5: Local Reduce
        with Profiler().trace_with_timer("sum", level=4):
            moe_output = ttnn.to_layout(moe_output, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            moe_output = ttnn.sum(moe_output, 0)

        # Step 5: Allreduce across devices (sum partial outputs from all devices)
        with Profiler().trace_with_timer("allreduce", level=4):
            T, H = moe_output.shape
            final_output = ttnn.reshape(moe_output, shape=(1, 1, T * H // 256, 256), memory_config=mem_cfg)
            final_output = ttnn.to_layout(final_output, ttnn.TILE_LAYOUT, memory_config=mem_cfg)
            final_output = ttnn.experimental.all_reduce_async(
                final_output,
                math_op=ttnn.ReduceType.Sum,
                memory_config=mem_cfg,
                topology=ttnn.Topology.Linear,
                from_remote_multi_device_global_semaphore=self.ccl.get_semaphore(0),
                to_remote_multi_device_global_semaphore=self.ccl.get_semaphore(0),
                gather_multi_device_global_semaphore=self.ccl.get_semaphore(0),
                num_links=1,
            )
            ttnn.synchronize_device(self.mesh_device)

        # Reshape to original shape
        with Profiler().trace_with_timer("reshape", level=4):
            # Convert to ROW_MAJOR before reshaping
            final_output = ttnn.to_layout(final_output, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem_cfg)
            # Reshape from (1, 1, T*H//256, 256) back to (T, H)
            num_tokens = batch_size * sequence_length
            final_output = ttnn.reshape(final_output, (num_tokens, hidden_dim), memory_config=mem_cfg)

            if mode == InferenceMode.PREFILL:
                final_hidden_states = ttnn.reshape(
                    final_output,
                    (batch_size, sequence_length, hidden_dim),
                    memory_config=mem_cfg,
                )
            elif mode == InferenceMode.DECODE:
                final_hidden_states = ttnn.reshape(
                    final_output,
                    (1, 1, batch_size, hidden_dim),
                    memory_config=mem_cfg,
                )

        return final_hidden_states

    @profile_trace("Qwen3MoeSparseMoeBlock", level=3)
    def forward(self, hidden_states: ttnn.Tensor, mode: InferenceMode = InferenceMode.PREFILL) -> ttnn.Tensor:
        if mode == InferenceMode.PREFILL:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        elif mode == InferenceMode.DECODE:
            _, sequence_length, batch_size, hidden_dim = hidden_states.shape
            mem_cfg = ttnn.L1_MEMORY_CONFIG # FIXME: Use L1 for decode mode

        with Profiler().trace_with_timer("reshape", level=4):
            hidden_states = ttnn.reshape(hidden_states, (-1, hidden_dim), memory_config=mem_cfg)

        with Profiler().trace_with_timer("moe-router", level=4):
            router_logits = ttnn.linear(hidden_states, self.gate_weight, dtype=ttnn.bfloat16, memory_config=mem_cfg)

        with Profiler().trace_with_timer("softmax-topk-div", level=4):
            routing_weights = ttnn.softmax(router_logits, memory_config=mem_cfg)
            routing_weights, selected_experts = ttnn.topk(
                routing_weights, self.top_k, largest=True, memory_config=mem_cfg
            )
            if self.norm_topk_prob:
                routing_weights = ttnn.div(
                    routing_weights,
                    ttnn.sum(routing_weights, dim=-1, keepdim=True, memory_config=mem_cfg),
                    memory_config=mem_cfg,
                )

        # Convert selected_experts to UINT32 and ROW_MAJOR for routing tensor preparation
        with Profiler().trace_with_timer("typecast-selected-experts", level=4):
            # typecast requires TILE layout
            selected_experts = ttnn.typecast(selected_experts, ttnn.uint32, memory_config=mem_cfg)
            # Convert to ROW_MAJOR after typecast
            selected_experts = ttnn.to_layout(
                selected_experts, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem_cfg
            )
    
        # Convert routing_weights to ROW_MAJOR
        with Profiler().trace_with_timer("to-layout-routing-weights", level=4):
            routing_weights = ttnn.to_layout(
                routing_weights, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem_cfg
            )

        # Create device-expert mapping (uniform partitioning)
        # Each device gets contiguous range of experts
        with Profiler().trace_with_timer("device-expert-mapping", level=4):
            device_expert_mappings = []
            for device_id in range(self.num_devices):
                mapping = torch.arange(
                    device_id * self.num_experts_per_device,
                    (device_id + 1) * self.num_experts_per_device,
                    dtype=torch.int32
                )
                device_expert_mappings.append(mapping)

            device_expert_mapping_torch = torch.stack(device_expert_mappings, dim=0)  # (D, E/D)
            device_expert_mapping = ttnn.from_torch(
                device_expert_mapping_torch,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=mem_cfg,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            )

        # Step 1: Prepare routing tensors with device-expert mapping
        with Profiler().trace_with_timer("prepare-routing-tensors", level=4):
            num_routed, routed_tokens, routed_weights, token_idx_map = ttnn.prepare_moe_routing_tensors(
                selected_experts, routing_weights, device_expert_mapping, self.num_experts
            )
        
        # Step 2: Scatter MoE input
        with Profiler().trace_with_timer("scatter-moe-input", level=4):
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem_cfg)
            scattered_hidden_states = ttnn.scatter_moe_input(hidden_states, num_routed, routed_tokens)
        
        with Profiler().trace_with_timer("to-layout", level=4):
            scattered_hidden_states = ttnn.to_layout(scattered_hidden_states, ttnn.TILE_LAYOUT, memory_config=mem_cfg)

        # Prepare expert weights - squeeze first dimension from (1, E/D, H, H') to (E/D, H, H')
        with Profiler().trace_with_timer("prepare-expert-weights", level=4):
            gate_proj = ttnn.squeeze(self.gate_proj, dim=0)
            up_proj = ttnn.squeeze(self.up_proj, dim=0)
            down_proj = ttnn.squeeze(self.down_proj, dim=0)

        # Step 2: Gate & Up Projection (batched matmul for all experts)
        with Profiler().trace_with_timer("gate-projection", level=4):
            gate_output = ttnn.experimental.moe_bmm(
                scattered_hidden_states,
                gate_proj,
                num_routed
            )

        with Profiler().trace_with_timer("up-projection", level=4):
            up_output = ttnn.experimental.moe_bmm(
                scattered_hidden_states,
                up_proj,
                num_routed
            )

        # Step 3: SiLU(gate) * up
        with Profiler().trace_with_timer("silu-multiply", level=4):
            # Apply SiLU to gate_output: silu(x) = x * sigmoid(x)
            gate_silu = ttnn.silu(gate_output, memory_config=mem_cfg)

            # Elementwise multiply
            combined_activations = ttnn.multiply(gate_silu, up_output, memory_config=mem_cfg)

        # Step 4: Down Projection with routing weights and accumulation
        with Profiler().trace_with_timer("down-projection", level=4):
            num_tokens = batch_size * sequence_length
            moe_output = ttnn.experimental.moe_bmm(
                combined_activations,
                down_proj,
                num_routed
            )
        
        # Step 5: Local Reduce
        with Profiler().trace_with_timer("sum", level=4):
            moe_output = ttnn.to_layout(moe_output, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem_cfg)
            moe_output = ttnn.local_reduce_moe_output(
                moe_output,
                token_idx_map,
                routed_weights,
                num_routed,
                num_tokens,
                memory_config=mem_cfg,
            )

        # Step 5: Allreduce across devices (sum partial outputs from all devices)
        with Profiler().trace_with_timer("allreduce", level=4):
            T, H = moe_output.shape
            final_output = ttnn.reshape(moe_output, shape=(1, 1, T, H), memory_config=mem_cfg)
            final_output = ttnn.to_layout(final_output, ttnn.TILE_LAYOUT, memory_config=mem_cfg)

            final_output = ttnn.experimental.reduce_scatter_minimal_async(
                final_output,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
                barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=1,
                memory_config=mem_cfg,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            final_output = ttnn.experimental.all_gather_async(
                final_output,
                3,
                cluster_axis=1,
                mesh_device=self.mesh_device,
                topology=ttnn.Topology.Linear,
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
                memory_config=mem_cfg,
                num_links=1,
            )
            ttnn.synchronize_device(self.mesh_device)

        # Reshape to original shape
        with Profiler().trace_with_timer("reshape", level=4):
            final_output = ttnn.reshape(final_output, (T, H), memory_config=mem_cfg)

            if mode == InferenceMode.PREFILL:
                final_hidden_states = ttnn.reshape(
                    final_output,
                    (batch_size, sequence_length, hidden_dim),
                    memory_config=mem_cfg,
                )
            elif mode == InferenceMode.DECODE:
                final_hidden_states = ttnn.reshape(
                    final_output,
                    (1, 1, batch_size, hidden_dim),
                    memory_config=mem_cfg,
                )

        return final_hidden_states

__all__ = ["Qwen3MoeMLP", "Qwen3MoeSparseMoeBlock"]
