import ttnn
import torch
import math
from models.demos.gpt_oss.config import MeshConfig, Mode, ModeConfig
from models.demos.gpt_oss.tt.ccl import CCLManager
from loguru import logger

def print_sync_print(tag, device):
    if False:
        logger.info(f'{tag} - BEFORE SYNC')
        ttnn.synchronize_device(device)
        logger.info(f'{tag} - AFTER SYNC')

def topk_router(g, experts_per_token, mesh_device):
    print_sync_print('topkrouter before topk', mesh_device)
    if g.dtype != ttnn.bfloat16:
        expert_weights, expert_indices = ttnn.topk(ttnn.typecast(g, dtype=ttnn.bfloat16), k=experts_per_token, dim=-1, sorted=True)
        expert_weights = ttnn.typecast(expert_weights, dtype=g.dtype)
    else:
        expert_weights, expert_indices = ttnn.topk(g, k=experts_per_token, dim=-1, sorted=True)
    compute_config = ttnn.init_device_compute_kernel_config(
        g.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    print_sync_print('topkrouter before softmax', mesh_device)
    expert_weights = ttnn.softmax(expert_weights, dim=1, numeric_stable=True, compute_kernel_config=compute_config)
    router_scores = ttnn.scatter(ttnn.zeros_like(g), dim=1, index=expert_indices, src=expert_weights)
    return router_scores, expert_weights, expert_indices

class TopKRouter:
    def __init__(self, mesh_device, num_experts_per_tok, num_local_experts, hidden_size, tensor_cache_path=None):
        self.top_k = num_experts_per_tok
        self.num_experts = num_local_experts
        self.hidden_dim = hidden_size
        self.weight = ttnn.as_tensor(
            torch.randn((self.num_experts, self.hidden_dim), dtype=torch.bfloat16).transpose(0, 1),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bias = ttnn.as_tensor(
            torch.randn((self.num_experts,), dtype=torch.bfloat16).unsqueeze(0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.compute_config = None
        self.mesh_device = mesh_device

    def __call__(self, hidden_states):
        print_sync_print('topkrouter before reshape', self.mesh_device)
        hidden_states = ttnn.reshape(hidden_states, (-1, self.hidden_dim))
        print_sync_print('topkrouter before linear', self.mesh_device)
        print_sync_print('topkrouter before typecast', self.mesh_device)
        hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)
        print_sync_print('topkrouter after typecast', self.mesh_device)
        router_logits = ttnn.linear(
            hidden_states, self.weight, bias=self.bias, compute_kernel_config=self.compute_config
        )
        print_sync_print('topkrouter after linear', self.mesh_device)
        router_logits = ttnn.typecast(router_logits, ttnn.bfloat8_b)
        router_scores, _expert_weights, router_indices = topk_router(router_logits, self.top_k, self.mesh_device)
        return router_scores, router_indices, router_logits

class Experts:
    def __init__(
        self,
        mesh_device,
        intermediate_size,
        num_local_experts,
        hidden_size,
        num_experts_per_tok,
        ccl_manager,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        mesh_config=None,
    ):
        self.intermediate_size = intermediate_size
        self.num_experts = num_local_experts
        self.hidden_size = hidden_size
        self.expert_dim = self.intermediate_size
        self.ccl_manager = ccl_manager
        self.mesh_device = mesh_device
        self.num_experts_per_tok = num_experts_per_tok

        # Use mode-aware MeshConfig for clean parallelization
        # Decode: EP=rows for expert parallelism, SP=1
        # Prefill: EP=1, SP=rows for sequence parallelism (auto-defaults from MeshConfig)
        self.mesh_config = mesh_config or MeshConfig(
            mesh_device.shape,
            decode=ModeConfig(tp=mesh_device.shape[1], ep=mesh_device.shape[0], sp=1)
            # prefill auto-defaults to: ModeConfig(tp=cols, sp=rows, ep=1)
        )

        # Use decode config for weight loading (conservative choice)
        self.intermediate_size_per_device = self.mesh_config.shard_size(self.intermediate_size, mode=Mode.DECODE)

        # num_experts = 32
        # hidden_size = 2880
        # expert_dim = 2880
        gate_proj = torch.randn((1, self.num_experts, self.hidden_size, self.expert_dim), dtype=torch.bfloat16)
        up_proj = torch.randn((1, self.num_experts, self.hidden_size, self.expert_dim), dtype=torch.bfloat16)
        gate_proj_bias = torch.randn((1, self.num_experts, 1, self.expert_dim), dtype=torch.bfloat16)
        up_proj_bias = torch.randn((1, self.num_experts, 1, self.expert_dim), dtype=torch.bfloat16)

        # Clean mesh mapping using MeshConfig (use decode config for weights)
        col_mesh_mapper = self.mesh_config.column_parallel(mesh_device)
        row_mesh_mapper = self.mesh_config.row_parallel(mesh_device)
        self.gate_proj = ttnn.as_tensor(
            gate_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_proj = ttnn.as_tensor(
            up_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.gate_proj_bias = ttnn.as_tensor(
            gate_proj_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_proj_bias = ttnn.as_tensor(
            up_proj_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        down_proj = torch.randn((1, self.num_experts, self.expert_dim, self.hidden_size), dtype=torch.bfloat16)
        down_proj_bias = torch.randn((1, self.num_experts, 1, self.hidden_size), dtype=torch.bfloat16)
        self.down_proj = ttnn.as_tensor(
            down_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=row_mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Row-parallel bias must not be replicated. Extend it with zeros for TP devices.
        if self.mesh_config.decode.tp > 1:
            down_proj_bias = torch.cat(
                [down_proj_bias] + [torch.zeros_like(down_proj_bias)] * (self.mesh_config.decode.tp - 1), dim=-1
            )
        self.down_proj_bias = ttnn.as_tensor(
            down_proj_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=col_mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.alpha = 1.702  # from https://github.com/huggingface/transformers/blob/b4067472aee9b566237091dbcd3659dd2ce92004/src/transformers/models/gpt_oss/modular_gpt_oss.py#L77
        self.limit = 7.0

        # Prefill sparsity setup uses prefill config (for ep dimension)
        prefill_ep = self.mesh_config.prefill.ep
        tokens_per_ep = self.num_experts // prefill_ep
        sparsity = torch.zeros(1, 1, prefill_ep, self.num_experts)
        for i in range(prefill_ep):
            sparsity[:, :, i, i * tokens_per_ep : (i + 1) * tokens_per_ep] = torch.ones(1, 1, 1, tokens_per_ep)
        self.prefill_sparsity = ttnn.from_torch(
            sparsity,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                dims=(-2, None) if prefill_ep > 1 else (None, None),
                mesh_shape=self.mesh_device.shape,
                mesh_device=self.mesh_device,
            ),
        )

        self.sparse_matmul_program_config = (
            lambda core_x, core_y, m, n: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=1,
                per_core_M=max(32, m) // 32,
                per_core_N=int(math.ceil(n / 32)) // (core_x * core_y),
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )
        )
        self.batched_sparse_matmul_program_config = (
            lambda core_x, core_y, m, n: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=1,
                per_core_M=max(32, m) // 32,
                per_core_N=int(math.ceil(n / 32)) // (core_x * core_y),
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )
        )

    def __call__(self, hidden_states, routing_weights):
        print_sync_print('Experts start', self.mesh_device)

        # Select the appropriate mode based on sequence length
        is_prefill = hidden_states.shape[-2] > 32
        mode = Mode.PREFILL if is_prefill else Mode.DECODE

        # Get mode-specific config
        config = self.mesh_config.get_config(mode)
        ep, sp, tp = config.ep, config.sp, config.tp

        seq_len_global = hidden_states.shape[1]

        if sp > 1:
            hidden_states_torch = ttnn.to_torch(ttnn.get_device_tensors(hidden_states)[0])
            routing_weights_torch = ttnn.to_torch(ttnn.get_device_tensors(routing_weights)[0])
            #hidden_states.deallocate(False)
            routing_weights.deallocate(True)
            routing_weights = ttnn.from_torch(
                routing_weights_torch,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    dims=(-2, None), mesh_shape=self.mesh_device.shape, mesh_device=self.mesh_device
                ),
            )
            hidden_states = ttnn.from_torch(
                hidden_states_torch,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    dims=(-2, None), mesh_shape=self.mesh_device.shape, mesh_device=self.mesh_device
                ),
            )
        CHUNK_SIZE = 4 * 1024  # TODO: Make this a parameter

        print_sync_print('Before split', self.mesh_device)

        if hidden_states.shape[1] > CHUNK_SIZE:
            hidden_states_chunks_list = ttnn.split(hidden_states, CHUNK_SIZE, dim=1)
            #hidden_states.deallocate(False)
            routing_weights_chunks_list = ttnn.split(routing_weights, CHUNK_SIZE, dim=0)
            routing_weights.deallocate(True)

        else:
            hidden_states_chunks_list = [hidden_states]
            routing_weights_chunks_list = [routing_weights]

        print_sync_print('Before chunk', self.mesh_device)

        next_states_list = []
        for j in range(len(hidden_states_chunks_list)):
            hidden_states = hidden_states_chunks_list[j]
            routing_weights = routing_weights_chunks_list[j]
            batch_size = hidden_states.shape[0]
            assert batch_size == 1, "batch_size must be 1, we only support batch size 1 for now"
            seq_len = hidden_states.shape[1]
            print_sync_print('unsqueeze_to_4D', self.mesh_device)
            hidden_states_4D = ttnn.unsqueeze_to_4D(hidden_states)
            print_sync_print('to_layout', self.mesh_device)
            sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)
            output_tile = ttnn.Tile([32, 32])
            print_sync_print('reshape', self.mesh_device)
            if seq_len > 1:
                TILE_SIZE = 32
                hidden_states_4D = ttnn.reshape(
                    hidden_states_4D, (1, seq_len // TILE_SIZE, TILE_SIZE, self.hidden_size)
                )
                group_size = seq_len // TILE_SIZE
                sparsity = ttnn.repeat(self.prefill_sparsity, (1, 1, group_size, 1))

            print_sync_print('moe_routing_remap', self.mesh_device)
            if ep > 1 and seq_len == 1:
                sparsity = ttnn.moe_routing_remap(ttnn.reshape(sparsity, (1, sparsity.shape[-1])), 4, 4, 0)
                routing_weights = ttnn.tilize_with_zero_padding(sparsity, use_multicore=True)

            num_experts_per_tok = (
                (self.num_experts // ep) * group_size if seq_len > 1 else self.num_experts_per_tok // ep
            )
            program_config = self.sparse_matmul_program_config(3, 4, hidden_states_4D.shape[2], self.gate_proj.shape[3])

            print_sync_print('sparse_matmul', self.mesh_device)
            gate = ttnn.sparse_matmul(
                hidden_states_4D,
                self.gate_proj,
                sparsity=sparsity,
                nnz=num_experts_per_tok,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=program_config,
                dtype=ttnn.bfloat8_b,
            )

            print_sync_print('transpose', self.mesh_device)
            if seq_len > 1:
                gate_transposed = ttnn.transpose(gate, 1, 3)
                gate.deallocate(True)
                gate = gate_transposed

            print_sync_print('reshape', self.mesh_device)
            gate = ttnn.reshape(gate, (batch_size, self.num_experts, seq_len, self.intermediate_size_per_device))
            gate = ttnn.add(gate, self.gate_proj_bias, output_tensor=gate)
            gate_clamped = ttnn.clamp(gate, min=None, max=self.limit)
            gate.deallocate(True)
            gate = gate_clamped

            print_sync_print('sparse_matmul2', self.mesh_device)
            up = ttnn.sparse_matmul(
                hidden_states_4D,
                self.up_proj,
                sparsity=sparsity,
                nnz=num_experts_per_tok,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=program_config,
                dtype=ttnn.bfloat8_b,
            )

            hidden_states_4D.deallocate(False)

            if seq_len > 1:
                up_transposed = ttnn.transpose(up, 1, 3)
                up.deallocate(True)
                up = up_transposed

            print_sync_print('upup', self.mesh_device)
            up = ttnn.reshape(up, (batch_size, self.num_experts, seq_len, self.intermediate_size_per_device))
            up = ttnn.add(up, self.up_proj_bias, output_tensor=up)
            up_clamped = ttnn.clamp(up, min=-self.limit, max=self.limit)
            up.deallocate(True)
            up = up_clamped

            print_sync_print('gategate', self.mesh_device)
            gate_alpha = ttnn.mul(gate, self.alpha)
            gate_sigmoid = ttnn.sigmoid(gate_alpha)
            gate_alpha.deallocate(True)
            glu = ttnn.mul(gate, gate_sigmoid, output_tensor=gate)
            gate_sigmoid.deallocate(True)
            up = ttnn.add(up, 1, output_tensor=up)
            down_in0 = ttnn.mul(up, glu, output_tensor=up)
            ttnn.deallocate(glu)
            down_in0 = ttnn.reshape(down_in0, (1, self.num_experts, seq_len, self.intermediate_size_per_device))

            if seq_len > 1:
                # down_in0 = ttnn.reshape(down_in0, (1, self.num_experts, group_size, seq_len//group_size, self.intermediate_size_per_device))
                # down_in0 = ttnn.transpose(down_in0, 1, 3)
                # down_in0 = ttnn.reshape(down_in0, (1, self.num_experts, seq_len, self.intermediate_size_per_device))
                sparsity = self.prefill_sparsity
                num_experts_per_tok = self.num_experts // ep
                routing_weights = ttnn.mul(
                    routing_weights,
                    ttnn.reshape(self.prefill_sparsity, (1, self.num_experts)),
                    output_tensor=routing_weights,
                )

            print_sync_print('routing_weights', self.mesh_device)
            routing_weights_transposed = ttnn.permute(routing_weights, (1, 0))
            routing_weights.deallocate(True)
            routing_weights = routing_weights_transposed
            routing_weights = ttnn.reshape(routing_weights, (batch_size, self.num_experts, seq_len, 1))

            SPLIT_SIZE = 1024
            if seq_len > SPLIT_SIZE:
                down_in0_list = ttnn.split(down_in0, SPLIT_SIZE, dim=2)
                down_in0.deallocate(True)
                routing_weights_list = ttnn.split(routing_weights, SPLIT_SIZE, dim=2)
                routing_weights.deallocate(True)
            else:
                down_in0_list = [down_in0]
                routing_weights_list = [routing_weights]

            print_sync_print('downdown', self.mesh_device)
            next_states_reduced_list = []
            for i, down_in0 in enumerate(down_in0_list):
                down = ttnn.sparse_matmul(
                    down_in0,
                    self.down_proj,
                    sparsity=sparsity,
                    nnz=num_experts_per_tok,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    output_tile=output_tile,
                    is_input_a_sparse=True,
                    program_config=self.batched_sparse_matmul_program_config(
                        5, 6, down_in0.shape[2], self.down_proj.shape[-1]
                    ),
                    dtype=ttnn.bfloat8_b,
                )

                down_in0.deallocate(True)
                next_states = ttnn.reshape(
                    down,
                    (batch_size, self.num_experts, (seq_len if seq_len < SPLIT_SIZE else SPLIT_SIZE), self.hidden_size),
                )
                next_states = ttnn.add(next_states, self.down_proj_bias, output_tensor=next_states)

                next_states = ttnn.mul(next_states, routing_weights_list[i], output_tensor=next_states)
                next_states_reduced = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
                next_states.deallocate(True)
                next_states_reduced_list.append(next_states_reduced)
                routing_weights_list[i].deallocate(True)

            next_states = ttnn.concat(next_states_reduced_list, dim=2)


            print_sync_print('EP comm', self.mesh_device)
            # EP communication (expert parallel allreduce)
            if ep > 1:
                next_states = self.mesh_config.allreduce(next_states, self.ccl_manager, axis=self.mesh_config.ep_axis)

            print_sync_print('TP comm', self.mesh_device)
            # TP communication (tensor parallel allreduce)
            if tp > 1:
                if seq_len > 1:
                    ttnn.synchronize_device(self.mesh_device)
                next_states = self.mesh_config.allreduce(
                    ttnn.typecast(next_states, ttnn.bfloat16),
                    self.ccl_manager,
                    pad_size=192 if tp == 8 else 0,
                    axis=self.mesh_config.tp_axis,
                )
                next_states = ttnn.typecast(next_states, ttnn.bfloat8_b)

            next_states_list.append(next_states)

        print_sync_print('Before concat', self.mesh_device)

        next_states = ttnn.concat(next_states_list, dim=2)

        print_sync_print('Before allgather', self.mesh_device)

        # SP communication (sequence parallel all-gather)
        if sp > 1:
            next_states_allgathered = self.mesh_config.allgather(
                next_states, self.ccl_manager, axis=self.mesh_config.sp_axis, dim=-2
            )
            print_sync_print('Before allgather - deallocate', self.mesh_device)
            next_states.deallocate(True)
            next_states = next_states_allgathered
        print_sync_print('Before allgather - reshape', self.mesh_device)
        next_states = ttnn.reshape(
            next_states,
            (batch_size, seq_len_global, self.hidden_size),
            (batch_size, max(32, seq_len_global), self.hidden_size),
        )
        print_sync_print('Before allgather - return', self.mesh_device)

        return next_states

class MLP:
    def __init__(
        self,
        mesh_device,
        num_experts_per_tok,
        num_local_experts,
        hidden_size,
        intermediate_size,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        mesh_config=None,
    ):
        self.ccl_manager = CCLManager(mesh_device)
        self.router = TopKRouter(
            mesh_device,
            num_experts_per_tok,
            num_local_experts,
            hidden_size,
        )
        self.experts = Experts(
            mesh_device,
            intermediate_size,
            num_local_experts,
            hidden_size,
            num_experts_per_tok,
            self.ccl_manager,
            dtype=dtype,
            mesh_config=mesh_config,
        )

    def __call__(self, hidden_states):
        """Forward pass: route -> experts"""
        router_scores, router_indices, router_logits = self.router(hidden_states)
        router_logits.deallocate()
        router_indices.deallocate()
        expert_output = self.experts(hidden_states, router_scores)
        return expert_output, router_scores
