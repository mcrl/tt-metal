# APIs

## Expert Parallelization Strategies

The API supports flexible expert-to-device mapping via the **Device-Expert Mapping Tensor**.

### Device-Expert Mapping Tensor
- **Shape**: `E / D` per device (e.g., 16 for 128 experts / 8 devices)
- **Data type**: `int32`
- **Content**: Global expert indices (0 to E-1) assigned to this device
- **Usage**: `global_expert_id = device_expert_mapping[local_expert_id]`

### Strategy 1: Equally Distributed (Uniform Partitioning)

**Mapping**:
- Device 0 → Experts `[0, 1, 2, ..., 15]`
- Device 1 → Experts `[16, 17, 18, ..., 31]`
- ...
- Device 7 → Experts `[112, 113, 114, ..., 127]`

**Example**:
- On **Device 1**, local expert 0 → global expert 16
- On **Device 7**, local expert 2 → global expert 114

### Strategy 2: Dynamic Mapping (Load-Balanced Partitioning)

**Mapping** (example with non-contiguous assignment):
- Device 0 → Experts `[0, 5, 12, 18, 27, 33, 41, 50, 58, 66, 74, 82, 90, 98, 106, 114]`
- Device 1 → Experts `[1, 7, 13, 19, 28, 34, 42, 51, 59, 67, 75, 83, 91, 99, 107, 115]`
- ...
- Device 7 → Experts `[6, 11, 17, 26, 32, 40, 49, 57, 65, 73, 81, 89, 97, 105, 113, 127]`

**Use Cases**:
- Load balancing based on token routing statistics
- Dynamic expert reallocation
- Custom partitioning strategies

### Key Insight

The device-expert mapping is **only used in `prepare_moe_routing_tensors`** to filter global routing information into device-local routing tables. Once the device-local routing tables are created, subsequent operations (`projection_to_intermediate` and `projection_to_output`) only use local expert indices and don't need to reference global expert IDs.

---

## `prepare_moe_routing_tensors`

**Python API**
```python
num_routed_tokens, routed_tokens, routed_token_weights = ttnn.prepare_moe_routing_tensors(
    selected_experts,         # (T, K) uint32 tensor
    routing_weights,          # (T, K) bfloat16 tensor
    device_expert_mapping,    # (E/D,) int32 tensor
    num_experts,              # scalar int - total number of experts (E)
    *,
    memory_config=None,
    queue_id=0
)
```

**Input**
- **selected_experts**: `(T, K)` uint32 tensor, ROW_MAJOR layout
	- Global expert IDs selected by each token
	- Shape: T = number of tokens, K = top-k experts per token
	- Each token selects K unique experts (no duplicates per token)
	- Values: 0 to E-1 (global expert indices)
- **routing_weights**: `(T, K)` bfloat16 tensor, ROW_MAJOR layout
	- Routing weights for each selected expert
	- Typically normalized: `sum(routing_weights[t, :]) ≈ 1` for each token t
- **device_expert_mapping**: `(E/D,)` int32 1D tensor, ROW_MAJOR layout
	- Global expert indices (0 to E-1) assigned to this device
	- `device_expert_mapping[local_expert_id]` returns the global expert index
	- Used to filter which tokens should be processed on this device
- **num_experts**: scalar integer
	- Total number of experts (E) in the model

**Output** (all device-local)
- **num_routed_tokens**: `(E/D,)` uint32 1D tensor
	- Count of tokens routed to each local expert
	- `num_routed_tokens[e]` = number of tokens assigned to local expert e
- **routed_tokens**: `(E/D, max_tokens)` uint32 2D tensor
	- Token indices for each local expert (padded)
	- Each row e contains `num_routed_tokens[e]` valid token indices
	- `routed_tokens[e, k]` = token index for k-th token assigned to local expert e
	- Padded entries (beyond `num_routed_tokens[e]`) are set to `0xFFFFFFFF`
	- `max_tokens = T` (worst case: all tokens routed to one expert)
- **routed_token_weights**: `(E/D, max_tokens)` bfloat16 2D tensor
	- Routing weights for each local expert (padded)
	- `routed_token_weights[e, k]` = routing weight for k-th token assigned to local expert e
	- Padded entries are set to `0.0`

**Behavior**
- Filters global routing information to only include experts assigned to this device
- Converts sparse MoE expert selection into device-local routing tensors
- Enables expert-parallel computation where each device processes E/D experts independently

---

## `projection_to_intermediate`

**Python API**
```python
output = ttnn.projection_to_intermediate(
    hidden_states,       # (T, H) bfloat16 tensor - replicated
    routed_tokens,       # (E/D, max_tokens) uint32 tensor - sharded
    num_routed_tokens,   # (E/D,) uint32 tensor - sharded
    expert_weights,      # (E/D, H, H') bfloat16 tensor - sharded
    top_k,               # scalar int - number of experts per token
    *,
    memory_config=None,
    queue_id=0
)
```

**Input**
- **hidden_states**: `(T, H)` bfloat16 tensor, ROW_MAJOR layout
	- Input token embeddings, replicated across all devices
	- `hidden_states[t, :]` is the hidden state vector for token t
- **routed_tokens**: `(E/D, max_tokens)` uint32 tensor, ROW_MAJOR layout
	- Device-local token indices from `prepare_moe_routing_tensors`
	- Sharded across devices (each device has different E/D experts)
- **num_routed_tokens**: `(E/D,)` uint32 1D tensor, ROW_MAJOR layout
	- Device-local token counts from `prepare_moe_routing_tensors`
	- Sharded across devices
- **expert_weights**: `(E/D, H, H')` bfloat16 tensor, ROW_MAJOR layout
	- Expert weight matrices, sharded across devices by expert dimension
	- `expert_weights[e, :, :]` contains weights for local expert e
	- Indexed by local expert index (0 to E/D-1)
- **top_k**: scalar integer
	- Number of experts selected per token (K)

**Output**
- **output**: `(K * T, H')` bfloat16 tensor, ROW_MAJOR layout
	- Pre-allocated tensor of size K × T rows
	- Only the first `sum(num_routed_tokens)` rows contain valid data
	- Remaining rows are zero-padded
	- Rows are filled sequentially by local expert index

**Computation**
For each local expert e in [0, E/D-1):
1. Read `T_e = num_routed_tokens[e]` (number of tokens for this expert)
2. Read `token_indices = routed_tokens[e, :T_e]` (which tokens to process)
3. For each of the T_e tokens:
   - Gather input: `x = hidden_states[token_indices[i], :]` (shape: 1 × H)
   - Compute: `y = x @ expert_weights[e, :, :]` (shape: 1 × H')
   - Write to output at position: `write_pos = sum(num_routed_tokens[0:e]) + i`
   - `output[write_pos, :] = y`

**Memory Layout**
- Output rows are organized by local expert:
  ```
  [expert_0_token_0, expert_0_token_1, ..., expert_0_token_{T0-1},
   expert_1_token_0, expert_1_token_1, ..., expert_1_token_{T1-1},
   ...,
   expert_{E/D-1}_token_0, ..., expert_{E/D-1}_token_{T-1},
   <zero padding>]
  ```

**Usage**
- Used for both `gate_proj` and `up_proj` in MoE layers
- Each device processes its assigned E/D experts in parallel
- No cross-device communication required during computation
- Device-expert mapping is NOT needed (routing tensors are already device-local)

---

## `projection_to_output`

**Python API**
```python
output = ttnn.projection_to_output(
    combined_activations,     # (T_d, H') bfloat16 tensor - replicated
    routed_tokens,            # (E/D, max_tokens) uint32 tensor - sharded
    num_routed_tokens,        # (E/D,) uint32 tensor - sharded
    routed_token_weights,     # (E/D, max_tokens) bfloat16 tensor - sharded
    down_proj_weights,        # (E/D, H', H) bfloat16 tensor - sharded
    num_tokens,               # scalar int - T
    top_k,                    # scalar int - K
    *,
    memory_config=None,
    queue_id=0
)
```

**Input**
- **combined_activations**: `(T_d, H')` bfloat16 tensor, ROW_MAJOR layout
	- Combined gate × up activations from previous MoE layer
	- `T_d = sum(num_routed_tokens)` across all local experts on this device
	- Only the first T_d rows contain valid data (compacted format)
	- Typically replicated across devices (but could be distributed)
- **routed_tokens**: `(E/D, max_tokens)` uint32 tensor, ROW_MAJOR layout
	- Device-local token indices from `prepare_moe_routing_tensors`
	- Sharded across devices
- **num_routed_tokens**: `(E/D,)` uint32 1D tensor, ROW_MAJOR layout
	- Device-local token counts from `prepare_moe_routing_tensors`
	- Sharded across devices
- **routed_token_weights**: `(E/D, max_tokens)` bfloat16 tensor, ROW_MAJOR layout
	- Device-local routing weights from `prepare_moe_routing_tensors`
	- Sharded across devices
	- Each weight multiplies the corresponding expert output before accumulation
- **down_proj_weights**: `(E/D, H', H)` bfloat16 tensor, ROW_MAJOR layout
	- Down projection weights, sharded across devices by expert dimension
	- `down_proj_weights[e, :, :]` contains weights for local expert e
	- Note: Up projection is `H → H'`, down projection is `H' → H` (dimensions transposed)
- **num_tokens**: scalar integer
	- Total number of tokens (T)
- **top_k**: scalar integer
	- Number of experts per token (K)

**Output**
- **output**: `(T, H)` bfloat16 tensor, ROW_MAJOR layout
	- Final MoE output for all tokens
	- Initialized to zeros, then accumulated
	- Each device produces partial results that need **allreduce** across devices

**Computation**
For each local expert e in [0, E/D-1):
1. Read `T_e = num_routed_tokens[e]` (number of tokens for this expert)
2. Read `token_indices = routed_tokens[e, :T_e]` (which tokens)
3. Read `weights = routed_token_weights[e, :T_e]` (routing weights)
4. Calculate input start: `input_start = sum(num_routed_tokens[0:e])`
5. For each of the T_e tokens:
   - Read input: `x = combined_activations[input_start + i, :]` (shape: 1 × H')
   - Compute: `y = x @ down_proj_weights[e, :, :]` (shape: 1 × H)
   - Apply routing weight: `y_weighted = y * weights[i]`
   - **Accumulate** (not overwrite): `output[token_indices[i], :] += y_weighted`

**Key Behavior**
- **Accumulation**: Each token receives contributions from multiple experts (up to K)
- **Partial results**: Each device computes partial sums for its assigned experts
- **Requires allreduce**: Final step (Step 5) sums outputs across all devices to get complete result
- **Single-core**: Current implementation uses single core (no atomic operations needed)

**Multi-Device Flow**
```
Device 0: Processes experts [0, 1, ..., E/D-1]     → partial_output_0
Device 1: Processes experts [E/D, E/D+1, ..., 2E/D-1] → partial_output_1
...
Device D-1: Processes experts [E-E/D, ..., E-1]    → partial_output_{D-1}

Final: allreduce(partial_output_0, ..., partial_output_{D-1}) → final_output
```

**Implementation Notes**
- Current: Single-core per device (no race conditions)
- Future: Multi-core would require atomic operations or reduction patterns
- Device-expert mapping is NOT needed (routing tensors are already device-local)

---

## Complete MoE Pipeline

The three operations work together to implement expert-parallel MoE computation:

### Step 1: Prepare Routing (once per forward pass)
```python
# Input: Global routing information
selected_experts      # (T, K) - which experts each token selected
routing_weights       # (T, K) - routing weights for each expert
device_expert_mapping # (E/D,) - which experts this device owns

# Output: Device-local routing information
num_routed_tokens, routed_tokens, routed_token_weights = \
    ttnn.prepare_moe_routing_tensors(
        selected_experts, routing_weights, device_expert_mapping, num_experts
    )
```

### Step 2: Projection to Intermediate (gate_proj and up_proj)
```python
# Gate projection
gate_output = ttnn.projection_to_intermediate(
    hidden_states, routed_tokens, num_routed_tokens, gate_weights, top_k
)  # Shape: (K*T, H')

# Up projection
up_output = ttnn.projection_to_intermediate(
    hidden_states, routed_tokens, num_routed_tokens, up_weights, top_k
)  # Shape: (K*T, H')

# Combine: gate * up (element-wise)
combined = gate_output * up_output  # Shape: (K*T, H')
```

### Step 3: Projection to Output (down_proj)
```python
# Down projection with accumulation
partial_output = ttnn.projection_to_output(
    combined, routed_tokens, num_routed_tokens, routed_token_weights,
    down_weights, num_tokens, top_k
)  # Shape: (T, H) - partial result per device
```

### Step 4: Allreduce (multi-device)
```python
# Sum partial outputs across all devices
final_output = ttnn.all_reduce(partial_output, mesh_device)  # Shape: (T, H)
```

### Data Flow Summary

```
Global Routing Info (T, K)
         ↓
    [prepare_moe_routing_tensors]
         ↓
Device-Local Routing (E/D, max_tokens) - SHARDED
         ↓
    ┌────────────────────────────────┐
    │                                │
    ↓                                ↓
[projection_to_intermediate]    [projection_to_intermediate]
   gate_proj (T→H')                up_proj (T→H')
    ↓                                ↓
    └──────────→ gate * up ←─────────┘
                   ↓
         combined (K*T, H') - COMPACTED
                   ↓
        [projection_to_output]
           down_proj (H'→H)
                   ↓
         partial_output (T, H) per device
                   ↓
             [allreduce]
                   ↓
         final_output (T, H) - COMPLETE
```

### Key Design Decisions

1. **Expert Parallelism**: Each device processes E/D experts independently
2. **Device-Local Routing**: Routing tensors filtered once, reused for all projections
3. **Compacted Format**: Intermediate activations stored densely (no padding between expert outputs)
4. **Accumulation Pattern**: Output accumulates contributions from multiple experts per token
5. **Allreduce Required**: Final step sums partial results across devices

### Memory Distribution

- **Replicated**: `hidden_states`, `combined_activations`
- **Sharded**: `expert_weights`, `routed_tokens`, `num_routed_tokens`, `routed_token_weights`
- **Partial**: `output` (requires allreduce)