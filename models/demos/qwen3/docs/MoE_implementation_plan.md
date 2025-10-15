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
num_routed_tokens, routed_tokens, routed_token_weights, token_idx_map = ttnn.prepare_moe_routing_tensors(
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
- **num_routed_tokens**: `(E/D, 1)` uint32 2D tensor
	- Count of tokens routed to each local expert
	- `num_routed_tokens[e, 0]` = number of tokens assigned to local expert e
	- **Implementation note**: Uses 2D shape `(E/D, 1)` instead of 1D `(E/D,)` to enable per-element pages in ROW_MAJOR layout
	- Each row is a separate page containing exactly one element, allowing safe multi-core writes without race conditions
- **routed_tokens**: `(E/D, max_tokens)` uint32 2D tensor
	- Token indices for each local expert (padded)
	- Each row e contains `num_routed_tokens[e, 0]` valid token indices
	- `routed_tokens[e, k]` = token index for k-th token assigned to local expert e
	- Padded entries (beyond `num_routed_tokens[e, 0]`) are set to `0xFFFFFFFF`
	- `max_tokens = T` (worst case: all tokens routed to one expert)
- **routed_token_weights**: `(E/D, max_tokens)` bfloat16 2D tensor
	- Routing weights for each local expert (padded)
	- `routed_token_weights[e, k]` = routing weight for k-th token assigned to local expert e
	- Padded entries are set to `0.0`
- **token_idx_map**: `(E/D, max_tokens)` uint32 2D tensor
	- Mapping from expert-local token index to global token index
	- `token_idx_map[e, k]` = global token index for k-th token assigned to local expert e
	- For expert e, `token_idx_map[e][t_e] = t_g` where t_e is the local index (0 to num_routed_tokens[e]-1) and t_g is the global token index in the original batch
	- Used by `projection_to_output` to map expert-local results back to global token positions

**Behavior**
- Filters global routing information to only include experts assigned to this device
- Converts sparse MoE expert selection into device-local routing tensors
- Enables expert-parallel computation where each device processes E/D experts independently

**Parallelization Strategy**

Uses **expert parallelism** where one Tensix core handles one local expert:
- **Core assignment**: Core `c` processes local expert `c` (for `c` in `[0, E/D-1]`)
- **Per-core computation**:
  1. Read global expert ID: `global_expert_id = device_expert_mapping[c]`
  2. Initialize counters: `count = 0`, `write_pos = 0`
  3. Scan all tokens: For each token `t` in `[0, T-1]`:
     - Check all K selections: For each `k` in `[0, K-1]`:
       - If `selected_experts[t, k] == global_expert_id`:
         - Write token index: `routed_tokens[c, write_pos] = t`
         - Write routing weight: `routed_token_weights[c, write_pos] = routing_weights[t, k]`
         - Increment: `count++`, `write_pos++`
  4. Write final count: `num_routed_tokens[c, 0] = count`
  5. Pad remaining entries: Set `routed_tokens[c, write_pos:] = 0xFFFFFFFF`, `routed_token_weights[c, write_pos:] = 0.0`

**Key insights**:
- Each core independently scans the entire input `(T, K)` but only writes output for its assigned expert
- No inter-core communication or synchronization needed during computation
- Per-element pages enable race-free writes: Each core writes to page ID `c`, which contains exactly one element `num_routed_tokens[c, 0]`

**Implementation details**:
- **Per-element pages**: `num_routed_tokens` uses shape `(E/D, 1)` so each row is one page with size `sizeof(uint32_t)`
- **TensorAccessor**: Use `page_size = sizeof(uint32_t)` to access individual elements via `get_noc_addr(local_expert_id, accessor)`
- **Safe multi-core writes**: Core `c` writes to page ID `c`, eliminating race conditions without read-modify-write
- **Downstream operations**: `projection_to_intermediate` and `projection_to_output` read `num_routed_tokens` with same per-element page size

---

## `projection_to_intermediate`

**Python API**
```python
output = ttnn.projection_to_intermediate(
    hidden_states,       # (T, H) bfloat16 tensor - replicated
    routed_tokens,       # (E/D, max_tokens) uint32 tensor - sharded
    num_routed_tokens,   # (E/D, 1) uint32 tensor - sharded
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
- **num_routed_tokens**: `(E/D, 1)` uint32 2D tensor, ROW_MAJOR layout
	- Device-local token counts from `prepare_moe_routing_tensors`
	- Sharded across devices
	- Access as `num_routed_tokens[e, 0]` or squeeze to 1D before use
- **expert_weights**: `(E/D, H, H')` bfloat16 tensor, ROW_MAJOR layout
	- Expert weight matrices, sharded across devices by expert dimension
	- `expert_weights[e, :, :]` contains weights for local expert e
	- Indexed by local expert index (0 to E/D-1)
- **top_k**: scalar integer
	- Number of experts selected per token (K)

**Output**
- **output**: `(E/D, T, H')` bfloat16 tensor, ROW_MAJOR layout
	- Projection outputs organized by expert
	- Shape: E/D experts × T tokens × H' intermediate dimensions
	- Each expert processes its routed tokens and writes to its dedicated slice
	- Only the first `num_routed_tokens[e, 0]` rows contain valid data for expert e
	- Remaining rows are zero-padded to maintain uniform tensor shape
	- Output is compacted per expert: expert e's outputs are at `output[e, 0:num_routed_tokens[e, 0], :]`

**Computation**
For each local expert e in [0, E/D-1):
1. Read `T_e = num_routed_tokens[e, 0]` (number of tokens for this expert)
2. Read `token_indices = routed_tokens[e, :T_e]` (which tokens to process)
3. For each token position t in [0, T):
   - If t < T_e:
     - Gather input: `x = hidden_states[token_indices[t], :]` (shape: 1 × H)
     - Compute: `y = x @ expert_weights[e, :, :]` (shape: 1 × H')
     - Write to output at position: `output[e, t, :] = y`
   - Else:
     - Write zero padding: `output[e, t, :] = 0`

**Memory Layout**
- Output rows are organized by expert with padding:
  ```
  [expert_0: [token_0, token_1, ..., token_{T0-1}, <padding to T>],
   expert_1: [token_0, token_1, ..., token_{T1-1}, <padding to T>],
   ...,
   expert_{E/D-1}: [token_0, ..., token_{T-1}, <padding to T>]]
  ```
- Each expert's slice has T rows, but only the first `num_routed_tokens[e, 0]` contain valid data

**Usage**
- Used for both `gate_proj` and `up_proj` in MoE layers
- Each device processes its assigned E/D experts in parallel
- No cross-device communication required during computation
- Device-expert mapping is NOT needed (routing tensors are already device-local)

**Implementation Notes**
- Current: Single-core implementation (sequential expert processing)
- Future: Multi-core would process experts in parallel across multiple cores

---

## `projection_to_output`

**Python API**
```python
output = ttnn.projection_to_output(
    combined_activations,     # (E/D, T, H') bfloat16 tensor - sharded
    token_idx_map,            # (E/D, max_tokens) uint32 tensor - sharded
    routed_tokens,            # (E/D, max_tokens) uint32 tensor - sharded
    num_routed_tokens,        # (E/D, 1) uint32 tensor - sharded
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
- **combined_activations**: `(E/D, T, H')` bfloat16 tensor, ROW_MAJOR layout
	- Combined gate × up activations from previous MoE layer
	- Shape: E/D experts × T tokens × H' intermediate dimensions
	- Each expert's slice contains outputs for its routed tokens (padded to T)
	- Only the first `num_routed_tokens[e, 0]` rows contain valid data for expert e
	- Sharded across devices (each device has different E/D experts)
- **token_idx_map**: `(E/D, max_tokens)` uint32 tensor, ROW_MAJOR layout
	- Mapping from expert-local token index to global token index
	- Device-local mapping from `prepare_moe_routing_tensors`
	- `token_idx_map[e, k]` = global token index for k-th token assigned to local expert e
	- Used to write expert outputs to correct positions in global output tensor
	- Sharded across devices
- **routed_tokens**: `(E/D, max_tokens)` uint32 tensor, ROW_MAJOR layout
	- Device-local token indices from `prepare_moe_routing_tensors`
	- Sharded across devices
- **num_routed_tokens**: `(E/D, 1)` uint32 2D tensor, ROW_MAJOR layout
	- Device-local token counts from `prepare_moe_routing_tensors`
	- Sharded across devices
	- Access as `num_routed_tokens[e, 0]` or squeeze to 1D before use
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
- **output**: `(E/D, T, H)` bfloat16 tensor, ROW_MAJOR layout
	- Partial MoE output for all tokens
	- Shape: E/D experts × T tokens × H hidden dimensions
	- Initialized to zeros, then accumulated
	- Each device produces partial results for its E/D experts
	- Requires **allreduce** across devices to get complete result
	- Each expert e writes to positions determined by `token_idx_map[e, :]`

**Computation**
For each local expert e in [0, E/D-1):
1. Read `T_e = num_routed_tokens[e, 0]` (number of tokens for this expert)
2. Read `token_indices = token_idx_map[e, :T_e]` (global token indices)
3. Read `weights = routed_token_weights[e, :T_e]` (routing weights)
4. For each token position t in [0, T):
   - If t < T_e:
     - Read input: `x = combined_activations[e, t, :]` (shape: 1 × H')
     - Compute: `y = x @ down_proj_weights[e, :, :]` (shape: 1 × H)
     - Apply routing weight: `y_weighted = y * weights[t]`
     - Get global token index: `global_idx = token_indices[t]`
     - **Accumulate** (not overwrite): `output[e, global_idx, :] += y_weighted`
   - Else: skip (padding position)

**Key Behavior**
- **Accumulation**: Each token receives contributions from multiple experts (up to K)
- **Partial results**: Each device computes partial sums for its assigned experts
- **Local reduce**: After expert computation, sum across expert dimension to reduce (E/D, T, H) → (T, H)
- **Requires allreduce**: Final step sums local outputs across all devices to get complete result
- **Single-core**: Current implementation uses single core (no atomic operations needed)

**Multi-Device Flow**
```
Device 0: Processes experts [0, 1, ..., E/D-1]     → partial_output_0 (E/D, T, H)
                                                    → local_sum → (T, H)
Device 1: Processes experts [E/D, E/D+1, ..., 2E/D-1] → partial_output_1 (E/D, T, H)
                                                      → local_sum → (T, H)
...
Device D-1: Processes experts [E-E/D, ..., E-1]    → partial_output_{D-1} (E/D, T, H)
                                                    → local_sum → (T, H)

Local Reduce: ttnn.sum(partial_output, dim=0) on each device
Final: allreduce(local_output_0, ..., local_output_{D-1}) → final_output (T, H)
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
num_routed_tokens, routed_tokens, routed_token_weights, token_idx_map = \
    ttnn.prepare_moe_routing_tensors(
        selected_experts, routing_weights, device_expert_mapping, num_experts
    )
```

### Step 2: Projection to Intermediate (gate_proj and up_proj)
```python
# Gate projection
gate_output = ttnn.projection_to_intermediate(
    hidden_states, routed_tokens, num_routed_tokens, gate_weights, top_k
)  # Shape: (E/D, T, H')

# Up projection
up_output = ttnn.projection_to_intermediate(
    hidden_states, routed_tokens, num_routed_tokens, up_weights, top_k
)  # Shape: (E/D, T, H')

# Combine: gate * up (element-wise)
combined = gate_output * up_output  # Shape: (E/D, T, H')
```

### Step 3: Projection to Output (down_proj)
```python
# Down projection with accumulation
partial_output = ttnn.projection_to_output(
    combined, token_idx_map, routed_tokens, num_routed_tokens, routed_token_weights,
    down_weights, num_tokens, top_k
)  # Shape: (E/D, T, H) - partial result per device
```

### Step 4: Local Reduce (sum across expert dimension)
```python
# Reduce expert dimension locally on each device before allreduce
# Sum across dim=0 (expert dimension) to combine contributions from all local experts
partial_output = ttnn.sum(partial_output, dim=0)  # Shape: (T, H) - per device
```

### Step 5: Allreduce (multi-device)
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
    + token_idx_map for global position mapping
         ↓
    ┌────────────────────────────────┐
    │                                │
    ↓                                ↓
[projection_to_intermediate]    [projection_to_intermediate]
   gate_proj (T→H')                up_proj (T→H')
   Output: (E/D, T, H')            Output: (E/D, T, H')
    ↓                                ↓
    └──────────→ gate * up ←─────────┘
                   ↓
         combined (E/D, T, H') - PADDED
                   ↓
        [projection_to_output]
           down_proj (H'→H)
         uses token_idx_map for positioning
                   ↓
         partial_output (E/D, T, H) per device
                   ↓
              [local sum]
         sum across expert dim (dim=0)
                   ↓
         local_output (T, H) per device
                   ↓
             [allreduce]
         sum across devices
                   ↓
         final_output (T, H) - COMPLETE
```

### Key Design Decisions

1. **Expert Parallelism**: Each device processes E/D experts independently
2. **Device-Local Routing**: Routing tensors filtered once, reused for all projections
3. **Padded Format**: Intermediate activations stored with padding to uniform shape (E/D, T, H')
4. **Token Index Mapping**: `token_idx_map` enables correct positioning of expert outputs in global token space
5. **Accumulation Pattern**: Output accumulates contributions from multiple experts per token
6. **Local Reduce First**: Sum across expert dimension (E/D→1) on each device before allreduce to reduce communication overhead
7. **Allreduce Required**: Final step sums partial results across devices to combine all expert contributions
