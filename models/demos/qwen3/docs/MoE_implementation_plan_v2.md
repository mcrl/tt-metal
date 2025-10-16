# MoE Implementation Plan V2

## Overview

This document describes the V2 API design for MoE operations, which simplifies the pipeline by:
1. **Separating scatter/gather logic** from compute operations
2. **Unifying projection operations** into a single BMM kernel
3. **Optimizing layout conversions** by keeping intermediate results in TILE_LAYOUT
4. **Simplifying accumulation** with dedicated local_reduce operation

## Implementation Status (2025-10-16)

### ✅ Completed
- **`scatter_moe_input`**: Fully implemented with V2 API
- **`local_reduce_moe_output`**: Fully implemented with V2 API
  - Multi-core parallelization (token-parallel approach)
  - Each core processes different output token rows
  - Always uses multi-core for optimal performance
- **Unified `moe_bmm`**: Single-core implementation

### ⚠️ Pending
- **Multi-core `moe_bmm`**
---

## API Call Sequence

```
1. scatter_moe_input           : Rearrange input by expert assignment
2. ttnn.to_layout              : ROW_MAJOR → TILE_LAYOUT
3. ttnn.experimental.moe_bmm   : Gate projection (BMM on TILE)
4. ttnn.experimental.moe_bmm   : Up projection (BMM on TILE)
5. silu & elementwise multiply : Activation (elementwise on TILE)
6. ttnn.experimental.moe_bmm   : Down projection (BMM on TILE)
7. ttnn.to_layout              : TILE_LAYOUT → ROW_MAJOR
8. local_reduce_moe_output     : Intra-device reduce and accumulation
9. inter-device reduce         : All-reduce across devices
```

---

## `scatter_moe_input`

**Purpose**: Rearranges input tokens based on expert assignments, gathering all tokens assigned to each local expert.

**Python API**
```python
output_hidden_state = ttnn.scatter_moe_input(
    input_hidden_state,   # (T, H) bfloat16 tensor - ROW_MAJOR layout
    num_routed_tokens,    # (E/D, 1) uint32 tensor
    routed_tokens,        # (E/D, T) uint32 tensor
    *,
    memory_config=None,
    queue_id=0
)
```

**Input**
- **input_hidden_state**: `(T, H)` bfloat16 tensor, ROW_MAJOR layout
	- Input token embeddings, replicated across all devices
	- `input_hidden_state[t, :]` is the hidden state vector for token t
	- T = number of tokens, H = hidden dimension
- **num_routed_tokens**: `(E/D, 1)` uint32 2D tensor, ROW_MAJOR layout
	- Device-local token counts from `prepare_moe_routing_tensors`
	- `num_routed_tokens[e, 0]` = number of tokens assigned to local expert e
	- Sharded across devices
- **routed_tokens**: `(E/D, T)` uint32 2D tensor, ROW_MAJOR layout
	- Device-local token indices from `prepare_moe_routing_tensors`
	- `routed_tokens[e, i]` = global token index for i-th token assigned to expert e
	- Valid entries: `routed_tokens[e, 0:num_routed_tokens[e, 0]]`
	- Padded entries (beyond `num_routed_tokens[e, 0]`) are ignored
	- Sharded across devices

**Output**
- **output_hidden_state**: `(E/D, T, H)` bfloat16 tensor, ROW_MAJOR layout
	- Scattered input organized by expert
	- Shape: E/D experts × T tokens × H hidden dimensions
	- For each local expert e, the number of assigned tokens is `t_e = num_routed_tokens[e, 0]`
	- Local expert e uses the output region `output[e, :t_e, :H]`
	- Remaining rows are zero-padded

**Computation**
For each local expert e in [0, E/D-1):
1. Read `t_e = num_routed_tokens[e, 0]` (number of tokens for this expert)
2. For each assigned token position i in [0, t_e):
   - Read global token index: `t_{e,i} = routed_tokens[e, i]`
   - Gather from input: `hidden_vec = input_hidden_state[t_{e,i}, :H]`
   - Write to output: `output_hidden_state[e, i, :H] = hidden_vec`
3. For remaining positions i in [t_e, T):
   - Write zero padding: `output_hidden_state[e, i, :H] = 0`

**Memory Layout**
```
output[0, :t_0, :]   ← tokens assigned to expert 0 (padded to T)
output[1, :t_1, :]   ← tokens assigned to expert 1 (padded to T)
...
output[E/D-1, :t_{E/D-1}, :] ← tokens assigned to expert E/D-1 (padded to T)
```

**Key Features**
- **Input-side scatter**: Rearranges tokens before computation
- **Expert-local organization**: All tokens for each expert are contiguous
- **Padding**: Uniform shape (E/D, T, H) with zero-padding after valid tokens
- **Enables efficient BMM**: Subsequent operations can process entire expert slices as batched matmuls

---

## `ttnn.experimental.moe_bmm`

**Purpose**: Unified batched matrix multiplication operation for MoE projections. Operates on already-scattered input in TILE_LAYOUT. Used for gate projection, up projection, and down projection.

**Python API**
```python
output = ttnn.experimental.moe_bmm(
    input_hidden_state,   # (E/D, T, H_in) bfloat16 tensor - TILE layout
    expert_weights,       # (E/D, H_in, H_out) bfloat16 tensor - TILE layout
    num_routed_tokens,    # (E/D, 1) uint32 tensor
    *,
    memory_config=None,
    queue_id=0
)
```

**Input**
- **input_hidden_state**: `(E/D, T, H_in)` bfloat16 tensor, TILE_LAYOUT
	- Scattered input from `scatter_moe_input` (after layout conversion)
	- Shape: E/D experts × T tokens × H_in input dimensions
	- For expert e, only first `num_routed_tokens[e, 0]` rows contain valid data
	- Already in TILE_LAYOUT for efficient BMM
	- For gate/up_proj: H_in = H (hidden dimension)
	- For down_proj: H_in = H' (intermediate dimension)
- **expert_weights**: `(E/D, H_in, H_out)` bfloat16 tensor, TILE_LAYOUT
	- Expert weight matrices, sharded across devices by expert dimension
	- `expert_weights[e, :, :]` contains weights for local expert e
	- For gate/up_proj: (H, H') - projects to intermediate dimension
	- For down_proj: (H', H) - projects back to hidden dimension
- **num_routed_tokens**: `(E/D, 1)` uint32 2D tensor, ROW_MAJOR layout
	- Device-local token counts from `prepare_moe_routing_tensors`
	- `num_routed_tokens[e, 0]` = number of tokens assigned to local expert e
	- Used for determining number of valid output tiles per expert

**Output**
- **output**: `(E/D, T, H_out)` bfloat16 tensor, TILE_LAYOUT
	- Projection outputs organized by expert
	- Shape: E/D experts × T tokens × H_out output dimensions
	- For expert e, only first `num_routed_tokens[e, 0]` rows contain valid data
	- Remaining rows are zero (due to padded input)
	- Stays in TILE_LAYOUT for subsequent operations
	- For gate/up_proj: H_out = H' (intermediate dimension)
	- For down_proj: H_out = H (hidden dimension)

**Computation**
For each local expert e in [0, E/D-1):
- Compute batched matmul: `output[e, :, :] = input[e, :, :] @ expert_weights[e, :, :]`
- This multiplies (T × H_in) @ (H_in × H_out) → (T × H_out)
- Only first `num_routed_tokens[e, 0]` rows produce non-zero results
- Remaining rows produce zeros (due to zero-padded input)

**Parallelization Strategy: Single-Core (Current Implementation)**

The current implementation uses a **single Tensix core** to sequentially process all experts:

1. **Sequential expert processing**:
   - Loop through each expert e in [0, E/D-1)
   - For each expert, perform full matrix multiplication

2. **Per-expert computation**:
   - For expert e with `t_e = num_routed_tokens[e, 0]` tokens:
   - Compute: `output[e, :t_e, :] = input[e, :t_e, :] @ expert_weights[e, :, :]`
   - This multiplies (t_e × H_in) @ (H_in × H_out) → (t_e × H_out)

3. **Tile-level operations**:
   - Number of token tiles: `num_token_tiles[e] = ceil(t_e / 32)`
   - Number of input dimension tiles: `num_input_tiles = ceil(H_in / 32)`
   - Number of output dimension tiles: `num_output_tiles = ceil(H_out / 32)`
   - For each output tile `(token_tile_idx, output_dim_tile_idx)`:
     - Accumulate across input dimension tiles: `sum over k in [0, num_input_tiles)`
     - Load input tile: `input[e, token_tile_idx*32:(token_tile_idx+1)*32, k*32:(k+1)*32]`
     - Load weight tile: `expert_weights[e, k*32:(k+1)*32, output_dim_tile_idx*32:(output_dim_tile_idx+1)*32]`
     - Compute: `output_tile += matmul_tiles(input_tile, weight_tile)`
     - Write: `output[e, token_tile_idx*32:(token_tile_idx+1)*32, output_dim_tile_idx*32:(output_dim_tile_idx+1)*32] = output_tile`

**Future Multi-Core Optimization**

For future multi-core implementation, use **output-stationary parallelization**:

1. **Calculate total tiles across all experts**:
   - For expert e: `tiles[e] = ceil(num_routed_tokens[e, 0] / 32) * ceil(H_out / 32)`
   - Total tiles: `total_tiles = sum(tiles[0:E/D])`

2. **Distribute tiles across 64 Tensix cores**:
   - Assign each output tile to a core
   - Use round-robin or block distribution
   - Each core independently computes its assigned output tiles

3. **Core-local computation**:
   - Each core loads input tiles, weight tiles for its assigned output tiles
   - Performs matmul and writes results
   - No inter-core communication needed during compute

**Usage**
- **Gate projection**: `gate_output = moe_bmm(scattered_input, gate_weights, num_routed_tokens)`
- **Up projection**: `up_output = moe_bmm(scattered_input, up_weights, num_routed_tokens)`
- **Down projection**: `down_output = moe_bmm(combined_activations, down_weights, num_routed_tokens)`

All three projections use the same unified operation with different weight matrices.

**Key Features**
- **Unified operation**: Single implementation for all MoE projections
- **Pure BMM**: No scatter/gather logic embedded
- **TILE_LAYOUT native**: Operates efficiently on tile layout
- **Simple interface**: Only requires input, weights, and token counts
- **Flexible dimensions**: Works with any H_in, H_out dimensions

---

## `local_reduce_moe_output`

**Purpose**: Performs intra-device reduction by gathering expert outputs back to token order and applying routing weights.

**Python API**
```python
output_hidden_state = ttnn.local_reduce_moe_output(
    input_hidden_state,     # (E/D, T, H) bfloat16 tensor - ROW_MAJOR layout
    token_idx_map,          # (E/D, T) uint32 tensor
    routed_token_weights,   # (E/D, T) bfloat16 tensor
    num_routed_tokens,      # (E/D, 1) uint32 tensor
    num_tokens,             # scalar int - T
    *,
    memory_config=None,
    queue_id=0
)
```

**Input**
- **input_hidden_state**: `(E/D, T, H)` bfloat16 tensor, ROW_MAJOR layout
	- Expert outputs from `moe_bmm` (after layout conversion)
	- Shape: E/D experts × T tokens × H hidden dimensions
	- For expert e, only first `num_routed_tokens[e, 0]` rows contain valid data
	- Organized by expert (not yet gathered by token)
- **token_idx_map**: `(E/D, T)` uint32 tensor, ROW_MAJOR layout
	- Mapping from expert-local position to global token index
	- From `prepare_moe_routing_tensors`
	- `token_idx_map[e, i]` = global token index for i-th position of expert e
	- Valid entries: `token_idx_map[e, 0:num_routed_tokens[e, 0]]`
	- Padded entries are ignored
- **routed_token_weights**: `(E/D, T)` bfloat16 tensor, ROW_MAJOR layout
	- Routing weights for each expert-token assignment
	- From `prepare_moe_routing_tensors`
	- `routed_token_weights[e, i]` = routing weight for i-th token of expert e
	- Valid entries: `routed_token_weights[e, 0:num_routed_tokens[e, 0]]`
- **num_routed_tokens**: `(E/D, 1)` uint32 tensor, ROW_MAJOR layout
	- Device-local token counts
	- `num_routed_tokens[e, 0]` = number of valid entries for expert e
- **num_tokens**: scalar integer
	- Total number of tokens (T)

**Output**
- **output_hidden_state**: `(T, H)` bfloat16 tensor, ROW_MAJOR layout
	- Final output for all tokens on this device
	- Shape: T tokens × H hidden dimensions
	- Initialized to zeros, then accumulated
	- Contains weighted sum of all expert contributions per token

**Computation**

For each global token index t in [0, T):
1. Initialize: `output[t, :] = 0`
2. For each local expert e in [0, E/D-1):
   - Read `t_e = num_routed_tokens[e, 0]`
   - For each expert-local position i in [0, t_e):
     - If `token_idx_map[e, i] == t`:
       - Read hidden state: `hidden = input_hidden_state[e, i, :]`
       - Read routing weight: `weight = routed_token_weights[e, i]`
       - Accumulate: `output[t, :] += hidden * weight`

**Mathematical Formulation**

For each global token index t:
```
output[t, :H] = sum over all (e, i) where token_idx_map[e, i] = t:
                  input_hidden_state[e, i, :H] * routed_token_weights[e, i]
```

**Parallelization Strategy: Token-Stationary**

1. **Outer loop over tokens** (t from 0 to T-1):
   - Each token can be processed independently
   - Enables easy multi-core parallelization (distribute T tokens across cores)

2. **Inner loop over hidden dimensions** (h from 0 to H-1):
   - For each token, accumulate contributions from all experts
   - Can be vectorized for efficiency

**Pseudo-code**:
```python
# Parallel over tokens (distribute across cores)
for t in range(T):
    accumulator[0:H] = 0.0
    
    # Sequential over experts (typically E/D = 16, small)
    for e in range(E/D):
        t_e = num_routed_tokens[e, 0]
        
        # Sequential over expert's tokens
        for i in range(t_e):
            if token_idx_map[e, i] == t:
                weight = routed_token_weights[e, i]
                # Vectorized over H dimension
                accumulator[:] += input_hidden_state[e, i, :] * weight
                break  # Each token appears at most once per expert
    
    output[t, :] = accumulator[:]
```

**Optimization Notes**:
- Each token t appears at most K times across all experts (where K = top_k, typically 4-8)
- The inner expert loop (E/D iterations) is small (typically 16)
- Most time is spent in the vectorized accumulation over H dimension
- Multi-core implementation: Each core processes an independent subset of T tokens
- Token distribution uses `split_work_to_cores()` for balanced load

**Key Features**
- **Token-order restoration**: Converts from expert-organized to token-organized
- **Routing weight application**: Applies weights during accumulation
- **Intra-device reduction**: Sums contributions from all local experts
- **Partial results**: Output still needs inter-device allreduce to combine experts from all devices

---

## Complete MoE Pipeline (V2)

### Step 0: Prepare Routing (once per forward pass)
```python
# Same as V1 - prepares device-local routing information
num_routed_tokens, routed_tokens, routed_token_weights, token_idx_map = \
    ttnn.prepare_moe_routing_tensors(
        selected_experts, routing_weights, device_expert_mapping, num_experts
    )
```

### Step 1: Scatter Input by Expert
```python
scattered_input = ttnn.scatter_moe_input(
    hidden_states,      # (T, H) ROW_MAJOR
    num_routed_tokens,  # (E/D, 1)
    routed_tokens       # (E/D, T)
)  # Shape: (E/D, T, H) ROW_MAJOR
```

### Step 2: Convert to TILE Layout
```python
scattered_input_tile = ttnn.to_layout(
    scattered_input, ttnn.TILE_LAYOUT
)  # Shape: (E/D, T, H) TILE
```

### Step 3: Gate Projection
```python
gate_output = ttnn.experimental.moe_bmm(
    scattered_input_tile,  # (E/D, T, H) TILE
    gate_weights,          # (E/D, H, H') TILE
    num_routed_tokens      # (E/D, 1)
)  # Shape: (E/D, T, H') TILE
```

### Step 4: Up Projection
```python
up_output = ttnn.experimental.moe_bmm(
    scattered_input_tile,  # (E/D, T, H) TILE
    up_weights,            # (E/D, H, H') TILE
    num_routed_tokens      # (E/D, 1)
)  # Shape: (E/D, T, H') TILE
```

### Step 5: SiLU and Element-wise Multiply
```python
gate_activated = ttnn.silu(gate_output)  # (E/D, T, H') TILE
combined = ttnn.mul(gate_activated, up_output)  # (E/D, T, H') TILE
```

### Step 6: Down Projection
```python
down_output = ttnn.experimental.moe_bmm(
    combined,           # (E/D, T, H') TILE
    down_weights,       # (E/D, H', H) TILE
    num_routed_tokens   # (E/D, 1)
)  # Shape: (E/D, T, H) TILE
```

### Step 7: Convert to ROW_MAJOR Layout
```python
down_output_rm = ttnn.to_layout(
    down_output, ttnn.ROW_MAJOR_LAYOUT
)  # Shape: (E/D, T, H) ROW_MAJOR
```

### Step 8: Local Reduce (Intra-device)
```python
local_output = ttnn.local_reduce_moe_output(
    down_output_rm,        # (E/D, T, H) ROW_MAJOR
    token_idx_map,         # (E/D, T)
    routed_token_weights,  # (E/D, T)
    num_routed_tokens,     # (E/D, 1)
    num_tokens             # scalar T
)  # Shape: (T, H) ROW_MAJOR - per device
```

### Step 9: Inter-device Reduce (Allreduce)
```python
final_output = ttnn.all_reduce(
    local_output,   # (T, H) ROW_MAJOR
    mesh_device,
    math_op=ttnn.ReduceType.Sum
)  # Shape: (T, H) ROW_MAJOR - complete result
```

---

## Data Flow Summary (V2)

```
Global Routing Info (T, K)
         ↓
    [prepare_moe_routing_tensors]
         ↓
Device-Local Routing Tables
    num_routed_tokens, routed_tokens, token_idx_map, routed_token_weights
         ↓
         ↓ input: (T, H) ROW_MAJOR
         ↓
    [scatter_moe_input]
         ↓
    (E/D, T, H) ROW_MAJOR - organized by expert
         ↓
    [to_layout: ROW_MAJOR → TILE]
         ↓
    (E/D, T, H) TILE
         ↓
    ┌────────────────────────────────┐
    │                                │
    ↓                                ↓
[ttnn.experimental.moe_bmm]    [ttnn.experimental.moe_bmm]
   gate_proj                        up_proj
   (E/D, T, H')                     (E/D, T, H')
    ↓                                ↓
    └──────→ silu & mul ←────────────┘
                ↓
         combined (E/D, T, H') TILE
                ↓
   [ttnn.experimental.moe_bmm]
         down_proj
                ↓
         (E/D, T, H) TILE
                ↓
    [to_layout: TILE → ROW_MAJOR]
                ↓
         (E/D, T, H) ROW_MAJOR
                ↓
    [local_reduce_moe_output]
    (gathers to token order + applies weights)
                ↓
         (T, H) ROW_MAJOR - per device
                ↓
         [allreduce]
    (sums across devices)
                ↓
         (T, H) ROW_MAJOR - COMPLETE
```

---

## Implementation Recommendations

### 1. **Optimize Layout Conversions**
- Minimize conversions by staying in TILE_LAYOUT as long as possible
- Consider if any operations can work directly on TILE

### 2. **Multi-Core Parallelization** (Future Work)
- **Current**: Single-core implementation, processes experts sequentially
- **Future BMM operations**: Distribute output tiles across 64 cores
  - Calculate total output tiles across all experts
  - Assign tiles to cores using round-robin or block distribution
  - Each core independently computes its assigned tiles
- **Future Reduce operation**: Distribute output tokens across 64 cores
- Balance work to avoid stragglers

### 3. **Memory Management**
- Pre-allocate output tensors to avoid runtime allocation
- Reuse buffers where possible (e.g., gate/up can share output buffer after consumed)

---