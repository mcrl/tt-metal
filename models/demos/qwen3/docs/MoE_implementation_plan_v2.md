# MoE Implementation Plan V2

## Overview

This document describes the V2 API design for MoE operations, which simplifies the pipeline by:
1. **Separating scatter/gather logic** from compute operations
2. **Unifying projection operations** into a single BMM kernel
3. **Optimizing layout conversions** by keeping intermediate results in TILE_LAYOUT
4. **Simplifying accumulation** with dedicated local_reduce operation

---

## API Call Sequence

```
1. scatter_moe_input           : Rearrange input by expert assignment
2. ttnn.to_layout              : ROW_MAJOR → TILE_LAYOUT
3. projection_to_intermediate  : Gate projection (BMM on TILE)
4. projection_to_intermediate  : Up projection (BMM on TILE)
5. silu & elementwise multiply : Activation (elementwise on TILE)
6. projection_to_output        : Down projection (BMM on TILE)
7. ttnn.to_layout              : TILE_LAYOUT → ROW_MAJOR
8. local_reduce_moe_output     : Intra-device reduce and accumulation
9. inter-device reduce         : Allreduce across devices
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

## `projection_to_intermediate` (V2)

**Purpose**: Performs batched matrix multiplication for MoE projections. Operates on already-scattered input in TILE_LAYOUT.

**Python API**
```python
output = ttnn.projection_to_intermediate(
    input_hidden_state,   # (E/D, T, H) bfloat16 tensor - TILE layout
    num_routed_tokens,    # (E/D, 1) uint32 tensor
    expert_weights,       # (E/D, H, H') bfloat16 tensor - TILE layout
    *,
    memory_config=None,
    queue_id=0
)
```

**Input**
- **input_hidden_state**: `(E/D, T, H)` bfloat16 tensor, TILE_LAYOUT
	- Scattered input from `scatter_moe_input` (after layout conversion)
	- Shape: E/D experts × T tokens × H hidden dimensions
	- For expert e, only first `num_routed_tokens[e, 0]` rows contain valid data
	- Already in TILE_LAYOUT for efficient BMM
- **num_routed_tokens**: `(E/D, 1)` uint32 2D tensor, ROW_MAJOR layout
	- Device-local token counts
	- Used for determining number of valid output tiles per expert
- **expert_weights**: `(E/D, H, H')` bfloat16 tensor, TILE_LAYOUT
	- Expert weight matrices, sharded across devices by expert dimension
	- `expert_weights[e, :, :]` contains weights for local expert e
	- H = input hidden dimension, H' = intermediate dimension

**Output**
- **output**: `(E/D, T, H')` bfloat16 tensor, TILE_LAYOUT
	- Projection outputs organized by expert
	- Shape: E/D experts × T tokens × H' intermediate dimensions
	- For expert e, only first `num_routed_tokens[e, 0]` rows contain valid data
	- Remaining rows are zero (due to padded input)
	- Stays in TILE_LAYOUT for subsequent operations

**Computation**
For each local expert e in [0, E/D-1):
- Compute batched matmul: `output[e, :, :] = input[e, :, :] @ expert_weights[e, :, :]`
- This multiplies (T × H) @ (H × H') → (T × H')
- Only first `num_routed_tokens[e, 0]` rows produce non-zero results

**Parallelization Strategy: Output-Stationary**

1. **Calculate tiles per expert**:
   - For expert e with `t_e = num_routed_tokens[e, 0]` tokens:
   - Number of token tiles: `num_token_tiles[e] = ceil(t_e / 32)`
   - Number of output dimension tiles: `num_output_tiles = ceil(H' / 32)`
   - Total tiles for expert e: `tiles[e] = num_token_tiles[e] * num_output_tiles`

2. **Total device tiles**: `total_tiles = sum(tiles[0:E/D])`

3. **Core distribution**:
   - Distribute `total_tiles` across 64 Tensix cores
   - Each core computes a subset of output tiles
   - Cores process tiles in a round-robin or block distribution

4. **Per-tile computation**:
   - To compute output tile at `output[e, i:i+32, h':h'+32]`:
   - Load input tiles: `input[e, i:i+32, 0:H]` (multiple tiles across H dimension)
   - Load weight tiles: `expert_weights[e, 0:H, h':h'+32]` (multiple tiles across H dimension)
   - Compute: `output_tile = matmul_tiles(input_tiles, weight_tiles)`
   - Write: `output[e, i:i+32, h':h'+32] = output_tile`

**Usage**
- Used for **gate_proj** and **up_proj** in MoE layers
- Both projections use identical logic (only weights differ)
- Can be called twice or potentially unified into single `moe_bmm` operation

**Changes from V1**
- ✅ No longer needs `routed_tokens` (scattering already done)
- ✅ Input already organized by expert (no gathering needed)
- ✅ Operates on TILE_LAYOUT (no layout conversion during compute)
- ✅ Simpler implementation (pure BMM without gather logic)

---

## `projection_to_output` (V2)

**Purpose**: Performs down projection for MoE. Produces pre-gathered output (still organized by expert).

**Python API**
```python
output = ttnn.projection_to_output(
    input_hidden_state,   # (E/D, T, H') bfloat16 tensor - TILE layout
    num_routed_tokens,    # (E/D, 1) uint32 tensor
    expert_weights,       # (E/D, H', H) bfloat16 tensor - TILE layout
    *,
    memory_config=None,
    queue_id=0
)
```

**Input**
- **input_hidden_state**: `(E/D, T, H')` bfloat16 tensor, TILE_LAYOUT
	- Combined gate × up activations (after SiLU and elementwise multiply)
	- Shape: E/D experts × T tokens × H' intermediate dimensions
	- For expert e, only first `num_routed_tokens[e, 0]` rows contain valid data
	- In TILE_LAYOUT for efficient BMM
- **num_routed_tokens**: `(E/D, 1)` uint32 2D tensor, ROW_MAJOR layout
	- Device-local token counts
	- Used for determining number of valid output tiles per expert
- **expert_weights**: `(E/D, H', H)` bfloat16 tensor, TILE_LAYOUT
	- Down projection weight matrices, sharded across devices
	- `expert_weights[e, :, :]` contains weights for local expert e
	- H' = intermediate dimension, H = output hidden dimension

**Output**
- **output**: `(E/D, T, H)` bfloat16 tensor, TILE_LAYOUT
	- Projection outputs organized by expert (pre-gathered)
	- Shape: E/D experts × T tokens × H hidden dimensions
	- For expert e, only first `num_routed_tokens[e, 0]` rows contain valid data
	- Remains in TILE_LAYOUT for layout conversion
	- **Note**: Does NOT apply routing weights or accumulate across experts

**Computation**
For each local expert e in [0, E/D-1):
- Compute batched matmul: `output[e, :, :] = input[e, :, :] @ expert_weights[e, :, :]`
- This multiplies (T × H') @ (H' × H) → (T × H)
- Only first `num_routed_tokens[e, 0]` rows produce non-zero results

**Parallelization Strategy: Output-Stationary**

Same strategy as `projection_to_intermediate`:

1. **Calculate tiles per expert**:
   - For expert e with `t_e = num_routed_tokens[e, 0]` tokens:
   - Number of token tiles: `num_token_tiles[e] = ceil(t_e / 32)`
   - Number of output dimension tiles: `num_output_tiles = ceil(H / 32)`
   - Total tiles for expert e: `tiles[e] = num_token_tiles[e] * num_output_tiles`

2. **Total device tiles**: `total_tiles = sum(tiles[0:E/D])`

3. **Core distribution**: Distribute across 64 Tensix cores

4. **Per-tile computation**: Standard tile matmul

**Changes from V1**
- ✅ No longer needs `routed_tokens` (already scattered)
- ✅ No longer needs `token_idx_map` (gathering deferred to reduce stage)
- ✅ No longer needs `routed_token_weights` (weighting deferred to reduce stage)
- ✅ No longer performs accumulation (deferred to `local_reduce_moe_output`)
- ✅ Simpler implementation (pure BMM without scatter-gather logic)

**Unification Note**
- `projection_to_intermediate` and `projection_to_output` have **identical logic**
- Both perform: `output[e] = input[e] @ weights[e]` for each expert e
- Could be unified into single `moe_bmm` operation with different weight parameters

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
	- Expert outputs from `projection_to_output` (after layout conversion)
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

**Recommended Implementation** (easier for parallelization):

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
- Multi-core: Assign each core a subset of T tokens to process

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
gate_output = ttnn.projection_to_intermediate(
    scattered_input_tile,  # (E/D, T, H) TILE
    num_routed_tokens,     # (E/D, 1)
    gate_weights           # (E/D, H, H') TILE
)  # Shape: (E/D, T, H') TILE
```

### Step 4: Up Projection
```python
up_output = ttnn.projection_to_intermediate(
    scattered_input_tile,  # (E/D, T, H) TILE
    num_routed_tokens,     # (E/D, 1)
    up_weights             # (E/D, H, H') TILE
)  # Shape: (E/D, T, H') TILE
```

### Step 5: SiLU and Element-wise Multiply
```python
gate_activated = ttnn.silu(gate_output)  # (E/D, T, H') TILE
combined = ttnn.mul(gate_activated, up_output)  # (E/D, T, H') TILE
```

### Step 6: Down Projection
```python
down_output = ttnn.projection_to_output(
    combined,           # (E/D, T, H') TILE
    num_routed_tokens,  # (E/D, 1)
    down_weights        # (E/D, H', H) TILE
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
[projection_to_intermediate]    [projection_to_intermediate]
   gate_proj                        up_proj
   (E/D, T, H')                     (E/D, T, H')
    ↓                                ↓
    └──────→ silu & mul ←────────────┘
                ↓
         combined (E/D, T, H') TILE
                ↓
      [projection_to_output]
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

## Key Improvements in V2

### 1. **Cleaner Separation of Concerns**
- **V1**: Scatter/gather logic embedded in projection operations
- **V2**: Dedicated `scatter_moe_input` and `local_reduce_moe_output` operations
- **Benefit**: Simpler projection kernels, easier to optimize

### 2. **Unified Projection Operations**
- **V1**: `projection_to_intermediate` and `projection_to_output` had different interfaces
- **V2**: Both are identical pure BMM operations (can unify into `moe_bmm`)
- **Benefit**: Code reuse, single kernel to optimize

### 3. **Reduced Layout Conversions**
- **V1**: Multiple conversions between ROW_MAJOR and TILE throughout pipeline
- **V2**: Convert to TILE once, stay in TILE through all BMMs, convert back once
- **Benefit**: 4 fewer layout conversions, significant performance improvement

### 4. **Optimized Parallelization**
- **V1**: Complex gather/scatter in kernels made parallelization difficult
- **V2**: 
  - Projections: Output-stationary parallelization (simple tile distribution)
  - Reduce: Token-stationary parallelization (easy to distribute across cores)
- **Benefit**: Better utilization of 64 Tensix cores

### 5. **Deferred Accumulation**
- **V1**: `projection_to_output` performed accumulation during compute
- **V2**: Accumulation deferred to `local_reduce_moe_output`
- **Benefit**: BMM kernels don't need atomic operations or special accumulation logic

### 6. **Clearer Tensor Naming**
- **V1**: Mixed usage of `routed_tokens`, `token_idx_map` in different contexts
- **V2**: Clear separation:
  - `routed_tokens`: Used only for scatter (which token to gather)
  - `token_idx_map`: Used only for reduce (where to accumulate)
- **Benefit**: Less confusion, easier to understand data flow

---

## Comparison: V1 vs V2

| Aspect | V1 | V2 |
|--------|----|----|
| **Scatter/Gather** | Embedded in projections | Dedicated operations |
| **Projection Operations** | Different interfaces | Unified (can use single `moe_bmm`) |
| **Layout Conversions** | ~6 conversions | 2 conversions |
| **Compute Layout** | Mixed ROW_MAJOR/TILE | Stay in TILE |
| **Accumulation** | During projection | Separate reduce operation |
| **Parallelization** | Complex (gather + compute) | Simple (pure BMM or pure reduce) |
| **Code Complexity** | High (interleaved logic) | Low (separated concerns) |
| **Performance** | Baseline | Expected improvement from layout reduction |

---

## Implementation Recommendations

### 1. **Unify Projection Operations**
Consider implementing a single `moe_bmm` operation:
```python
output = ttnn.moe_bmm(
    input,              # (E/D, T, H_in) TILE
    weights,            # (E/D, H_in, H_out) TILE
    num_routed_tokens,  # (E/D, 1)
)  # Returns: (E/D, T, H_out) TILE
```
- Use for gate_proj, up_proj, and down_proj
- Only weights differ between calls

### 2. **Optimize Layout Conversions**
- Minimize conversions by staying in TILE_LAYOUT as long as possible
- Consider if any operations can work directly on TILE (e.g., scatter/reduce)

### 3. **Multi-Core Parallelization**
- **BMM operations**: Distribute output tiles across 64 cores
- **Reduce operation**: Distribute output tokens across 64 cores
- Balance work to avoid stragglers

### 4. **Memory Management**
- Pre-allocate output tensors to avoid runtime allocation
- Reuse buffers where possible (e.g., gate/up can share output buffer after consumed)

### 5. **Profiling Points**
Key operations to profile:
1. `scatter_moe_input` (gather overhead)
2. Layout conversions (ROW_MAJOR ↔ TILE)
3. BMM operations (compute time)
4. `local_reduce_moe_output` (accumulation overhead)
5. Allreduce (communication time)

---

## Migration Path from V1 to V2

For existing code using V1 API:

1. **Add scatter step** before projections:
   ```python
   scattered = ttnn.scatter_moe_input(hidden_states, num_routed, routed_tokens)
   ```

2. **Move layout conversion** outside projection loop:
   ```python
   scattered_tile = ttnn.to_layout(scattered, TILE_LAYOUT)
   ```

3. **Update projection calls** (remove routed_tokens parameter):
   ```python
   # V1: projection_to_intermediate(hidden_states, routed_tokens, num_routed, weights, top_k)
   # V2: projection_to_intermediate(scattered_tile, num_routed, weights)
   ```

4. **Replace local sum + allreduce** with reduce operation:
   ```python
   # V1: local = ttnn.sum(output, dim=0); final = ttnn.all_reduce(local)
   # V2: local = ttnn.local_reduce_moe_output(...); final = ttnn.all_reduce(local)
   ```

5. **Remove intermediate layout conversions** between operations
