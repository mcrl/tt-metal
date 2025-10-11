# TT-Metal Version

- It is preferable to align the **TT-Metal versions** used in the matrix multiplication paper and the MoE (Mixture of Experts) work.  
  The matrix multiplication paper should also use the current version, and both can be published together once stable.  
- Currently, upgrading to **v0.63.0** is on hold because the MoE implementation does not produce correct results.  
- Once the MoE kernel (even a single-core version) works properly on the current version (v0.61?), we can retry the version upgrade.  
	- The TT implementation itself will not be used, so its stability is not a priority.  
- When upgrading the version, we must decide how to migrate newly implemented kernels.

---

# Tasks

1. **Debug the MoE merge issue**  
	- (First step) Rebuild the MoE implementation using only the reference API from the test code, and compare the results.

2. **Kernel renaming**  
	- `moe_expert_projection` → `projection_to_intermediate`  
	- `moe_down_projection` → `projection_to_output`

3. **Separate kernel stages**: Reader – Compute – Writer  

4. **Implement tile layout usage**

5. **Implement multi-core version**

---

# API Redesign Steps

- Perform **kernel renaming**.
- Update the **Plan document**, and create a **diff document** summarizing the changes.
	- API naming: More descriptive operation names
	- Expert mapping: Support both uniform and dynamic mapping via device-expert mapping tensor
	- Tensor shapes: Clarified device-local vs global dimensions

---

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

The device-expert mapping is **only used in `prepare_moe_routing_tensors`** to filter global routing information into device-local routing tables. Once the device-local routing tables are created, subsequent operations (`moe_up_projection` and `projection_to_output`) only use local expert indices and don't need to reference global expert IDs.

---

## `prepare_moe_routing_tensors`

**Input**
- **Routing weights**: `T × K`
	- TODO (Do not proceed until this TODO is resolved. Notify the user.)
- **Token–Expert mapping (Top-K routing indices)**: `T × K`
	- TODO (Do not proceed until this TODO is resolved. Notify the user.)
- **Device-Expert mapping**: `E / D`
	- Data type: `int32`
	- Contains global expert indices (0 to E-1) assigned to this device
	- `device_expert_mapping[local_expert_id]` returns the global expert index
	- Used to determine which tokens should be processed on this device

**Output**
- **Tokens per expert**: `E / D`
	- `tokens_per_expert[e]` represents the number of tokens routed to the locally *e*-th expert on the device.
- **Expert–Token routing table**: `E / D × T`
	- Each row contains `tokens_per_expert[e]` entries.
	- Indicates which tokens are assigned to expert *e*.
	- If token *t* is assigned to expert *e* and is the *k*-th token for that expert, then `expert_token_routing[e][k] = t`.
- **Expert–Token routing weight**: `E / D × T`
	- Similar structure as the routing table.
	- If token *t* is assigned to expert *e* as the *k*-th token, and its routing weight is *w*, then `expert_token_weight[e][k] = w`.

---

## `moe_up_projection`

**Input**
- **Input hidden state**: `T × H`
	- `input_hidden_state[t][:]` is the hidden state vector for token *t*.
- **Expert weights**: `E / D × H × H'`
	- `expert_weights[e][:][:]` contains the weights of the locally *e*-th expert on the device.
	- Indexed by local expert index (0 to E/D-1)
- **Expert–Token routing table**: `E / D × T`
	- Device-local routing table (already filtered by `prepare_moe_routing_tensors`)
- **Tokens per expert**: `E / D`
	- Device-local token counts

**Output**
- **Output hidden state**: `TK × H'`
	- Only the first `sum(tokens_per_expert)` rows are used.
	- The rows are filled sequentially by expert index.
	- For example, `tokens_per_expert[0]` corresponds to the tokens routed to the first expert on this device.
	- In the extreme case, a device can have up to `T × min(K, E / D)` tokens assigned (currently `K=8`, `E / D=16`, so `K` is used).

**Computation Method**
- For each local expert index `e` in [0, E/D):
	- Get token count: `T_e = tokens_per_expert[e]`
	- Get token indices: `expert_token_routing[e][:T_e]`
	- Get weights: `expert_weights[e]` (local expert index)
	- For each token `i` in [0, T_e):
		- If `expert_token_routing[e][i] = t`, output row index = `sum(tokens_per_expert[0:e]) + i`
		- Compute: `output_hidden_state[row][:] = input_hidden_state[t][:] @ expert_weight[e][:][:]`
		- Shape: `(1 × H') = (1 × H) @ (H × H')`

- (TODO) In the future, consecutive rows should be grouped into batched matrix multiplications for efficiency.

- Multiple reads and single writes occur, so race conditions are not a concern.

**Note**: Device-expert mapping is NOT needed here because:
- Routing tables are already device-local (filtered in `prepare_moe_routing_tensors`)
- Expert weights are indexed by local expert index
- No need to reference global expert indices during computation

---

## `projection_to_output`

**Input**
- **Input hidden state**: `T × K × H'`
	- Result tensor from `moe_up_projection`.
	- Only the first `sum(tokens_per_expert)` rows are used.
- **Tokens per expert**: `E / D`
	- Device-local token counts
- **Expert weights**: `E / D × H' × H`
	- `expert_weights[e][:][:]` contains weights for the locally *e*-th expert.
	- Indexed by local expert index (0 to E/D-1)
	- Note: Up projection is `H → H'`, and down projection is `H' → H`, so dimensions are transposed.
- **Expert–Token routing table**: `E / D × T`
	- Device-local routing table (already filtered by `prepare_moe_routing_tensors`)
	- Each row *e* uses only `tokens_per_expert[e]` entries.
- **Expert–Token routing weight**: `E / D × T`
	- Device-local routing weights
	- Each row *e* uses only `tokens_per_expert[e]` entries.

**Output**
- **Output hidden state**: `T × H`

**Computation Method**
- For each local expert index `e` in [0, E/D):
	- Get token count: `T_e = tokens_per_expert[e]`
	- Get token indices and weights: `expert_token_routing[e][:T_e]`, `expert_token_weight[e][:T_e]`
	- Get weights: `expert_weights[e]` (local expert index)
	- For each token `i` in [0, T_e):
		- If `expert_token_routing[e][i] = t` and `expert_token_weight[e][i] = w`
		- Input row index = `sum(tokens_per_expert[0:e]) + i`
		- Accumulate: `output_hidden_state[t][:] += (input_hidden_state[row][:] @ expert_weight[e][:][:]) * w`
		- Shape: `(1 × H) += (1 × H') @ (H' × H) × (1)`

- (TODO) In the future, investigate **data reuse optimization**.

- Multiple reads and multiple writes occur; **atomic operations** are required for multi-core implementations.

- Alternatively, instead of performing local reduction, consider keeping the output tensor as `T × K × H` and performing a **collapse-sum** along the K dimension.
	This may be necessary if atomic operations are unavailable.

**Note**: Device-expert mapping is NOT needed here because:
- Routing tables are already device-local (filtered in `prepare_moe_routing_tensors`)
- Expert weights are indexed by local expert index
- No need to reference global expert indices during computation