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
	- Major change: **Remove dynamic expert mapping** — use **equally distributed mapping** instead.

---

# APIs

### Expert Parallelization: Equally Distributed

- Device 0 → Experts 0–15  
- Device 1 → Experts 16–31  
- …  
- Device 7 → Experts 112–127  

Example:  
- On **Device 1**, the locally 0th expert corresponds to **Expert 16** (global index 16).  
- On **Device 7**, the locally 2nd expert corresponds to **Expert 114** (global index 114).

---

## `prepare_moe_routing_tensors`

**Input**
- **Routing weights**: `T × K`  
	- TODO (Do not proceed until this TODO is resolved. Notify the user.)  
- **Token–Expert mapping (Top-K routing indices)**: `T × K`  
	- TODO (Do not proceed until this TODO is resolved. Notify the user.)  

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
- **Expert–Token routing table**: `E / D × T`  
- **Tokens per expert**: `E / D`  

**Output**
- **Output hidden state**: `TK × H'`  
	- Only the first `sum(tokens_per_expert)` rows are used.  
	- The rows are filled sequentially by expert index.  
	- For example, `tokens_per_expert[0]` corresponds to the tokens routed to the first expert on this device.  
	- In the extreme case, a device can have up to `T × min(K, E / D)` tokens assigned (currently `K=8`, `E / D=16`, so `K` is used).  

**Computation Method**
- If `expert_token_routing[e][i] = t`,  
	the output row index = `sum(tokens_per_expert[0:e]) + i`.  
	Then:  
	```
	output_hidden_state[row][:] = input_hidden_state[t][:] @ expert_weight[e][:][:]
	```
	That is, `(1 × H') = (1 × H) @ (H × H')`.  

- (TODO) In the future, consecutive rows should be grouped into batched matrix multiplications for efficiency.

- Multiple reads and single writes occur, so race conditions are not a concern.

---

## `projection_to_output`

**Input**
- **Input hidden state**: `T × K × H'`  
	- Result tensor from `moe_up_projection`.  
	- Only the first `sum(tokens_per_expert)` rows are used.  
- **Tokens per expert**: `E / D`  
- **Expert weights**: `E / D × H' × H`  
	- `expert_weights[e][:][:]` contains weights for the locally *e*-th expert.  
	- Note: Up projection is `H → H'`, and down projection is `H' → H`, so dimensions are transposed.  
- **Expert–Token routing table**: `E / D × T`  
	- Each row *e* uses only `tokens_per_expert[e]` entries.  
- **Expert–Token routing weight**: `E / D × T`  
	- Each row *e* uses only `tokens_per_expert[e]` entries.  

**Output**
- **Output hidden state**: `T × H`

**Computation Method**
- If `expert_token_routing[e][i] = t` and `expert_token_weight[e][i] = w`,  
	then the input row index = `sum(tokens_per_expert[0:e]) + i`.  
	The result is accumulated as:  
	```
	output_hidden_state[t][:] += (input_hidden_state[row][:] @ expert_weight[e][:][:]) * w
	```
	Shape-wise: `(1 × H) += (1 × H') @ (H' × H) × (1)`.  

- (TODO) In the future, investigate **data reuse optimization**.

- Multiple reads and multiple writes occur; **atomic operations** are required for multi-core implementations.

- Alternatively, instead of performing local reduction, consider keeping the output tensor as `T × K × H` and performing a **collapse-sum** along the K dimension.  
	This may be necessary if atomic operations are unavailable.