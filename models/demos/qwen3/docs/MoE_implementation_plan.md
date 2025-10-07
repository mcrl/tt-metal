# MoE Implementation Plan for TT Kernel

## Implementation Structure

### Model Integration
- **Main usage**: `models/demos/qwen3/tt/moe.py`
- Will call the TTNN MoE operations implemented below

### Test Files
- **Location**: `models/demos/qwen3/tests/`
- Test files for MoE functionality

### TTNN API and Kernels
- **Location**: `ttnn/cpp/ttnn/operations/experimental/moe/`
- Subdirectories for different MoE operations
- Contains C++ operations and device kernels


## Terminology
- **B**: Batch size. For prefill phase, B = 32
- **S**: Sequence length. For prefill phase, S = 128
- **T**: Number of input tokens. T = B * S. For prefill phase, T = 32 * 128 = 4096
- **E**: Total number of experts. 128 for Qwen3-30B-A3B model
- **D**: Number of devices. We'll use 8 for 1x8 mesh device configuration
- **K**: Number of experts selected per token. 8 for Qwen3-30B-A3B model
- **H**: Hidden dimension. 2048 for Qwen3-30B-A3B model
- **H'**: Expert hidden dimension (MoE intermediate size). 768 for Qwen3-30B-A3B model
- **T_e**: Number of tokens routed to expert e
- **T_d**: Number of token-expert pairs routed to device d. T_d = sum of T_e for all experts on device d. Maximum: K * T

## Expert Parallelism
- Using expert parallelism strategy
- Experts per device: E / D = 128 / 8 = 16

## Input Hidden State
- Input hidden state shape: T x H
- Input hidden state is replicated across all devices

## Selected Expert Indices (unused)
- Shape: T x K per device
- Replicated across all devices
- [t, k]: Global expert index (0 to E-1) for token t's k-th selected expert

## Routing Weights (unused)
- Shape: T x K per device
- Replicated across all devices
- [t, k]: Mapping weight of token t's k-th expert

## Token-Expert Mapping (unused)
- Shape: T x E
- [t, e]: Routing weight of token t to expert e
- Zero if token t is not routed to expert e
- Derived from selected expert indices and routing weights

## Expert-Token Mapping (unused)
- Shape: E x T (transpose of Token-Expert Mapping)
- [e, t]: Routing weight of expert e for token t
- Zero if expert e is not assigned to token t
- Used as input for efficient expert-parallel computation

## Num Routed Tokens (Input Tensor)
- Shape: E (one value per expert)
- [e]: Number of tokens routed to expert e (T_e)
- Used to determine iteration bounds for each expert

## Routed Tokens (Input Tensor)
- Shape: E x T
- [e, i]: Index of i-th token routed to expert e
- Variable length per expert, padded to T for rectangular tensor shape
- Entries beyond num_routed_tokens[e] are padding/invalid
- Used for efficient token lookup during expert computation

## Routed Token Weights (Input Tensor)
- Shape: E x T
- [e, i]: Routing weight for i-th token routed to expert e
- Corresponds to the routing weights in Routed Tokens tensor
- Entries beyond num_routed_tokens[e] are padding/invalid
- Used to apply routing weights during final accumulation

## Device-Expert Mapping Tensor (Input Tensor)
- Shape: (E/D,) per device (16 for our configuration)
- Data type: int32
- Contains global expert indices assigned to this device
- **Default uniform partitioning**:
  - Device 0: [0, 1, 2, ..., 15]
  - Device 1: [16, 17, 18, ..., 31]
  - ...
  - Device 7: [112, 113, 114, ..., 127]
- **Future: Custom/load-balanced partitioning**:
  - Device 0: [0, 5, 12, 18, ...]  (non-contiguous expert indices)
  - Device 1: [1, 7, 13, 19, ...]
  - Enables dynamic expert allocation and load balancing
- Used to map local expert index (0 to E/D-1) to global expert index (0 to E-1)
- Sharded across devices (each device has different E/D experts)

## Expert Weights 1 (gate_proj)
- Total shape: E x H x H'
- Each device has: (E/D) x H x H' = 16 x 2048 x 768

## Expert Weights 2 (up_proj)
- Total shape: E x H x H'
- Each device has: (E/D) x H x H' = 16 x 2048 x 768

## Expert Weights 3 (down_proj)
- Total shape: E x H' x H
- Each device has: (E/D) x H' x H = 16 x 768 x 2048

## Output Tensors (Pre-allocated)

### Gate Projection Output Tensor
- Shape: (T * K) x H'
- Maximum size: (T * 8) x 768
- Pre-allocated tensor for gate projection outputs
- Stores gate projection outputs for all token-expert pairs on device

### Up Projection Output Tensor
- Shape: (T * K) x H'
- Maximum size: (T * 8) x 768
- Pre-allocated tensor for up projection outputs
- Stores up projection outputs for all token-expert pairs on device

## Output
- Shape: T x H
- Final output after aggregating all expert computations

## Kernel Design Plan

### Overview: Batched Matrix Multiplications
Steps 1, 2, and 4 are batched matrix multiplications where:
- The "batch" dimension is effectively the expert dimension within each device
- Each expert on the device processes its assigned tokens independently
- All experts on a device are processed in parallel as a batched operation

### Step 1: Gate Projection (Batched Matrix Multiplication)
- Input:
  - Hidden states (T x H)
  - Num routed tokens (E) - to know T_e for each expert
  - Routed tokens (E x T) - token indices for each expert
  - Device-expert mapping (E/D) - global expert indices for this device
- Weights: gate_proj weights (E/D x H x H' per device)
- Computation:
  - For each local expert index i in [0, E/D):
    - Get global expert index: global_e = device_expert_mapping[i]
    - Get token count: T_e = num_routed_tokens[global_e]
    - Get token indices: routed_tokens[global_e, :T_e]
    - Get weights: gate_proj_weights[i] (local index for weights)
    - Perform: (T_e x H) @ (H x H') = T_e x H'
  - This is batched across all E/D experts in parallel
- Output:
  - Actual output size: T_d x H' per device (sum of all T_e for experts on device)
  - Pre-allocated tensor size: (K * T) x H'
- Compute for selected experts only (experts on this device)

### Step 2: Up Projection (Batched Matrix Multiplication)
- Input:
  - Hidden states (T x H)
  - Num routed tokens (E) - to know T_e for each expert
  - Routed tokens (E x T) - token indices for each expert
  - Device-expert mapping (E/D) - global expert indices for this device
- Weights: up_proj weights (E/D x H x H' per device)
- Computation:
  - For each local expert index i in [0, E/D):
    - Get global expert index: global_e = device_expert_mapping[i]
    - Get token count: T_e = num_routed_tokens[global_e]
    - Get token indices: routed_tokens[global_e, :T_e]
    - Get weights: up_proj_weights[i] (local index for weights)
    - Perform: (T_e x H) @ (H x H') = T_e x H'
  - This is batched across all E/D experts in parallel
- Output:
  - Actual output size: T_d x H' per device (sum of all T_e for experts on device)
  - Pre-allocated tensor size: (K * T) x H'
- Compute for selected experts only (experts on this device)

### Step 3: Elementwise Multiplication with SiLU
- Input:
  - Gate activations from Step 1 (T_d x H' actual, stored in pre-allocated tensor)
  - Up activations from Step 2 (T_d x H' actual, stored in pre-allocated tensor)
- Apply SiLU activation to gate activations
- Elementwise multiply with up activations
- Output: Combined activations (T_d x H')

### Step 4: Down Projection (Batched Matrix Multiplication)
- Input:
  - Combined activations from Step 3 (T_d x H' actual, stored in pre-allocated tensor)
  - Num routed tokens (E) - to know T_e for each expert
  - Routed tokens (E x T) - token indices for each expert
  - Routed token weights (E x T) - routing weights for accumulation
  - Device-expert mapping (E/D) - global expert indices for this device
- Weights: down_proj weights (E/D x H' x H per device)
- Computation:
  - For each local expert index i in [0, E/D):
    - Get global expert index: global_e = device_expert_mapping[i]
    - Get token count: T_e = num_routed_tokens[global_e]
    - Get token indices: routed_tokens[global_e, :T_e]
    - Get routing weights: routed_token_weights[global_e, :T_e]
    - Get combined activations for this expert from Step 3 output
    - Get weights: down_proj_weights[i] (local index for weights)
    - Perform: (T_e x H') @ (H' x H) = T_e x H
    - Multiply each result by corresponding routing weight
  - This is batched across all E/D experts in parallel
- Output:
  - Each token-expert pair produces H-dimensional output
  - Results are multiplied by routing weights
  - Accumulate to final output tensor (T x H) at appropriate token positions
- **Important**: Results must be accumulated (not overwritten) to the final T x H output tensor
- Multiple experts' results are aggregated per token using accumulation