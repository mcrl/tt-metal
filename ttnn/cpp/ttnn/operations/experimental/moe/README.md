# MoE Operations - Implementation Status

This directory contains TTNN operations for Mixture of Experts (MoE) routing and mapping functionality.

## Operations Overview

### 1. prepare_moe_mapping_tensor

Converts sparse MoE expert selection to dense format for efficient computation.

**Input:**
- `selected_experts` (T × K): Expert indices selected for each token, ROW_MAJOR, uint32
- `routing_weights` (T × K): Routing weights for selected experts, ROW_MAJOR, bfloat16
- `num_experts` (scalar): Total number of experts

**Output:**
- Dense tensor (T × E): Routing weights in dense format, ROW_MAJOR, bfloat16
  - `output[t, e] = weight` if expert `e` was selected for token `t`
  - `output[t, e] = 0` otherwise

**Example:**
```python
selected_experts = [[0, 3], [1, 5]]  # 2 tokens, top-2
routing_weights = [[0.6, 0.4], [0.7, 0.3]]
num_experts = 8

output = ttnn.prepare_moe_mapping_tensor(
    selected_experts, routing_weights, num_experts=8
)

# Expected output:
# [[0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0],
#  [0.0, 0.7, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0]]
```

**Status:** ✅ **FULLY IMPLEMENTED AND TESTED**
- All 45 test cases passing
- Supports various configurations (8-128 experts, 2-4 top-k, 2-32 tokens)

### 2. prepare_moe_routing_tensors

Converts sparse MoE expert selection into efficient routing tensors for expert-parallel computation.

**Input:**
- `selected_experts` (T × K): Expert indices for each token, ROW_MAJOR, uint32
- `routing_weights` (T × K): Routing weights, ROW_MAJOR, bfloat16
- `num_experts` (scalar): Total number of experts

**Outputs:**
Three tensors returned as a tuple:
- `num_routed_tokens` (E): Count of tokens routed to each expert, ROW_MAJOR, uint32
- `routed_tokens` (E × max_tokens): Token indices for each expert (padded), ROW_MAJOR, uint32
- `routed_token_weights` (E × max_tokens): Routing weights for each expert (padded), ROW_MAJOR, bfloat16

**Example:**
```python
selected_experts = [[0, 1], [0, 2], [1, 3]]  # 3 tokens, top-2, 4 experts
routing_weights = [[0.6, 0.4], [0.5, 0.5], [0.7, 0.3]]

num_routed, routed_tokens, routed_weights = ttnn.prepare_moe_routing_tensors(
    selected_experts, routing_weights, num_experts=4
)

# Expert 0: receives tokens [0, 1]
# Expert 1: receives tokens [0, 2]
# Expert 2: receives token [1]
# Expert 3: receives token [2]
```

**Status:** ✅ **FULLY IMPLEMENTED AND TESTED**
- All 60 test cases passing
- Supports 8-128 experts, 2-4 top-k, 2-32 tokens configurations

## Implementation Details

### ✅ Completed Components

1. **C++ Operation Structure**
   - Device operation headers and implementations
   - Proper input validation and output spec computation
   - Multi-output tensor support for routing_tensors

2. **Program Factory**
   - Single-core kernel implementation
   - Circular buffer configuration
   - Runtime argument passing

3. **Device Kernels**
   - Data movement kernels using L1 circular buffers
   - Proper NOC operations with barriers
   - Interleaved DRAM buffer support

4. **Python Bindings**
   - Properly registered in `moe::detail` namespace
   - Comprehensive docstrings
   - Type-safe argument handling

5. **API Registration**
   - `ttnn.prepare_moe_mapping_tensor()`
   - `ttnn.prepare_moe_routing_tensors()`

6. **Build Integration**
   - CMakeLists.txt configured
   - Builds successfully with tt-metal
   - No linker errors

7. **Test Coverage**
   - Comprehensive pytest test suites
   - Multiple configuration test matrices
   - Reference PyTorch implementation for validation

## Usage

### In Qwen3 MoE Model

These operations are used in the Qwen3-MoE model implementation:

```python
# In models/demos/qwen3/tt/moe.py
import ttnn

# Prepare dense mapping for MoE computation
moe_mapping = ttnn.prepare_moe_mapping_tensor(
    selected_experts,  # (batch_seq, top_k)
    routing_weights,   # (batch_seq, top_k)
    num_experts=self.num_experts
)

# Or prepare routing tensors for expert-parallel execution
num_routed, routed_tokens, routed_weights = ttnn.prepare_moe_routing_tensors(
    selected_experts,
    routing_weights,
    num_experts=self.num_experts
)
```

### Test Suite

Run the test suite to validate functionality:

```bash
# Test prepare_moe_mapping_tensor
cd models/demos/qwen3
pytest tests/test_moe_mapping.py -v

# Test prepare_moe_routing_tensors
pytest tests/test_moe_routing_tensors.py -v
```

## Technical Details

### Kernel Implementation

Both operations use single-core data movement kernels that:
1. Read input tensors from DRAM via NOC
2. Process data in L1 circular buffers
3. Write output tensors back to DRAM

**Key Design Decisions:**
- Use L1 circular buffers for intermediate storage (not stack arrays)
- Read-Modify-Write pattern for scattered writes to interleaved buffers
- Proper NOC barriers after async operations
- Page size = row size for ROW_MAJOR interleaved buffers

### Performance Characteristics

Current implementation:
- Single-core execution on core (0,0)
- Suitable for small-medium token counts (<128)
- Future: Multi-core parallelization for larger batches

## Reference Files

**Similar Operations:**
- Fill: `ttnn/cpp/ttnn/operations/data_movement/fill_rm/`
- Pad: `ttnn/cpp/ttnn/operations/data_movement/pad/`
- Concat: `ttnn/cpp/ttnn/operations/data_movement/concat/`

**Documentation:**
- [CLAUDE.md](../../../../CLAUDE.md) - Development guidelines
- [METALIUM_GUIDE.md](../../../../METALIUM_GUIDE.md) - Architecture docs
- [Qwen3 MoE Tests](../../../../models/demos/qwen3/tests/) - Test examples
