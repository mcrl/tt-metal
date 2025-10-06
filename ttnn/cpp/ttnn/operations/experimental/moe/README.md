# MoE Operations - Implementation Status

## prepare_moe_mapping_tensor

### Purpose
Converts sparse MoE expert selection to dense format for efficient computation.

**Input:**
- `selected_experts` (T × K): Expert indices selected for each token
- `routing_weights` (T × K): Routing weights for selected experts
- `num_experts` (scalar): Total number of experts

**Output:**
- Dense tensor (T × E): Routing weights in dense format
  - `output[t, e] = weight` if expert `e` was selected for token `t`
  - `output[t, e] = 0` otherwise

**Example:**
```
selected_experts = [[0, 3], [1, 5]]  # 2 tokens, top-2
routing_weights = [[0.6, 0.4], [0.7, 0.3]]
num_experts = 8

Expected output:
[[0.6, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0],
 [0.0, 0.7, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0]]
```

## Current Implementation Status

### ✅ Completed Components

1. **C++ Operation Structure**
   - [prepare_moe_mapping_tensor_op.hpp](prepare_moe_mapping_tensor/device/prepare_moe_mapping_tensor_op.hpp)
   - [prepare_moe_mapping_tensor_op.cpp](prepare_moe_mapping_tensor/device/prepare_moe_mapping_tensor_op.cpp)
   - Proper validation and output spec computation

2. **Program Factory**
   - [prepare_moe_mapping_tensor_program_factory.hpp](prepare_moe_mapping_tensor/device/prepare_moe_mapping_tensor_program_factory.hpp)
   - [prepare_moe_mapping_tensor_program_factory.cpp](prepare_moe_mapping_tensor/device/prepare_moe_mapping_tensor_program_factory.cpp)
   - Single-core kernel creation with proper CBs and args

3. **Python Bindings**
   - [prepare_moe_mapping_tensor_pybind.hpp](prepare_moe_mapping_tensor/prepare_moe_mapping_tensor_pybind.hpp)
   - [prepare_moe_mapping_tensor_pybind.cpp](prepare_moe_mapping_tensor/prepare_moe_mapping_tensor_pybind.cpp)
   - Registered in experimental_pybind.cpp

4. **API Registration**
   - [prepare_moe_mapping_tensor.hpp](prepare_moe_mapping_tensor/prepare_moe_mapping_tensor.hpp)
   - [prepare_moe_mapping_tensor.cpp](prepare_moe_mapping_tensor/prepare_moe_mapping_tensor.cpp)
   - Accessible as `ttnn.prepare_moe_mapping_tensor()`

5. **Build Integration**
   - CMakeLists.txt updated
   - Project compiles successfully
   - No linker errors

6. **Test Framework**
   - [test_moe_mapping.py](../../../../models/demos/qwen3/tests/test_moe_mapping.py)
   - API test passes
   - Functional tests in place

### ❌ Current Issue

**Kernel Data Movement Problem**

The device kernel ([reader_writer_moe_mapping.cpp](prepare_moe_mapping_tensor/device/kernels/dataflow/reader_writer_moe_mapping.cpp)) runs without crashing but produces incorrect output:

- **Expected**: Sparse-to-dense mapping with routing weights
- **Actual**: All zeros

**Symptoms:**
- Kernel executes successfully (no crashes)
- Zero-initialization works correctly
- Scatter writes don't happen (or write zeros)
- Tests fail with output mismatch

**Likely Causes:**
1. TensorAccessor usage incorrect for ROW_MAJOR layout
2. Memory addressing calculations wrong
3. NOC read operations not reading actual data
4. Page size calculation mismatch
5. Data type handling (uint32 vs bfloat16)

## Debugging Steps

### Immediate Actions
1. **Add Debug Prints**
   ```cpp
   DPRINT << "expert_indices[0]=" << expert_indices[0] << ENDL();
   DPRINT << "weights[0]=" << weights[0] << ENDL();
   DPRINT << "output_noc_addr=" << output_noc_addr << ENDL();
   ```

2. **Verify Buffer Addresses**
   ```cpp
   DPRINT << "selected_experts_addr=" << selected_experts_addr << ENDL();
   DPRINT << "routing_weights_addr=" << routing_weights_addr << ENDL();
   ```

3. **Simplify Kernel**
   - Remove TensorAccessor
   - Use direct `get_noc_addr(base + offset)` pattern
   - Reference: [fill_rm_interleaved.cpp](../../data_movement/fill_rm/device/kernels/dataflow/fill_rm_interleaved.cpp)

### Alternative Approaches

**Option A: Composite Operation (Recommended)**
Implement using existing TTNN operations instead of custom kernel:
```python
def prepare_moe_mapping_tensor_composite(selected_experts, routing_weights, num_experts):
    # Use ttnn.scatter or ttnn.gather operations
    # More maintainable, leverages tested operations
    pass
```

**Option B: Simplified Kernel**
Rewrite kernel without TensorAccessor complexity:
- Use interleaved buffer pattern from reference kernels
- Manual address calculation with known strides
- Add extensive validation and assertions

**Option C: Multi-Core Implementation**
Once single-core works, parallelize:
- Split tokens across cores
- Use work_split utilities
- Better performance for large batches

## Key Constraints

**Tensor Layouts:**
- `selected_experts`: ROW_MAJOR, dtype=uint32
- `routing_weights`: ROW_MAJOR, dtype=bfloat16
- `output`: ROW_MAJOR or TILE, dtype=bfloat16

**Memory:**
- All tensors in DRAM (interleaved)
- Output padded to tile boundary (32) for TILE layout

**Hardware:**
- Single core (0,0) implementation
- RISC-V data movement processor

## Reference Files

**Similar Operations:**
- Pad: `ttnn/cpp/ttnn/operations/data_movement/pad/`
- Concat: `ttnn/cpp/ttnn/operations/data_movement/concat/`
- Fill: `ttnn/cpp/ttnn/operations/data_movement/fill_rm/`

**Kernel Examples:**
- `pad/device/kernels/dataflow/reader_pad_dims_rm_interleaved.cpp`
- `fill_rm/device/kernels/dataflow/fill_rm_interleaved.cpp`

## Contact

For questions or further development, refer to:
- [CLAUDE.md](../../../../CLAUDE.md) - Development guidelines
- [METALIUM_GUIDE.md](../../../../METALIUM_GUIDE.md) - Architecture docs
