# Extract Attention Input API Plan

## Implementation Status

### ✅ Phase 1: API Skeleton (COMPLETED - 2025-10-27)

**Status:** API skeleton with dummy kernels was fully implemented and tested.

**Achieved:**
- ✅ Python API callable: `ttnn.extract_attention_input_prefill()` and `ttnn.extract_attention_input_decode()`
- ✅ All 24 operation files created (operation, device op, program factory, pybind, kernels)
- ✅ Build system integrated (CMakeLists.txt, experimental_pybind.cpp)
- ✅ Compiles successfully
- ✅ Output shapes correct: `[B//dp, 1, S, H]` (prefill) and `[1, 1, B//dp, H]` (decode)
- ✅ Tests execute without crashes or hangs
- ✅ Shape/dtype validation passes
- ✅ Works with multiple mesh configurations: 1x8, 2x4, 4x2, 8x1

### 🚀 Phase 2: Real Kernel Implementation with dp_degree Parameter (COMPLETED - 2025-10-27)

**Status:** Real kernels implemented with `dp_degree` tensor parameter for dynamic device indexing.

**Major Changes:**
- ✅ **Added `dp_degree` tensor parameter** to both operations
  - Each device receives a `[1]` integer tensor containing its row index in the mesh
  - Replaces compile-time device index calculation with runtime value
  - Enables proper batch extraction based on device's DP position

**What Works:**
- ✅ **Prefill operation (bfloat16)**:
  - ✅ Real reader kernel: reads dp_degree, calculates start_tile_idx, extracts correct batch slice
  - ✅ Real writer kernel: writes tiles sequentially to output buffer
  - ✅ Program factory: creates CB for dp_degree, passes proper compile/runtime args
  - ✅ Python API: accepts dp_degree tensor parameter
  - ✅ Test helper: `create_dp_degree_tensor()` using `ShardTensor2dMesh`
  - ✅ **Tests passing**: Prefill-bf16 case verified working

- ✅ **Decode operation (bfloat16)**:
  - ✅ Real reader kernel: reads dp_degree, calculates start_tile_idx, extracts correct batch slice
  - ✅ Real writer kernel: writes tiles sequentially to output buffer
  - ✅ Program factory: creates CB for dp_degree, passes proper compile/runtime args
  - ✅ Python API: accepts dp_degree tensor parameter
  - ✅ Implementation complete: ready for testing

**What Doesn't Work Yet:**
- ❌ **bfloat8_b output** (both prefill/decode):
  - Current implementation only supports bfloat16 (direct copy)
  - Missing: Compute kernel for format conversion (bfloat16 → bfloat8_b)
  - Needs: Three-kernel pipeline (reader → compute → writer) as planned in original design
  - Current: Two-kernel pipeline (reader → writer) for bfloat16 only

**Test Results:**
- ✅ Prefill-bf16: All mesh configurations passing (1x8, 2x4, 4x2, 8x1)
- ✅ Decode-bf16: All mesh configurations passing (1x8, 2x4, 4x2, 8x1)
- ❌ Prefill-bf8: Not implemented (needs compute kernel)
- ❌ Decode-bf8: Not implemented (needs compute kernel)

### 🎯 Phase 3: API Unification (COMPLETED - 2025-10-27)

**Status:** Successfully merged separate prefill and decode operations into single unified API.

**Achievement:**
- ✅ **Created unified `ttnn.extract_attention_input()` operation**
  - Automatically detects prefill/decode mode based on input tensor rank
  - Rank 3 `[B, S, H]` → Prefill mode → `[B//dp, 1, S, H]`
  - Rank 4 `[1, 1, B, H]` → Decode mode → `[1, 1, B//dp, H]`
  - Single implementation, shared kernels for both modes

- ✅ **Deleted legacy operations**
  - Removed `extract_attention_input_prefill/` (12 files)
  - Removed `extract_attention_input_decode/` (12 files)
  - Eliminated code duplication

- ✅ **Updated existing tests**
  - Tests now use `ttnn.extract_attention_input()` instead of mode-specific APIs
  - All bfloat16 tests passing (8/8 tests across 4 mesh configs × 2 modes)

**Benefits:**
- 50% less code (single implementation vs two)
- Simpler API (mode detection is automatic)
- Easier maintenance (one codebase to debug/optimize)
- Consistent behavior guaranteed (shared kernels)

**Test Results (Phase 3):**
```
✅ test_extract_attention_input_prefill[1x8-bfloat16]  PASSED
✅ test_extract_attention_input_prefill[2x4-bfloat16]  PASSED
✅ test_extract_attention_input_prefill[4x2-bfloat16]  PASSED
✅ test_extract_attention_input_prefill[8x1-bfloat16]  PASSED
✅ test_extract_attention_input_decode[1x8-bfloat16]   PASSED
✅ test_extract_attention_input_decode[2x4-bfloat16]   PASSED
✅ test_extract_attention_input_decode[4x2-bfloat16]   PASSED
✅ test_extract_attention_input_decode[8x1-bfloat16]   PASSED
❌ test_extract_attention_input_prefill[*-bfloat8_b]   FAILED (8 tests - format conversion not implemented)
❌ test_extract_attention_input_decode[*-bfloat8_b]    FAILED (8 tests - format conversion not implemented)
```

**Current API Signature:**
```python
ttnn.extract_attention_input(
    hidden_state,      # [B, S, H] OR [1, 1, B, H] - auto-detects mode
    dp_degree,         # [1] per-device row index
    mesh_device,
    *,
    output_dtype=None, # ttnn.bfloat16 (works) or ttnn.bfloat8_b (not implemented)
    memory_config=None,
    queue_id=0
)
```

**Implementation Files (Phase 3 - Unified):**
```
ttnn/cpp/ttnn/operations/experimental/attention/
├── CMakeLists.txt                                          ✅ Updated (unified operation only)
└── extract_attention_input/                                ✅ NEW (unified operation)
    ├── extract_attention_input.hpp                         ✅ Created
    ├── extract_attention_input.cpp                         ✅ Created
    ├── extract_attention_input_pybind.hpp                  ✅ Created
    ├── extract_attention_input_pybind.cpp                  ✅ Created
    └── device/
        ├── extract_attention_input_op.hpp                  ✅ Created
        ├── extract_attention_input_op.cpp                  ✅ Created (mode detection logic)
        ├── extract_attention_input_program_factory.hpp     ✅ Created
        ├── extract_attention_input_program_factory.cpp     ✅ Created (mode detection logic)
        └── kernels/dataflow/
            ├── reader_extract_attention_input.cpp          ✅ Created (mode-agnostic)
            └── writer_extract_attention_input.cpp          ✅ Created (mode-agnostic)
```

**Test Infrastructure:**
```
models/demos/qwen3/tests/
└── test_extract_attention_input.py                         ✅ Updated
    ├── create_dp_degree_tensor()                           ✅ Helper function (unchanged)
    │   └── Uses ShardTensor2dMesh to create [1] tensor per device
    ├── test_extract_attention_input_prefill()              ✅ Updated (uses unified API)
    └── test_extract_attention_input_decode()               ✅ Updated (uses unified API)
```

**Modified Build System Files:**
- ✅ `ttnn/cpp/ttnn/operations/experimental/attention/CMakeLists.txt` - Removed old sources, kept unified only
- ✅ `ttnn/cpp/ttnn/operations/experimental/experimental_pybind.cpp` - Removed old bindings, kept unified only
- ✅ `ttnn/CMakeLists.txt` - Updated pybind list (line 213), removed old entries (lines 214-215)

**Implementation Details:**

**API Evolution:**
```python
# Phase 1: Separate operations without dp_degree
ttnn.extract_attention_input_prefill(hidden_state, mesh_device, *, output_dtype=None, ...)
ttnn.extract_attention_input_decode(hidden_state, mesh_device, *, output_dtype=None, ...)

# Phase 2: Added dp_degree parameter
ttnn.extract_attention_input_prefill(hidden_state, dp_degree, mesh_device, *, output_dtype=None, ...)
ttnn.extract_attention_input_decode(hidden_state, dp_degree, mesh_device, *, output_dtype=None, ...)

# Phase 3: Unified API (CURRENT)
ttnn.extract_attention_input(hidden_state, dp_degree, mesh_device, *, output_dtype=None, ...)
  # Automatically detects mode from input shape rank
```

**dp_degree Tensor:**
- Shape: `[1]` per device (scalar integer)
- Dtype: `INT32` or `UINT32`
- Value: Device's row index in mesh (0, 1, 2, ..., dp-1)
- Creation: Use `create_dp_degree_tensor(mesh_device)` helper in tests
- Implementation: Uses `ShardTensor2dMesh` with `dims=(0, None)` and `mesh_shape=(num_devices, 1)`

**Kernel Implementation Pattern:**
```cpp
// Reader kernel:
1. Read dp_degree value from device buffer to L1 CB
2. Calculate start_tile_idx = dp_degree_value * tiles_per_device
3. Read tiles [start_tile_idx : start_tile_idx + tiles_per_device] from input
4. Push to CB for writer

// Writer kernel:
1. Wait for tiles from reader
2. Write tiles sequentially starting from index 0
3. One tile at a time with immediate barriers (safety-first)
```

### 🚧 Phase 3: Format Conversion Support (TODO)

**Next Steps:**
1. Implement compute kernel for bfloat16 → bfloat8_b conversion
2. Update program factory to use three-kernel pipeline when `output_dtype == BFLOAT8_B`
3. Test bfloat8_b output for both prefill and decode
4. Verify correctness with PyTorch reference

**Required Changes:**
- Add compute kernel: `copy_tiles_with_format_conversion.cpp` (as planned in original design)
- Update program factory: Conditionally create compute kernel based on output_dtype
- Create CB 16 for output with proper DataFormat (Float16_b or Bfp8_b)
- Current two-kernel pipeline (reader→writer) works for bfloat16 only

---

## Parameters

- `dp`: Data parallelism degree, obtained from `mesh_device.shape[0]`

## Implementation Plan

### Directory Structure

Create two new operations following the MoE operations pattern:

```
ttnn/cpp/ttnn/operations/experimental/attention/
├── CMakeLists.txt
├── extract_attention_input_prefill/
│   ├── extract_attention_input_prefill.hpp
│   ├── extract_attention_input_prefill.cpp
│   ├── extract_attention_input_prefill_pybind.hpp
│   ├── extract_attention_input_prefill_pybind.cpp
│   └── device/
│       ├── extract_attention_input_prefill_op.hpp
│       ├── extract_attention_input_prefill_op.cpp
│       ├── extract_attention_input_prefill_program_factory.hpp
│       ├── extract_attention_input_prefill_program_factory.cpp
│       └── kernels/
│           ├── dataflow/
│           │   ├── reader_extract_attention_input_prefill.cpp
│           │   └── writer_extract_attention_input_prefill.cpp
│           └── compute/
│               └── copy_tiles_with_format_conversion.cpp
└── extract_attention_input_decode/
    ├── extract_attention_input_decode.hpp
    ├── extract_attention_input_decode.cpp
    ├── extract_attention_input_decode_pybind.hpp
    ├── extract_attention_input_decode_pybind.cpp
    └── device/
        ├── extract_attention_input_decode_op.hpp
        ├── extract_attention_input_decode_op.cpp
        ├── extract_attention_input_decode_program_factory.hpp
        ├── extract_attention_input_decode_program_factory.cpp
        └── kernels/
            ├── dataflow/
            │   ├── reader_extract_attention_input_decode.cpp
            │   └── writer_extract_attention_input_decode.cpp
            └── compute/
                └── copy_tiles_with_format_conversion.cpp
```

### Files to Modify

1. **`ttnn/cpp/ttnn/operations/experimental/experimental_pybind.cpp`**
   - Add includes for both operation pybind headers
   - Add binding calls in `py_module` function

2. **`ttnn/CMakeLists.txt`** (3 locations)
   - Line ~200: Add to `TTNN_SRC_PYBIND` list
   - Line ~494: Add to `target_link_libraries`
   - Line ~647: Add `add_subdirectory(operations/experimental/attention)`

### Implementation Steps

#### Step 1: Create Operation Interface Files

**`extract_attention_input_prefill.hpp`:**
- Define `ExtractAttentionInputPrefillOperation` struct
- Register operation with `ttnn::register_operation`
- Define `invoke()` signature with parameters:
  - `queue_id`
  - `hidden_state`
  - `mesh_device`
  - `output_dtype` (optional, default: `DataType::BFLOAT16`)
  - `memory_config` (optional)

**`extract_attention_input_decode.hpp`:**
- Similar structure to prefill version
- Same parameters

#### Step 2: Create Operation Implementation Files

**`extract_attention_input_prefill.cpp`:**
- Implement `ExtractAttentionInputPrefillOperation::invoke()`
- Extract `dp` from `mesh_device.shape[0]`
- Validate `output_dtype` is either `BFLOAT16` or `BFLOAT8_B`
- Call `tt::tt_metal::operation::run()` with device operation struct, passing `output_dtype`

**`extract_attention_input_decode.cpp`:**
- Similar implementation to prefill

#### Step 3: Create Device Operation Files

**`extract_attention_input_prefill_op.hpp`:**
- Define `ExtractAttentionInputPrefill` struct in namespace `ttnn::operations::experimental::attention`
- Members:
  - `output_mem_config` (MemoryConfig)
  - `output_dtype` (DataType - BFLOAT16 or BFLOAT8_B)
  - `dp` (data parallelism degree)
- Methods: `validate_with_output_tensors()`, `compute_output_specs()`, `create_output_tensors()`, `create_program()`

**`extract_attention_input_prefill_op.cpp`:**
- Implement validation in `validate_with_output_tensors()`:
  - Assert `input.dtype() == DataType::BFLOAT16`
  - Assert input is in TILE_LAYOUT: `input.layout() == Layout::TILE`
  - Assert input is replicated across all devices in mesh
  - Assert `output_dtype` is either `BFLOAT16` or `BFLOAT8_B`
  - Assert `(B // dp) % 32 == 0` (safe assumption - batch per device is tile-aligned)
  - Assert `H % 32 == 0` (safe assumption - hidden dimension is tile-aligned)
  - Assert `(B * S) % 32 == 0` (input rows are tile-aligned)
  - Example assertions:
    ```cpp
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "Input must be bfloat16");
    TT_FATAL(input.layout() == Layout::TILE, "Input must be in TILE_LAYOUT");
    TT_FATAL(output_dtype == DataType::BFLOAT16 || output_dtype == DataType::BFLOAT8_B,
             "Output dtype must be BFLOAT16 or BFLOAT8_B");
    TT_FATAL((B * S) % 32 == 0, "Input rows (B * S) must be tile-aligned");
    TT_FATAL((B / dp) % 32 == 0, "Batch per device must be tile-aligned");
    TT_FATAL(H % 32 == 0, "Hidden dimension must be tile-aligned");
    ```
- Implement `compute_output_specs()` to return output shape `[B // dp, 1, S, H]` with `output_dtype`
  - The `1` dimension at index 1 is required for API compatibility with downstream operations:
    - `ttnn.linear()` expects 4D input for batched matrix multiplication
    - `ttnn.experimental.nlp_create_qkv_heads()` expects 4D input
    - Maintains consistency with decode mode shape pattern `[1, 1, B // dp, H]`
- Call program factory in `create_program()`

**`extract_attention_input_decode_op.cpp`:**
- Implement validation in `validate_with_output_tensors()`:
  - Assert `input.dtype() == DataType::BFLOAT16`
  - Assert input is in TILE_LAYOUT: `input.layout() == Layout::TILE`
  - Assert input is replicated across all devices in mesh
  - Assert `output_dtype` is either `BFLOAT16` or `BFLOAT8_B`
  - Assert `(B // dp) % 32 == 0` (safe assumption - automatically satisfied for decode)
  - Assert `H % 32 == 0` (safe assumption - automatically satisfied for decode)
  - Note: No additional tile alignment assertions needed - the safe assumptions cover all requirements
  - Example assertions:
    ```cpp
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "Input must be bfloat16");
    TT_FATAL(input.layout() == Layout::TILE, "Input must be in TILE_LAYOUT");
    TT_FATAL(output_dtype == DataType::BFLOAT16 || output_dtype == DataType::BFLOAT8_B,
             "Output dtype must be BFLOAT16 or BFLOAT8_B");
    TT_FATAL((B / dp) % 32 == 0, "Batch per device must be tile-aligned");
    TT_FATAL(H % 32 == 0, "Hidden dimension must be tile-aligned");
    ```
- Implement `compute_output_specs()` to return output shape `[1, 1, B // dp, H]` with `output_dtype`
- Call program factory in `create_program()`

#### Step 4: Create Program Factory

**`extract_attention_input_prefill_program_factory.cpp`:**

**Implementation approach:**
- **Single-core per device**: Each device uses one core to perform the extraction operation
- **Three-kernel pipeline** for format conversion support:
  1. **Reader kernel**: Reads bfloat16 tiles from input DRAM to input CB (bfloat16 format)
  2. **Compute kernel**: Copies tiles from input CB to output CB with format conversion via `pack_tile()`
  3. **Writer kernel**: Writes tiles from output CB to output DRAM

**Optimization**: When `output_dtype == BFLOAT16`, use simplified two-kernel approach (reader-writer only, no conversion needed).

**Program factory implementation:**
- Create `Program` object
- Extract input shape: `[B, S, H]` (guaranteed tile-aligned by validation)
- Extract `output_dtype` from operation struct
- Calculate tile dimensions (all exact divisions):
  - `constexpr uint32_t TILE_HEIGHT = 32`, `TILE_WIDTH = 32`
  - `batch_per_device = B / dp` (exact division, guaranteed)
  - `num_tile_rows_input = (B * S) / TILE_HEIGHT` (exact, validated)
  - `num_tile_cols = H / TILE_WIDTH` (exact, safe assumption)
  - `num_tile_rows_per_device = (batch_per_device * S) / TILE_HEIGHT` (exact)
  - `tiles_per_device = num_tile_rows_per_device * num_tile_cols` (total tiles to copy)
- Calculate data formats and tile size:
  - Input is always bfloat16 (validated)
  - `constexpr uint32_t tile_size_bytes = 2048` (32 * 32 * 2 bytes - same for both bfloat16 and bfloat8_b)
  - `input_data_format = DataFormat::Float16_b` (bfloat16)
  - `output_data_format = datatype_to_dataformat_converter(output_dtype)` (Float16_b or Bfp8_b)
  - `uint32_t input_tile_size = tt_metal::detail::TileSize(input_data_format)` (2048)
  - `uint32_t output_tile_size = tt_metal::detail::TileSize(output_data_format)` (2048)
- Define core to use (single-core implementation):
  - Use core (0, 0) from the device's compute grid
  - `CoreCoord core = {0, 0};` or use first available core from compute_with_storage_grid
- Create circular buffers:
  - `constexpr uint32_t tiles_per_batch = 8` (batch size for NOC operations)
  - `constexpr uint32_t cb_in = CBIndex::c_0` (input CB index)
  - `constexpr uint32_t cb_out = CBIndex::c_16` (output CB index)
  - CB 0 (input): Size = `input_tile_size * tiles_per_batch` = `2048 * 8` = **16384 bytes**
    - Stores 8 bfloat16 tiles for batched NOC reads
    - Configuration: `CircularBufferConfig(16384, {{cb_in, input_data_format}}).set_page_size(cb_in, 2048)`
  - CB 16 (output): Size = `output_tile_size * tiles_per_batch` = `2048 * 8` = **16384 bytes**
    - Stores 8 tiles in output format (bfloat16 or bfloat8_b) for batched NOC writes
    - Configuration: `CircularBufferConfig(16384, {{cb_out, output_data_format}}).set_page_size(cb_out, 2048)`
  - Both CBs are created on the single core: `CreateCircularBuffer(program, core, cb_config)`
- Create kernels (all on the same core):
  - Reader kernel with compile-time args: `cb_in`, `input_is_dram`, `tiles_per_device`
  - Compute kernel with compile-time args: `tiles_per_device`, `tiles_per_batch` (only if format conversion needed)
  - Writer kernel with compile-time args: `cb_out`, `output_is_dram`, `tiles_per_device`
- Set runtime args per device:
  - Reader: `input_buffer->address()`, `start_tile_idx`
  - Writer: `output_buffer->address()`
  - `start_tile_idx = device_idx * tiles_per_device`
- Setup override_runtime_arguments_callback for buffer address updates

**Simplified calculations (no padding, no ceiling divisions):**
```cpp
// All divisions are exact due to tile-alignment guarantees
uint32_t batch_per_device = B / dp;
uint32_t num_tile_rows_per_device = (batch_per_device * S) / 32;
uint32_t num_tile_cols = H / 32;
uint32_t tiles_per_device = num_tile_rows_per_device * num_tile_cols;

// For device i: copy tiles [i * tiles_per_device, (i+1) * tiles_per_device)
uint32_t start_tile_idx = device_idx * tiles_per_device;
```

**Example circular buffer configuration code:**
```cpp
// In program factory
constexpr uint32_t tiles_per_batch = 8;
constexpr uint32_t tile_size_bytes = 2048;
constexpr uint32_t cb_in = CBIndex::c_0;
constexpr uint32_t cb_out = CBIndex::c_16;

// Calculate CB sizes (must match kernel batch size!)
uint32_t cb_in_size = tiles_per_batch * input_tile_size;   // 8 * 2048 = 16384 bytes
uint32_t cb_out_size = tiles_per_batch * output_tile_size;  // 8 * 2048 = 16384 bytes

// Create input CB (bfloat16 format)
CircularBufferConfig cb_in_config =
    CircularBufferConfig(cb_in_size, {{cb_in, input_data_format}})
        .set_page_size(cb_in, input_tile_size);
CreateCircularBuffer(program, core, cb_in_config);

// Create output CB (bfloat16 or bfloat8_b format based on output_dtype)
CircularBufferConfig cb_out_config =
    CircularBufferConfig(cb_out_size, {{cb_out, output_data_format}})
        .set_page_size(cb_out, output_tile_size);
CreateCircularBuffer(program, core, cb_out_config);
```

**Decode version:**
- Shapes `[1, 1, B, H]` → `[1, 1, B // dp, H]`
- Even simpler calculations:
```cpp
uint32_t batch_per_device = B / dp;  // Exact, safe assumption
uint32_t num_tile_rows_per_device = batch_per_device / 32;  // Exact, safe assumption
uint32_t num_tile_cols = H / 32;  // Exact, safe assumption
uint32_t tiles_per_device = num_tile_rows_per_device * num_tile_cols;
uint32_t start_tile_idx = device_idx * tiles_per_device;

// Extract output_dtype and convert to data format
DataFormat output_data_format = datatype_to_dataformat_converter(output_dtype);
```

#### Step 5: Create Kernels

**Three-kernel implementation for format conversion:**

---

**Kernel 1: Reader (`reader_extract_attention_input_prefill.cpp`)**

**Purpose**: Read bfloat16 tiles from input DRAM to input circular buffer.

**Compile-time args:**
- `cb_id` (e.g., tt::CBIndex::c_0)
- `input_is_dram`
- `tiles_per_device`

**Runtime args:**
- `input_buffer_addr`
- `start_tile_idx`

**Implementation:**
```cpp
void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr bool input_is_dram = (bool)get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_device = get_compile_time_arg_val(2);
    constexpr uint32_t tile_size_bytes = 2048;
    constexpr uint32_t tiles_per_batch = 8;

    uint32_t input_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t start_tile_idx = get_arg_val<uint32_t>(1);

    const InterleavedAddrGenFast<input_is_dram> addrgen = {
        .bank_base_address = input_buffer_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b
    };

    // SAFE PATTERN: Reserve/push one tile at a time, batch NOC operations before barrier
    for (uint32_t batch_start = 0; batch_start < tiles_per_device; batch_start += tiles_per_batch) {
        uint32_t current_batch_size = min(tiles_per_batch, tiles_per_device - batch_start);

        // Issue batched NOC reads
        for (uint32_t i = 0; i < current_batch_size; i++) {
            cb_reserve_back(cb_id, 1);  // Reserve ONE tile - SAFE
            uint32_t l1_addr = get_write_ptr(cb_id);  // Safe pointer - just allocated
            uint64_t noc_addr = get_noc_addr(start_tile_idx + batch_start + i, addrgen);
            noc_async_read(noc_addr, l1_addr, tile_size_bytes);
            cb_push_back(cb_id, 1);  // Push ONE tile immediately
        }
        noc_async_read_barrier();  // Single barrier for batch
    }
}
```

---

**Kernel 2: Compute (`copy_tiles_with_format_conversion.cpp`)**

**Purpose**: Copy tiles from input CB to output CB, converting format from bfloat16 to bfloat8_b if needed.

**Note**: This kernel is only created when `output_dtype != input_dtype` (i.e., when conversion is needed).

**Compile-time args:**
- `tiles_per_device`
- `tiles_per_batch`

**Implementation:**
```cpp
#include <compute_api.h>

void MAIN {
    constexpr uint32_t tiles_per_device = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_batch = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Initialize copy operation (no actual computation, just pack with format conversion)
    copy_tile_to_dst_init_short(cb_in);

    for (uint32_t batch_start = 0; batch_start < tiles_per_device; batch_start += tiles_per_batch) {
        uint32_t current_batch_size = (batch_start + tiles_per_batch > tiles_per_device)
                                        ? (tiles_per_device - batch_start)
                                        : tiles_per_batch;

        // Wait for reader to fill input CB
        cb_wait_front(cb_in, current_batch_size);

        // Reserve space in output CB
        cb_reserve_back(cb_out, current_batch_size);

        // Acquire destination registers
        acquire_dst();

        // Copy tiles with format conversion
        // pack_tile() handles bfloat16 -> bfloat8_b conversion based on CB formats
        for (uint32_t i = 0; i < current_batch_size; i++) {
            copy_tile(cb_in, i, i);  // Copy tile from input CB to DST register
            pack_tile(i, cb_out);     // Pack DST to output CB (format conversion happens here)
        }

        // Release destination registers
        release_dst();

        // Push output tiles, pop input tiles
        cb_push_back(cb_out, current_batch_size);
        cb_pop_front(cb_in, current_batch_size);
    }
}
```

**Key insight**: `pack_tile(dst_reg, cb_out)` performs format conversion automatically based on the output CB's configured `DataFormat`. Hardware handles the complex block floating point packing for bfloat8_b.

---

**Kernel 3: Writer (`writer_extract_attention_input_prefill.cpp`)**

**Purpose**: Write tiles from output circular buffer to output DRAM.

**Compile-time args:**
- `cb_id` (e.g., tt::CBIndex::c_16)
- `output_is_dram`
- `tiles_per_device`

**Runtime args:**
- `output_buffer_addr`

**Implementation:**
```cpp
void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr bool output_is_dram = (bool)get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_device = get_compile_time_arg_val(2);
    constexpr uint32_t tile_size_bytes = 2048;
    constexpr uint32_t tiles_per_batch = 8;

    uint32_t output_buffer_addr = get_arg_val<uint32_t>(0);

    const InterleavedAddrGenFast<output_is_dram> addrgen = {
        .bank_base_address = output_buffer_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Float16_b  // Tile size is same for both formats
    };

    // SAFE PATTERN: Reserve/push one tile at a time, batch NOC operations before barrier
    for (uint32_t batch_start = 0; batch_start < tiles_per_device; batch_start += tiles_per_batch) {
        uint32_t current_batch_size = min(tiles_per_batch, tiles_per_device - batch_start);

        // Wait for compute to fill output CB
        cb_wait_front(cb_id, current_batch_size);

        // Issue batched NOC writes
        for (uint32_t i = 0; i < current_batch_size; i++) {
            uint32_t l1_addr = get_read_ptr(cb_id);  // Safe - already pushed by compute
            uint64_t noc_addr = get_noc_addr(batch_start + i, addrgen);
            noc_async_write(l1_addr, noc_addr, tile_size_bytes);
            cb_pop_front(cb_id, 1);  // Pop ONE tile immediately
        }
        noc_async_write_barrier();  // Single barrier for batch
    }
}
```

---

**Optimization for bfloat16 Output (No Conversion Needed)**

When `output_dtype == DataType::BFLOAT16`, format conversion is not needed. The program factory should:
- Skip creating the compute kernel
- Use a combined reader-writer kernel for direct copy (optional optimization)
- Or keep the three-kernel approach with trivial compute (simpler implementation)

**Recommended approach**: Skip compute kernel entirely for maximum efficiency when no conversion is needed.

---

**Decode Version:**
- Same kernel implementations
- Different `tiles_per_device` value (fewer tiles due to no sequence dimension)
- Input shape `[1, 1, B, H]`, output shape `[1, 1, B // dp, H]`

#### Step 6: Create Python Bindings

**`extract_attention_input_prefill_pybind.cpp`:**
- Define `bind_extract_attention_input_prefill()` function
- Write comprehensive docstring with examples
- Use `ttnn::bind_registered_operation()`
- Arguments:
  - `hidden_state` (required)
  - `mesh_device` (required)
  - Keyword-only: `output_dtype` (optional, default: `ttnn.bfloat16`)
  - Keyword-only: `memory_config` (optional)
  - Keyword-only: `queue_id` (optional, default: 0)
- Document when to use `bfloat16` vs `bfloat8_b` for output

**Decode version:**
- Similar binding structure
- Same parameter order and defaults

#### Step 7: Create CMakeLists.txt

**`ttnn/cpp/ttnn/operations/experimental/attention/CMakeLists.txt`:**
```cmake
add_library(ttnn_op_experimental_attention ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Attention ALIAS ttnn_op_experimental_attention)

target_precompile_headers(ttnn_op_experimental_attention REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_attention)

target_sources(
    ttnn_op_experimental_attention
    PRIVATE
        extract_attention_input_prefill/extract_attention_input_prefill.cpp
        extract_attention_input_prefill/device/extract_attention_input_prefill_op.cpp
        extract_attention_input_prefill/device/extract_attention_input_prefill_program_factory.cpp
        extract_attention_input_decode/extract_attention_input_decode.cpp
        extract_attention_input_decode/device/extract_attention_input_decode_op.cpp
        extract_attention_input_decode/device/extract_attention_input_decode_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_attention PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_attention
    PRIVATE
        TT::Metalium
        TTNN::Core
)
```

#### Step 8: Update Main Build Files

1. **Update `experimental_pybind.cpp`:**
```cpp
#include "ttnn/cpp/ttnn/operations/experimental/attention/extract_attention_input_prefill/extract_attention_input_prefill_pybind.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/attention/extract_attention_input_decode/extract_attention_input_decode_pybind.hpp"

// In py_module function:
::ttnn::operations::experimental::extract_attention_input_prefill::detail::bind_extract_attention_input_prefill(module);
::ttnn::operations::experimental::extract_attention_input_decode::detail::bind_extract_attention_input_decode(module);
```

2. **Update `ttnn/CMakeLists.txt` (3 places):**
   - Add pybind sources to `TTNN_SRC_PYBIND`
   - Add `TTNN::Ops::Experimental::Attention` to `target_link_libraries`
   - Add `add_subdirectory(operations/experimental/attention)`

#### Step 9: Build and Test

1. Clear kernel cache: `rm -rf ~/.cache/tt-metal-cache/`
2. Incremental build: `conda run -n tt-base cmake --build build -j16`
3. Create test file: `models/demos/qwen3/tests/test_extract_attention_input.py`
4. Test with: `timeout 30 conda run -n tt-base pytest models/demos/qwen3/tests/test_extract_attention_input.py -v`

#### Step 10: Update Attention Implementation

Modify `models/demos/qwen3/tt/attention.py`:

**Prefill mode (line ~196-199):**
```python
# OLD:
hidden_states = ttnn.reshape(hidden_states, (batch_size, -1))
hidden_states = ttnn.matmul(self.slice_mat, hidden_states, dtype=ttnn.bfloat8_b, ...)
hidden_states = ttnn.reshape(hidden_states, hidden_shape)

# NEW:
hidden_states = ttnn.extract_attention_input_prefill(
    hidden_states,
    self.mesh_device,
    output_dtype=ttnn.bfloat8_b,  # Match the dtype used in matmul
    memory_config=mem_cfg
)
```

**Decode mode (line ~353-355):**
```python
# OLD:
hidden_states = ttnn.view(hidden_states, (batch_size, hidden_size))
hidden_states = ttnn.matmul(self.slice_mat, hidden_states, dtype=ttnn.bfloat8_b, ...)
hidden_states = ttnn.view(hidden_states, (1, 1, batch_size // self.dp, hidden_size))

# NEW:
hidden_states = ttnn.extract_attention_input_decode(
    hidden_states,
    self.mesh_device,
    output_dtype=ttnn.bfloat8_b,  # Match the dtype used in matmul
    memory_config=mem_cfg
)
```

**Choosing output_dtype:**
- Use `output_dtype=ttnn.bfloat8_b` when the extracted output feeds directly into `ttnn.linear()` with `dtype=ttnn.bfloat8_b`
  - This is the typical case in attention (feeding into qkv projection)
  - Saves memory bandwidth by avoiding intermediate bfloat16 storage
- Use `output_dtype=ttnn.bfloat16` (default) when:
  - Output needs full precision for subsequent operations
  - Debugging or validating correctness
  - Not immediately feeding into bfloat8_b matmul

### Notes

**Implementation philosophy:**
- **Single-core implementation**: Uses one core per device for simplicity and correctness
  - Avoids complexity of multi-core work distribution
  - Memory-bound operation doesn't benefit significantly from parallelization
  - Can be optimized to multi-core later if profiling shows need
- **Batched NOC operations**: Amortizes barrier overhead by batching 8 tiles per barrier
  - Circular buffers sized exactly for batch size (16384 bytes = 8 tiles × 2048 bytes)
  - No CB size mismatch, optimal memory usage

**Simplicity compared to MoE scatter:**
- These operations are simpler than MoE scatter because they use consecutive tile indexing
- No routing tensors needed
- No zero-padding required
- Output should be replicated across TP dimension (mesh axis 1) naturally since input is replicated

**TILE_LAYOUT handling:**
- Input and output are both in TILE_LAYOUT (32x32 tiles)
- Input data format is always **bfloat16** (asserted)
- Output data format can be **bfloat16** or **bfloat8_b** (specified via `output_dtype` parameter)
- Data is organized as tiles, not rows
- Tile size: 32 * 32 * 2 = **2048 bytes** (constant for both bfloat16 and bfloat8_b)
- All dimensions are guaranteed to be multiples of 32 (via assertions)
- Simple tile copy operation - extract contiguous range of tiles
- Type conversion mechanism:
  - When `output_dtype == BFLOAT16`: Direct copy, no conversion needed
  - When `output_dtype == BFLOAT8_B`: Use compute kernel with `pack_tile()` for format conversion
  - Hardware handles block floating point packing during `pack_tile()` operation
  - Conversion happens in compute kernel, not during NOC write
- Use `InterleavedAddrGenFast` with `page_size = 2048`

**Tile indexing for prefill (exact calculations):**
- Input shape: `[B, S, H]` → flattened to 2D: `[B * S, H]` for tiling
- Total tiles in input: `(B * S / 32) * (H / 32)` (exact division, validated)
- Tiles indexed in row-major order (left-to-right, top-to-bottom)
- For device at index `i` in DP dimension:
  - `batch_per_device = B / dp` (exact, guaranteed)
  - `rows_per_device = batch_per_device * S` (exact)
  - `tile_rows_per_device = rows_per_device / 32` (exact)
  - `tiles_per_row = H / 32` (exact, safe assumption)
  - `tiles_per_device = tile_rows_per_device * tiles_per_row` (exact)
  - `start_tile_idx = i * tiles_per_device`
- Copy tiles `[start_tile_idx : start_tile_idx + tiles_per_device)` to output
- No partial tiles, no padding - all calculations are exact

**Tile indexing for decode (exact calculations):**
- Input shape: `[1, 1, B, H]` → flattened to 2D: `[B, H]` for tiling
- Total tiles in input: `(B / 32) * (H / 32)` (exact, guaranteed by safe assumptions)
- For device at index `i` in DP dimension:
  - `batch_per_device = B / dp` (exact, guaranteed)
  - `tile_rows_per_device = batch_per_device / 32` (exact, safe assumption: `batch_per_device % 32 == 0`)
  - `tiles_per_row = H / 32` (exact, safe assumption: `H % 32 == 0`)
  - `tiles_per_device = tile_rows_per_device * tiles_per_row` (exact)
  - `start_tile_idx = i * tiles_per_device`
- Copy tiles `[start_tile_idx : start_tile_idx + tiles_per_device)` to output
- No partial tiles, no padding - all calculations are exact

**Performance considerations:**
- **Single-core implementation** (simplicity and correctness first):
  - Each device uses one core to perform the extraction operation
  - Simple tile copy operation doesn't require parallel execution
  - Avoids complexity of work distribution and synchronization
  - Future optimization: multi-core implementation for very large tensors
- **Batched NOC operations** for efficiency:
  - Issue 8 NOC reads/writes before each barrier (amortizes barrier overhead)
  - Circular buffers sized for 8 tiles (16384 bytes each)
  - CB size matches batch size: no mismatch, no wasted memory
- **Memory-bound operation**:
  - Performance limited by NOC bandwidth, not compute
  - Single-core NOC bandwidth typically sufficient for this use case
- **Future optimizations**:
  - Multi-core distribution for very large batch×sequence products
  - Sharded memory layout instead of interleaved
  - Adjust `tiles_per_batch` based on profiling (8 is a reasonable default)

**Validation requirements and assumptions:**
- `input.dtype() == DataType::BFLOAT16` - input is always bfloat16 (ASSERT THIS)
- `output_dtype` must be either `BFLOAT16` or `BFLOAT8_B` (ASSERT THIS)
- `(B // dp) % 32 == 0` - batch per device is tile-aligned (SAFE ASSUMPTION, assert)
- `H % 32 == 0` - hidden dimension is tile-aligned (SAFE ASSUMPTION, assert)
- For prefill: `(B * S) % 32 == 0` - total input rows are tile-aligned (need to assert)
- For prefill: `((B // dp) * S) % 32 == 0` - output rows are tile-aligned (automatically satisfied if above holds)
- For decode: All tile requirements automatically satisfied by the two safe assumptions
- Input must be in TILE_LAYOUT (assert)
- Input must be replicated across all devices in mesh (assert)

**Simplifications under these assumptions:**
- **Constant input data type** - input always bfloat16, so input tile size is constant 2048 bytes
- **Flexible output data type** - output can be bfloat16 or bfloat8_b (both use 2048 bytes per tile)
- **No padding needed** - all dimensions are already tile-aligned by guarantee
- **Exact arithmetic** - tile count calculations use exact divisions (no ceiling divisions or remainder handling)
  - `tiles_per_device = ((B / dp) * S / 32) * (H / 32)` for prefill
  - `tiles_per_device = ((B / dp) / 32) * (H / 32)` for decode
  - All divisions are exact, no need for `(x + 31) / 32` style ceiling
- **No edge cases** - no partial tiles, no boundary conditions
- **Simple validation** - just assert the alignment assumptions in `validate_with_output_tensors()`
- **Decode is trivial** - validation automatically satisfied by the two safe assumptions
- **Type conversion via compute kernel** - bfloat16 to bfloat8_b conversion (if needed) handled by `pack_tile()` in compute kernel
  - Circular buffers configured with appropriate DataFormats
  - `pack_tile()` performs hardware-accelerated block floating point packing
  - When no conversion needed (output is bfloat16), compute kernel can be skipped entirely

## `extract_attention_input_prefill`

### Description

Extracts a chunk of input hidden states for each device along the data parallel (DP) dimension for prefill mode. Each device receives a consecutive chunk of the batch dimension corresponding to its position in the first mesh axis. This replaces the current matrix multiplication-based approach using `slice_mat`.

### Input Tensors

1. `hidden_state`: `[B, S, H]` in TILE_LAYOUT
   - The input `hidden_state` is the same for every device in `mesh_device` (replicated across all devices)
   - Data format: **bfloat16** (always, must be asserted)
   - Layout: TILE_LAYOUT (32x32 tiles)

### Parameters

- `output_dtype`: Output data type (optional, default: `ttnn.bfloat16`)
  - Valid values: `ttnn.bfloat16`, `ttnn.bfloat8_b`
  - Use `bfloat16` when output will be used directly in operations requiring full precision
  - Use `bfloat8_b` when output feeds into matmul/linear ops that accept bfloat8_b (saves memory and bandwidth)

### Output Tensors

1. `extracted_hidden_state`: `[B // dp, 1, S, H]` in TILE_LAYOUT
   - The `1` dimension at index 1 is required for API compatibility with downstream operations:
     - `ttnn.linear()` expects 4D input for batched matrix multiplication
     - `ttnn.experimental.nlp_create_qkv_heads()` expects 4D input
     - Maintains consistency with decode mode shape pattern `[1, 1, B // dp, H]`
   - The output `extracted_hidden_state` is the same across the TP dimension (mesh_device axis 1)
   - Data format: specified by `output_dtype` parameter (bfloat16 or bfloat8_b)
   - Layout: TILE_LAYOUT (maintained from input)

### Actions

- Output is extracted from corresponding consecutive chunk of the input
- `batch_per_device = B // dp`
- For device at coordinate `i` in the first mesh axis:
  ```
  extracted_hidden_state[:, 0, :, :] = hidden_state[i * batch_per_device : (i+1) * batch_per_device, :, :]
  ```

## `extract_attention_input_decode`

### Description

Extracts a chunk of input hidden states for each device along the data parallel (DP) dimension for decode mode. Each device receives a consecutive chunk of the batch dimension (third dimension) corresponding to its position in the first mesh axis. This replaces the current matrix multiplication-based approach using `slice_mat`.

### Input Tensors

1. `hidden_state`: `[1, 1, B, H]` in TILE_LAYOUT
   - The input `hidden_state` is the same for every device in `mesh_device` (replicated across all devices)
   - Data format: **bfloat16** (always, must be asserted)
   - Layout: TILE_LAYOUT (32x32 tiles)

### Parameters

- `output_dtype`: Output data type (optional, default: `ttnn.bfloat16`)
  - Valid values: `ttnn.bfloat16`, `ttnn.bfloat8_b`
  - Use `bfloat16` when output will be used directly in operations requiring full precision
  - Use `bfloat8_b` when output feeds into matmul/linear ops that accept bfloat8_b (saves memory and bandwidth)

### Output Tensors

1. `extracted_hidden_state`: `[1, 1, B // dp, H]` in TILE_LAYOUT
   - The output `extracted_hidden_state` is the same across the TP dimension (mesh_device axis 1)
   - Data format: specified by `output_dtype` parameter (bfloat16 or bfloat8_b)
   - Layout: TILE_LAYOUT (maintained from input)

### Actions

- Output is extracted from corresponding consecutive chunk of the input
- `batch_per_device = B // dp`
- For device at coordinate `i` in the first mesh axis:
  ```
  extracted_hidden_state[:, :, :, :] = hidden_state[:, :, i * batch_per_device : (i+1) * batch_per_device, :]
  ```