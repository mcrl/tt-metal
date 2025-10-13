# Per-Element Page Strategy for Multi-Core Writes

## Problem: Race Conditions in Multi-Core Writes

When multiple Tensix cores need to write to different elements of the same tensor, naïve approaches cause race conditions:

### ❌ Approach 1: Direct Write (FAILS)
```cpp
// Multiple cores trying to write to same buffer
uint32_t value = compute_value();
write_to_dram(buffer_addr + offset, value);  // RACE CONDITION!
```
**Problem**: No synchronization between cores → data corruption

### ❌ Approach 2: Read-Modify-Write (FAILS)
```cpp
// Read entire array
uint32_t data[E];
read_from_dram(buffer_addr, data, E * sizeof(uint32_t));

// Modify one element
data[local_expert_id] = compute_value();

// Write back entire array
write_to_dram(buffer_addr, data, E * sizeof(uint32_t));  // RACE CONDITION!
```
**Problem**: Time window between read and write allows other cores to overwrite → lost updates

---

## ✅ Solution: Per-Element Pages

**Key Insight**: TT-Metal's TensorAccessor uses **page-based addressing**. By making each element its own page, cores can write independently without conflicts.

### Concept

```
Traditional 1D tensor (E,):
┌─────────────────────────────────────┐
│ e0  e1  e2  e3  e4  e5  e6  e7 ... │  Single page = entire row
└─────────────────────────────────────┘
↑ All cores compete for same page

Per-element 2D tensor (E, 1):
┌────┐
│ e0 │  Page 0 → Core 0 writes here
├────┤
│ e1 │  Page 1 → Core 1 writes here
├────┤
│ e2 │  Page 2 → Core 2 writes here
├────┤
│ e3 │  Page 3 → Core 3 writes here
└────┘
↑ Each core has its own page
```

### Implementation

#### 1. Tensor Shape Change

**Before (1D tensor):**
```cpp
// Output: num_routed_tokens (E/D,)
ttnn::Shape num_routed_shape({num_local_experts});
```

**After (2D tensor):**
```cpp
// Output: num_routed_tokens (E/D, 1)
ttnn::Shape num_routed_shape({num_local_experts, 1});
```

#### 2. TensorAccessor with Explicit Page Size

**Kernel code:**
```cpp
// Per-element page: page_size = sizeof(uint32_t) = 4 bytes
const auto num_routed_accessor = TensorAccessor(
    num_routed_accessor_args,
    num_routed_tokens_addr,
    sizeof(uint32_t)  // ← Explicit page size (one element)
);
```

**Comparison with other accessors:**
```cpp
// Row-based pages (multiple elements per page)
const auto experts_accessor = TensorAccessor(
    experts_accessor_args,
    selected_experts_addr,
    top_k * sizeof(uint32_t)  // Full row
);

const auto routed_tokens_accessor = TensorAccessor(
    routed_tokens_accessor_args,
    routed_tokens_addr,
    max_tokens_per_expert * sizeof(uint32_t)  // Full row
);

// Per-element page (ONE element per page)
const auto num_routed_accessor = TensorAccessor(
    num_routed_accessor_args,
    num_routed_tokens_addr,
    sizeof(uint32_t)  // Single element ← KEY DIFFERENCE
);
```

#### 3. Race-Free Write Using Page ID

**Kernel write pattern:**
```cpp
// Each core writes to its assigned expert's page
uint32_t local_expert_id = get_arg_val<uint32_t>(11);  // Core's assigned expert

// Compute value in L1
cb_reserve_back(cb_num_routed, 1);
uint32_t l1_addr = get_write_ptr(cb_num_routed);
uint32_t* num_routed_ptr = reinterpret_cast<uint32_t*>(l1_addr);
*num_routed_ptr = token_count;  // Write value to L1
cb_push_back(cb_num_routed, 1);

// Write to DRAM using page ID = local_expert_id
uint64_t num_routed_noc_addr = get_noc_addr(local_expert_id, num_routed_accessor);
//                                          ^^^^^^^^^^^^^^^^
//                                          Page ID = Expert ID
//                                          Each core has unique page!
noc_async_write(l1_addr, num_routed_noc_addr, sizeof(uint32_t));
noc_async_write_barrier();
```

**Why this is race-free:**
- Core 0 writes to page 0 (expert 0)
- Core 1 writes to page 1 (expert 1)
- Core 2 writes to page 2 (expert 2)
- ...
- **No page overlap** → No race condition → No synchronization needed

---

## Memory Layout Details

### ROW_MAJOR Layout with Per-Element Pages

For a 2D tensor `(E/D, 1)` with ROW_MAJOR layout:

```
Physical DRAM Memory:
┌────────┬────────┬────────┬────────┬─────
│ Page 0 │ Page 1 │ Page 2 │ Page 3 │ ...
│ (4B)   │ (4B)   │ (4B)   │ (4B)   │
└────────┴────────┴────────┴────────┴─────
    ↑        ↑        ↑        ↑
  Row 0    Row 1    Row 2    Row 3
Element 0  Elem 1   Elem 2   Elem 3
```

**TensorAccessor mapping:**
```cpp
get_noc_addr(0, accessor) → Address of page 0 (row 0, element 0)
get_noc_addr(1, accessor) → Address of page 1 (row 1, element 1)
get_noc_addr(2, accessor) → Address of page 2 (row 2, element 2)
```

**Key properties:**
- **Page ID = Row ID** for ROW_MAJOR layout
- Each row contains exactly 1 element (width = 1)
- Each page is exactly `sizeof(uint32_t)` = 4 bytes
- Pages are contiguous in DRAM

---

## Python Side Handling

### Test Code Adjustment

**Before:**
```python
# Expected 1D tensor
num_routed_torch = torch.tensor([5, 3, 7, 2])  # (E/D,)
assert num_routed.shape[0] == experts_per_device
```

**After:**
```python
# Handle 2D tensor and squeeze
num_routed_torch = torch.tensor([5, 3, 7, 2])  # (E/D,)

# Verify output shape is 2D
assert num_routed.shape[0] == experts_per_device  # (E/D, 1)
assert num_routed.shape[1] == 1

# Squeeze to 1D for comparison
num_routed_torch = num_routed_torch.squeeze(-1)
```

### Production Usage

```python
# API returns 2D tensor (E/D, 1)
num_routed_tokens, routed_tokens, routed_weights = ttnn.prepare_moe_routing_tensors(...)

# Squeeze to 1D if needed for subsequent operations
num_routed_tokens = num_routed_tokens.squeeze(-1)  # (E/D, 1) → (E/D,)
```

---

## Performance Characteristics

### Advantages

1. **Zero Synchronization Overhead**
   - No atomics, mutexes, or barriers needed
   - Cores execute completely independently
   - Linear speedup with number of cores (up to E/D)

2. **Optimal Memory Access**
   - Each core writes exactly once per page
   - No redundant reads or writes
   - Minimal DRAM bandwidth usage

3. **Simple Programming Model**
   - No complex synchronization logic in kernel
   - Easy to reason about correctness
   - No risk of deadlocks or race conditions

### Memory Overhead

**Minimal overhead for small tensors:**
```
1D tensor (E/D,):       E/D * 4 bytes
2D tensor (E/D, 1):     E/D * 4 bytes + shape metadata
```

For typical cases (E=8-128, D=8):
- E=8, D=8: 1 expert/device → 4 bytes total per device
- E=32, D=8: 4 experts/device → 16 bytes total per device
- E=128, D=8: 16 experts/device → 64 bytes total per device

**Negligible overhead** compared to other tensors in MoE pipeline (MB-GB scale).

---

## When to Use This Pattern

### ✅ Use Per-Element Pages When:

1. **Multiple cores write to different elements**
   - Each core has unique element to update
   - Elements are independently computed
   - No reduction/accumulation across cores

2. **Output tensor is small**
   - Shape overhead is negligible
   - Total size < 1KB typically fine

3. **Write-once semantics**
   - Each element written exactly once
   - No read-modify-update pattern needed

### ❌ Don't Use Per-Element Pages When:

1. **Large output tensors**
   - Shape change adds significant overhead
   - Better to use atomic operations or different parallelization strategy

2. **Reduction operations**
   - Need to accumulate from multiple cores
   - Requires different approach (local reduction + allreduce)

3. **Read-heavy workloads**
   - Pattern optimized for write-once
   - Normal tensors fine for read operations

---

## Example Use Cases in MoE

### ✅ Good Fit: `num_routed_tokens` in `prepare_moe_routing_tensors`

```cpp
// Output: (E/D, 1) uint32 - one counter per expert
// Each core counts tokens for one expert
// Write-once, small size (4-64 bytes typical)
```

**Why it works:**
- E/D cores, E/D experts → 1:1 mapping
- Each core computes one element independently
- Very small output (E/D integers)

### ❌ Poor Fit: `intermediate_output` in `projection_to_intermediate`

```cpp
// Output: (num_tokens, intermediate_size) bf16
// Size: potentially MB scale
// Better approach: shard by expert, use standard matmul
```

**Why it doesn't work:**
- Huge tensor → shape overhead unacceptable
- Each expert processes multiple tokens
- Standard sharded matmul more efficient

---

## Implementation Checklist

When implementing per-element pages:

- [ ] Change tensor shape from 1D `(N,)` to 2D `(N, 1)`
- [ ] Use explicit page size `sizeof(element_type)` in TensorAccessor
- [ ] Update TensorSpec with 2D shape in operation implementation
- [ ] Map core ID to page ID in runtime args
- [ ] Write using `get_noc_addr(page_id, accessor)` where `page_id = core_id`
- [ ] Update Python tests to handle 2D shape (squeeze if needed)
- [ ] Verify no race conditions in multi-device tests
- [ ] Document the 2D shape requirement in API docs

---

## References

- **Implementation**: `ttnn/cpp/ttnn/operations/experimental/moe/prepare_moe_routing_tensors/`
- **Kernel**: `device/kernels/dataflow/reader_writer_moe_routing.cpp:68`
- **Test**: `models/demos/qwen3/tests/test_moe_routing_tensors.py`
- **API Docs**: `models/demos/qwen3/docs/MoE_implementation_plan.md`
