# Qwen3 MoE Documentation

This directory contains technical documentation for the Qwen3 Mixture of Experts (MoE) implementation.

## Documentation Files

### Implementation Documentation
- **[MOE_IMPLEMENTATION_SUMMARY.md](MOE_IMPLEMENTATION_SUMMARY.md)** - Complete implementation summary for MoE operations including `prepare_moe_mapping_tensor` and `prepare_moe_routing_tensors`
- **[MoE_implementation_plan.md](MoE_implementation_plan.md)** - Original implementation planning document
- **[MOE_TESTING_STATUS.md](MOE_TESTING_STATUS.md)** - Detailed testing status and results

## Quick Links

### Source Code
- MoE TT-NN Implementation: `../tt/moe.py`
- MoE Tests: `../tests/test_moe*.py`
- TTNN Operations: `/ttnn/cpp/ttnn/operations/experimental/moe/`

### Key Operations

#### 1. `ttnn.prepare_moe_mapping_tensor()`
Converts sparse MoE expert selection to dense format for computation.

#### 2. `ttnn.prepare_moe_routing_tensors()`
Creates efficient routing tensors for expert-parallel computation.

## Status Summary

| Component | Status | Test Coverage |
|-----------|--------|---------------|
| prepare_moe_mapping_tensor | ✅ Complete | 46 tests passing |
| prepare_moe_routing_tensors | ✅ Complete | 60 tests passing |
| pytest fixtures | ✅ Optimized | Module-scoped caching |

## Running Tests

```bash
# Run all MoE tests
cd models/demos/qwen3
pytest tests/test_moe*.py -v

# Run specific operation tests
pytest tests/test_moe_mapping.py -v
pytest tests/test_moe_routing_tensors.py -v
```