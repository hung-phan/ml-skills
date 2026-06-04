---
name: gpu-lang
description: Python-based GPU kernel languages for writing custom high-performance ops. Use when PyTorch is too slow and you need fused kernels, custom attention, or quantized operations without writing CUDA C++.
---

# GPU Kernel Languages

## Why This Exists

PyTorch operators are general-purpose — they launch a separate CUDA kernel per op. For performance-critical code (attention, quantized matmul, fused activations), you need to fuse multiple ops into one kernel that keeps data in registers/shared memory. These tools let you write those kernels in Python.

## Skills

| Skill | Control Level | Best For |
|-------|--------------|----------|
| [triton](triton/) | Block-level, auto scheduling | Research kernels, fast prototyping, fused ops |
| [tilelang](tilelang/) | Shared memory + warp-level + pipelining | FlashAttention-level kernels, production performance in Python |
| CUDA (no skill — use C/C++) | Everything manual | Absolute peak, when Python tools aren't enough |

## Comparison

| | Triton | TileLang | CUDA (reference) |
|--|--------|----------|------------------|
| **Language** | Python | Python (TVM DSL) | C/C++ |
| **Shared memory** | Implicit (compiler decides) | Explicit (`T.alloc_shared`) | Manual `__shared__` |
| **Warp-level ops** | Hidden | Explicit (`T.WarpPolicy`) | Manual MMA/WMMA intrinsics |
| **Software pipelining** | No control | `T.Pipelined(num_stages=N)` | Manual `cp.async` + barriers |
| **Performance vs peak** | ~90-95% | ~95-100% | 100% |
| **Lines for attention** | ~80 | ~80 | ~500+ (or use CUTLASS) |
| **Compilation** | Triton → PTX | TVM/TIR → CUDA → PTX | nvcc → PTX |
| **Autotuning** | `@triton.autotune` | Grid search on params | Manual or Autotuning frameworks |
| **Learning curve** | Hours | Days | Weeks-months |
| **When to drop to this** | Default start | Need more control than Triton | Need intrinsics, inline PTX, or CUTLASS |

## When to Use What

| Scenario | Tool |
|----------|------|
| First custom kernel, prototype | **Triton** — easiest, most docs |
| Need shared memory / pipeline control | **TileLang** — Python with CUTLASS-level control |
| Fused elementwise ops | **Triton** — one-liner with `tl.load/store` |
| FlashAttention-class kernels | **TileLang** — explicit warp scheduling + multi-stage |
| Already have working Triton, need more perf | Try **TileLang** rewrite |
| torch.compile handles it fine | Neither — don't write kernels you don't need |

## References

- Triton: https://github.com/triton-lang/triton
- TileLang: https://github.com/tile-ai/tilelang
- NVIDIA CUTLASS (C++ alternative): https://github.com/NVIDIA/cutlass
