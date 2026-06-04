---
name: tilelang
description: Write high-performance GPU kernels (GEMM, FlashAttention, MLA, convolutions) in Python with explicit control over shared memory, warp-level ops, and multi-stage pipelining — compiling to CUDA/HIP via TVM/TIR
---

# TileLang

## Why This Exists

There's a gap in GPU kernel development tooling:

- **Triton** gives you Python syntax but abstracts away shared memory, warp-level operations, and pipeline staging. The compiler makes layout decisions you can't override. When you need FlashAttention-level control, you hit a wall.
- **CUTLASS/CuTe** gives you total control over every register layout, warp partition, and memory stage — but requires verbose C++ template metaprogramming that's hostile to iteration.

**TileLang bridges this gap.** You write Pythonic DSL code with explicit control over where buffers live (shared memory vs registers), how warps split work, and how deep the software pipeline runs — and a layout inference compiler pass handles the thread mappings, register layouts, swizzles, and warp specialization lowering that would be hundreds of lines of CuTe.

The result: 80 lines of Python for a kernel that matches hand-tuned CUTLASS performance (demonstrated on FlashMLA decode at H100 parity with FlashMLA).

## Key Concepts

### 1. Tile-Based Programming Model

A **tile** is a shaped chunk of data (e.g., block_M × block_K) owned by a thread block, warp, or thread. You think in tiles across the memory hierarchy:

```
Global Memory → T.copy → Shared Memory (T.alloc_shared)
Shared Memory → T.copy → Registers (T.alloc_fragment)
Registers → T.gemm/T.Parallel → Compute
Registers → T.copy → Global Memory (write-back)
```

### 2. Explicit Shared Memory Management

Unlike Triton where the compiler manages shared memory, you declare buffers at each level:

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)   # shared memory
B_shared = T.alloc_shared((block_K, block_N), dtype)   # shared memory
C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)  # registers
```

### 3. Warp-Level Primitives

`T.gemm` accepts a `policy` argument that controls how warps partition work:

- `T.GemmWarpPolicy.FullRow` — each warp computes an entire row of the output tile
- `T.GemmWarpPolicy.FullCol` — each warp computes an entire column slab (critical for MLA where you split acc_o across warpgroups)

### 4. Multi-Stage Software Pipelining

Explicit pipeline depth via `T.Pipelined`:

```python
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[...], A_shared)  # becomes cp.async automatically
    T.copy(B[...], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

`num_stages=3` means triple buffering — loads for ko+1 and ko+2 are in flight while ko computes.

### 5. Compilation via TVM/TIR

TileLang programs lower through TVM's Tensor IR (TIR) to CUDA/HIP/WebGPU/CPU. A **layout inference** pass propagates constraints from your `T.gemm` policy annotations through the program, deriving register layouts, swizzled shared memory patterns, and warp-specialized producer/consumer code automatically.

## Primitive Vocabulary

| Category | Primitives |
|----------|-----------|
| Allocate | `T.alloc_shared`, `T.alloc_fragment`, `T.alloc_local` |
| Move/Init | `T.copy(src, dst)`, `T.clear`, `T.fill` |
| Compute | `T.gemm(A, B, C, transpose_B=, policy=)`, `T.Parallel(d0, d1)` for elementwise |
| Reduce | `T.reduce_max`, `T.reduce_sum` |
| Math | `T.exp`, `T.exp2`, `T.max`, `T.infinity`, `T.if_then_else` |
| Schedule | `T.Pipelined(extent, num_stages=)`, `T.use_swizzle(panel_size=)` |
| Layout | `T.annotate_layout(...)` for bank-conflict avoidance |
| Debug | `T.print(buf)`, `kernel.get_kernel_source()` |
| Shape | `T.const('M, N, K')` for compile-time constants |

## Code Examples

### Basic GEMM with ReLU

```python
import tilelang
import tilelang.language as T
import torch

@tilelang.jit
def matmul_relu(
    A, B,
    block_M: int = 128, block_N: int = 128, block_K: int = 64,
    dtype: T.dtype = T.float16, accum_dtype: T.dtype = T.float32,
):
    M, N, K = T.const('M, N, K')
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty([M, N], dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.use_swizzle(panel_size=10)
        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        T.copy(C_local, C[by * block_M, bx * block_N])
    return C

# Usage
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = matmul_relu(a, b)

# Profile
kernel = matmul_relu.compile(a, b)
print(kernel.get_kernel_source())  # see generated CUDA
profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
print(f"Latency: {profiler.do_bench()} ms")
```

### FlashAttention Kernel (Simplified)

```python
@T.prim_func
def flash_attention(Q, K, V, Output):
    with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128) as (bx, by, bz):
        Q_shared   = T.alloc_shared([block_M, dim], dtype)
        K_shared   = T.alloc_shared([block_N, dim], dtype)
        V_shared   = T.alloc_shared([block_N, dim], dtype)
        acc_s      = T.alloc_fragment([block_M, block_N], accum_dtype)
        acc_o      = T.alloc_fragment([block_M, dim], accum_dtype)
        scores_max = T.alloc_fragment([block_M], accum_dtype)
        logsum     = T.alloc_fragment([block_M], accum_dtype)

        T.copy(Q[bz, bx * block_M:(bx+1) * block_M, by, :], Q_shared)
        T.fill(acc_o, 0)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.fill(logsum, 0)

        for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=2):
            T.copy(K[bz, k*block_N:(k+1)*block_N, by, :], K_shared)
            T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                   policy=T.GemmWarpPolicy.FullRow)
            T.copy(V[bz, k*block_N:(k+1)*block_N, by, :], V_shared)

            # Online softmax: rescale, exp, accumulate
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            T.copy(scores_max, scores_max_prev)
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # ... rescale acc_o, exponentiate acc_s, gemm with V ...
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        # Normalize by logsum
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] /= logsum[i]
        T.copy(acc_o, Output[bz, bx*block_M:(bx+1)*block_M, by, :])
```

### Equivalent Triton GEMM (for comparison)

```python
# Triton — compiler decides shared memory and layout
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k + tl.arange(0, BLOCK_K))[None, :])
        b = tl.load(b_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)  # compiler decides shared mem, layout, pipeline
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16))
```

**Key difference:** In Triton, `tl.dot` hides shared memory staging, pipeline depth, and warp policy. In TileLang, you write `T.alloc_shared` + `T.Pipelined` + `T.gemm(policy=...)` explicitly.

## Comparison Table

| Aspect | TileLang | Triton | CUTLASS/CuTe |
|--------|----------|--------|--------------|
| **Syntax** | Python DSL | Python DSL | C++ templates |
| **Shared Memory** | Explicit (`T.alloc_shared`) | Compiler-managed (hidden) | Explicit (manual layout) |
| **Warp Control** | Policy annotation (`FullRow`/`FullCol`) | None (compiler decides) | Full manual control |
| **Pipelining** | `T.Pipelined(num_stages=)` | `tl.range(num_stages=)` or compiler | Manual double/triple buffering |
| **Layout Inference** | Automatic from policy + copy | Automatic (no override) | Manual (you write every layout) |
| **Performance** | Near-CUTLASS (matches FlashMLA on H100) | 80-95% of CUTLASS typically | Baseline (hand-tuned) |
| **Lines of Code** | ~80 for MLA decode | ~60 for basic attention | ~500-1000+ for equivalent |
| **Compilation** | TVM/TIR → CUDA/HIP | MLIR → PTX | Direct C++ → PTX |
| **Backend Support** | NVIDIA, AMD, CPU, WebGPU, Ascend, Metal | NVIDIA (primary), AMD | NVIDIA only |
| **Iteration Speed** | Fast (Python, `pip install`) | Fast (Python) | Slow (C++ compile times) |
| **Debug Tools** | `T.print`, layout plotter, source dump | `print`, breakpoint | printf, NSight |
| **Best For** | Complex fusion with layout control | Simple fusion, elementwise | Maximum performance, production |

## When to Use TileLang

**Use TileLang when you need:**
- Shared memory control + Python syntax (the "Triton hit a wall" moment)
- FlashAttention-level kernels without writing C++ (attention variants, MLA, linear attention)
- Warp-level partitioning decisions (splitting accumulators across warpgroups for register pressure)
- Multi-stage software pipelining you can reason about (num_stages as a visible knob)
- Cross-backend portability (same kernel for NVIDIA H100 + AMD MI300X)
- Covering a config your hand-tuned kernel doesn't support (e.g., unusual hidden dims)

**Stick with Triton when:**
- Simple elementwise fusion or light reductions
- The compiler's layout/pipeline decisions are good enough
- You don't need explicit shared memory control

**Use CUTLASS/CuTe when:**
- Maximum absolute performance on a single NVIDIA target
- You need control beyond what TileLang's layout inference provides
- Production kernels with validated correctness across all edge cases

## Installation

```bash
pip install tilelang              # prebuilt wheel (easiest)
pip install tilelang -f https://tile-ai.github.io/whl/nightly  # nightly

# From source (for compiler development)
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang && pip install -e . -v
```

**Tested devices:** H100, A100, V100, RTX 4090/3090/A6000 (NVIDIA); MI250, MI300X (AMD); Apple Metal.

## Gotchas

1. **Register pressure is YOUR problem.** TileLang gives you control but won't save you from spilling. On Hopper, wgmma requires M≥64 per warpgroup — if your accumulator (e.g., 64×512 for MLA) doesn't fit, split across warpgroups using `FullCol` policy.

2. **`T.Pipelined` num_stages trades shared memory for latency hiding.** More stages = more overlap but more SMEM consumed. On H100 with 228KB shared memory, 3 stages is typical for GEMM; attention kernels with large tiles may need 2.

3. **`T.copy` inside `T.Pipelined` becomes `cp.async` automatically.** Outside the pipeline loop, it's a synchronous copy. Don't mix up placement.

4. **Layout inference propagates from `T.gemm` policy annotations.** If your kernel computes wrong values, check that your policy choice matches the data flow (FullRow for online softmax where each warp needs all columns of acc_s; FullCol when splitting the output dim).

5. **`T.use_swizzle` is free performance.** Always try `panel_size=4-10` for GEMM-like kernels. It reorders thread block scheduling for L2 locality.

6. **Type casting between fragment dtypes requires explicit `T.copy`.** E.g., `T.copy(acc_s, acc_s_cast)` to go from fp32 scores to fp16 before the second GEMM.

7. **Multiple kernels in one function execute sequentially.** Each `with T.Kernel(...)` block is a separate kernel launch.

## References

- GitHub: https://github.com/tile-ai/tilelang (6.4k+ stars, MIT license)
- Documentation: https://tilelang.com/
- Puzzles (interactive learning): https://github.com/tile-ai/tilelang-puzzles
- Benchmarks: https://github.com/tile-ai/tilelang-benchmark
- Used in: [BitBLAS](https://github.com/microsoft/BitBLAS), [AttentionEngine](https://github.com/microsoft/AttentionEngine)
- MLA Decode example (80 lines, FlashMLA parity): https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/example_mla_decode.py
- FlashAttention examples: https://github.com/tile-ai/tilelang/tree/main/examples/flash_attention
- CuTeDSL backend (CUTLASS codegen): https://github.com/tile-ai/tilelang/pull/1421
- DeepWiki (AI-generated docs): https://deepwiki.com/tile-ai/tilelang
- AtlasCloud tutorial: https://www.atlascloud.ai/blog/guides/writing-high-performance-kernels-in-tilelang
- Developed at Peking University, with Microsoft Research collaboration
