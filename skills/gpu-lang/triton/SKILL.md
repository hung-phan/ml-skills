---
name: triton-lang
description: |
  OpenAI Triton GPU kernel language for writing high-performance custom deep learning primitives.
  Triggers: triton kernel, custom GPU op, fused kernel, tl.load, tl.store, tl.dot, @triton.jit,
  triton autotune, block-level programming, CUDA alternative, custom attention kernel, quantized ops,
  tensor cores from Python, kernel fusion, GPU programming without CUDA
---

# Triton (GPU Kernel Language)

## Why This Exists

**Problem:** CUDA is too low-level for ML researchers (manual thread management, shared memory, synchronization barriers). PyTorch is too high-level for custom ops (can't fuse kernels, no control over memory access patterns). The gap between "I know the math" and "I have a fast GPU kernel" is enormous.

**Key Insight:** Block-level programming with automatic scheduling. You write operations on blocks of data (not individual threads), and the Triton compiler handles thread mapping, shared memory allocation, memory coalescing, and scheduling automatically. Think of it as "NumPy-like syntax that compiles to GPU machine code."

**Reach for this when:**
- Need fused kernels (softmax + dropout + mask in one pass)
- Writing custom attention variants (sparse, linear, sliding window)
- Implementing quantized operations (INT4/INT8/FP8 matmuls)
- PyTorch's existing ops leave performance on the table (extra memory reads/writes)
- Want tensor core utilization without writing PTX assembly
- Building custom training kernels (fused Adam, layer norm + residual)

## Core Concepts

### @triton.jit — The Kernel Decorator

Every Triton kernel is a Python function decorated with `@triton.jit`. It compiles to GPU code at first call.

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This runs on GPU — one instance per "program"
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x * 2, mask=mask)
```

**Key rules:**
- `tl.constexpr` parameters are compile-time constants (used for block shapes)
- `tl.program_id(axis)` identifies which block this instance handles (like CUDA blockIdx)
- All operations work on blocks of data, not scalar values

### Memory Operations: tl.load / tl.store

```python
# Load a 1D block
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements  # Guard against OOB
data = tl.load(ptr + offsets, mask=mask, other=0.0)

# Load a 2D block (for matmul)
row_offsets = tl.arange(0, BLOCK_M)[:, None]  # Column vector
col_offsets = tl.arange(0, BLOCK_N)[None, :]  # Row vector
ptrs = base_ptr + row_offsets * stride_row + col_offsets * stride_col
block = tl.load(ptrs, mask=mask_2d, other=0.0)

# Store results
tl.store(out_ptr + offsets, result, mask=mask)
```

### tl.dot — Tensor Core Matrix Multiply

```python
# Accumulate C += A @ B using tensor cores
# A: [BLOCK_M, BLOCK_K], B: [BLOCK_K, BLOCK_N]
# acc: [BLOCK_M, BLOCK_N] in float32 for precision
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
acc = tl.dot(a, b, acc)  # Uses tensor cores automatically
```

### Reductions

```python
row_max = tl.max(data, axis=0)    # Max across axis
row_sum = tl.sum(data, axis=0)    # Sum across axis
```

## Complete Examples

### 1. Vector Addition

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### 2. Fused Softmax

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    # Numerically stable softmax
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back
    output_row_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_ptr + col_offsets, softmax_output, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y
```

### 3. Tiled Matrix Multiplication

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # L2 cache optimization: grouped ordering
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c
```

### 4. Fused Attention Kernel (Simplified Skeleton)

```python
@triton.jit
def _fused_attention_fwd(Q, K, V, Out, sm_scale,
                         stride_qz, stride_qh, stride_qm, stride_qk,
                         stride_oz, stride_oh, stride_om, stride_ok,
                         N_CTX, HEAD_DIM: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Load Q block — stays in SRAM for entire inner loop
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

    # Running softmax state
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Iterate over K/V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        # Load K block, compute QK^T
        k = tl.load(K + off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        # Causal mask
        if CAUSAL:
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(mask, qk, float("-inf"))
        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        # Load V block, accumulate
        v = tl.load(V + off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_ij

    # Normalize and store
    acc = acc / l_i[:, None]
    tl.store(Out + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok, acc.to(tl.float16))
```

## Autotuning

The `@triton.autotune` decorator benchmarks multiple configurations and picks the fastest:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],  # Re-tune when this arg changes
)
@triton.jit
def my_kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

**Parameters:**
- `configs`: List of `triton.Config` dicts mapping constexpr names to values
- `key`: Argument names whose values trigger re-tuning
- `num_warps`: Threads per block = num_warps × 32
- `num_stages`: Software pipeline stages (higher = more memory, better latency hiding)
- `TRITON_PRINT_AUTOTUNING=1` env var prints selected config

## Integration with PyTorch

### Custom autograd.Function wrapping a Triton kernel

```python
class TritonSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        y = torch.empty_like(x)
        softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0),
                                  n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        # dy * y - y * sum(dy * y)  — standard softmax backward
        grad_input = grad_output * y - y * (grad_output * y).sum(dim=-1, keepdim=True)
        return grad_input

triton_softmax = TritonSoftmax.apply
```

### torch.compile + Triton

PyTorch 2.0+ uses Triton as its backend for `torch.compile`:

```python
@torch.compile
def fused_op(x, y):
    return torch.softmax(x + y, dim=-1)

# torch.compile generates Triton kernels automatically
# To see generated code: TORCH_LOGS="output_code" python script.py
```

## Real-World Usage

| Project | What it uses Triton for |
|---------|------------------------|
| **Flash Attention** (Tri Dao) | Fused multi-head attention with tiling, O(N) memory |
| **Unsloth** | Fused cross-entropy, RoPE, SwiGLU, QLoRA dequant — 2x faster fine-tuning |
| **xformers** (Meta) | Memory-efficient attention, fused dropout + bias |
| **vLLM** | PagedAttention kernel, fused activation + quant |
| **torch.compile** | Backend code generation for fused PyTorch ops |
| **DeepSpeed** | Inference kernels, quantization |
| **bitsandbytes** | INT4/INT8 quantized matmul kernels |

## Performance Tips

1. **Memory coalescing** — Adjacent threads should access adjacent memory. Use `tl.arange` offsets that map naturally to contiguous addresses.

2. **Power-of-2 block sizes** — Required by Triton. Use `triton.next_power_of_2(n)` and mask out-of-bounds.

3. **Accumulate in FP32** — Use `tl.zeros(..., dtype=tl.float32)` for accumulators, cast to FP16 only at store time.

4. **L2 cache reuse** — For matmul, use grouped program ordering (GROUP_SIZE_M) to keep data in L2 across nearby blocks.

5. **Pipeline stages** — Higher `num_stages` (3-4) hides memory latency but uses more registers/SMEM.

6. **Minimize loads** — Keep data in registers/SRAM. Load Q once, iterate over K/V (Flash Attention pattern).

7. **Use tl.dot** — It maps to tensor cores. Minimum 16×16 blocks for FP16 tensor core utilization.

8. **Avoid bank conflicts** — The compiler handles shared memory layout, but awkward strides can still cause issues. Profile with `ncu`.

## When to Use vs Alternatives

| Approach | Use when | Downsides |
|----------|----------|-----------|
| **Triton** | Custom fused ops, attention variants, quantized kernels, prototyping GPU code | Limited to block programs, no warp-level control, NVIDIA CC 8.0+ required |
| **CUDA C++** | Need warp shuffles, cooperative groups, maximum control | 10-100x more code, manual thread/memory management |
| **torch.compile** | Standard PyTorch ops, want automatic fusion | Can't express custom algorithms, limited control |
| **C++ extensions** | CPU ops, simple CUDA wrappers | No auto-scheduling, manual memory management |
| **cuBLAS/cuDNN** | Standard GEMM/conv that vendor already optimized | Can't fuse, proprietary, inflexible |

**Decision heuristic:** If torch.compile doesn't fuse it and you'd need >50 lines of CUDA, write it in Triton.

## Gotchas

1. **Block sizes MUST be powers of 2** — `tl.arange(0, N)` requires N to be a compile-time power of 2.
2. **No dynamic control flow** — Branches must be uniform across the block or use `tl.where`.
3. **First call compiles** — JIT compilation adds latency on first invocation. Use `.warmup()` if needed.
4. **FP precision** — `tl.exp` uses fast math (`__expf`). For exact results, use `tl.math.exp`.
5. **Autotuning caches** — Stored in `~/.triton/cache`. Delete if configs are stale.
6. **Debug with interpreter** — Set `TRITON_INTERPRET=1` to run on CPU with Python breakpoints.
7. **AMD support** — Works on ROCm 6.2+ (gfx940+) but performance tuning differs from NVIDIA.

## References

- **GitHub**: https://github.com/triton-lang/triton
- **Documentation**: https://triton-lang.org
- **Paper**: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations (MAPL 2019)](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- **Flash Attention paper**: https://arxiv.org/abs/2205.14135
- **Triton Puzzles** (learn without GPU): https://github.com/srush/Triton-Puzzles
- **Tutorials**: https://triton-lang.org/main/getting-started/tutorials/
- **Triton Conference 2025 videos**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLc_vA1r0qoiTjlKsBFqFG6P_OOVHhOTgK)
