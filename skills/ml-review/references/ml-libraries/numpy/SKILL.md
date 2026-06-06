---
name: numpy
description: NumPy arrays and numerical computing — broadcasting, vectorization, linear algebra, matrix operations, and memory layout. Use when working with N-dimensional arrays, optimizing Python loops via vectorization, or interfacing with PyTorch/JAX tensors.
---

# NumPy Patterns Skill

## Why This Exists

1. **Problem solved**: Fast numerical computation without Python's per-element loop overhead. NumPy provides fixed-type contiguous arrays that execute operations in compiled C/Fortran, giving 10-100x speedups over equivalent Python loops. Without it, any matrix math, signal processing, or batch computation would be impractically slow.

2. **When to pick this over alternatives**: Choose numpy for algorithm implementation, linear algebra, and any computation that stays on CPU. Choose pandas/polars when you need labeled columns and SQL-like operations. Choose PyTorch/JAX when you need GPU acceleration or automatic differentiation. NumPy is the foundation — virtually every other library converts to/from numpy arrays.

3. **Mental model**: numpy = typed, fixed-size, contiguous memory blocks with broadcasting arithmetic. An ndarray is a view into a flat buffer with shape/strides metadata. Operations broadcast smaller arrays to match larger ones (right-to-left dimension alignment), and ufuncs apply element-wise in C without Python interpreter overhead.

Reference for idiomatic NumPy usage: array manipulation, broadcasting, linear algebra, vectorization, and common pitfalls.

---

## Array Creation & Reshaping

```python
import numpy as np

# Creation
a = np.array([1, 2, 3])                    # from list
z = np.zeros((3, 4))                        # shape (3,4) of zeros
o = np.ones((2, 3, 4), dtype=np.float32)    # explicit dtype
e = np.empty((5,))                          # uninitialized (fast)
r = np.arange(0, 10, 0.5)                  # like range, float step OK
l = np.linspace(0, 1, 50)                  # 50 evenly spaced in [0,1]
i = np.eye(4)                              # 4x4 identity
f = np.full((3, 3), fill_value=7)          # all 7s

# Reshaping (returns view when possible)
b = a.reshape(1, 3)          # (3,) -> (1, 3)
c = a.reshape(-1, 1)         # (3,) -> (3, 1), -1 infers dimension
d = z.reshape(2, 6)          # (3, 4) -> (2, 6), total elements must match
flat = z.ravel()             # flatten to 1D (view if contiguous)
flat_copy = z.flatten()      # always returns copy

# Stacking
np.vstack([a, a])            # (2, 3)
np.hstack([a, a])            # (6,)
np.stack([a, a], axis=0)     # (2, 3) — new axis
np.concatenate([z, z], axis=1)  # (3, 8)
```

---

## Broadcasting Rules

Broadcasting aligns shapes right-to-left. Dimensions are compatible when:
1. They are equal, OR
2. One of them is 1

The dimension with size 1 is stretched to match the other.

```
Shape examples:
(3, 4) + (4,)     → (3, 4)     # (4,) becomes (1, 4) then broadcasts
(3, 1) + (1, 4)   → (3, 4)     # both stretched
(2, 3, 4) + (4,)  → (2, 3, 4)  # scalar-like broadcast on last dim
(5, 1, 3) + (1, 4, 3) → (5, 4, 3)

# FAILS:
(3, 4) + (3,)     → ERROR       # trailing dims 4 vs 3, neither is 1
(2, 3) + (2, 4)   → ERROR       # axis 1: 3 vs 4
```

```python
# Practical: subtract column means
data = np.random.randn(100, 5)          # (100, 5)
means = data.mean(axis=0)               # (5,)
centered = data - means                 # broadcasts (5,) -> (1, 5) -> (100, 5)

# Practical: outer product via broadcasting
x = np.arange(5).reshape(5, 1)          # (5, 1)
y = np.arange(3).reshape(1, 3)          # (1, 3)
outer = x * y                           # (5, 3)

# Practical: pairwise distances
A = np.random.randn(100, 3)             # 100 points in 3D
B = np.random.randn(50, 3)              # 50 points in 3D
diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # (100, 50, 3)
dists = np.linalg.norm(diff, axis=2)               # (100, 50)
```

---

## np.newaxis & expand_dims Patterns

```python
a = np.array([1, 2, 3])     # shape (3,)

# Add dimensions for broadcasting
row = a[np.newaxis, :]       # (1, 3) — row vector
col = a[:, np.newaxis]       # (3, 1) — column vector

# Equivalent with expand_dims
row = np.expand_dims(a, 0)   # (1, 3)
col = np.expand_dims(a, 1)   # (3, 1)

# Multiple axes (NumPy >= 1.18)
np.expand_dims(a, (0, 2))   # (1, 3, 1)

# Common pattern: make 1D array broadcast with 2D
weights = np.array([0.2, 0.5, 0.3])          # (3,)
images = np.random.randn(10, 3, 64, 64)      # (N, C, H, W)
# Weight channels:
weighted = images * weights[np.newaxis, :, np.newaxis, np.newaxis]  # (10,3,64,64)
# Or reshape:
weighted = images * weights.reshape(1, 3, 1, 1)
```

---

## Linear Algebra

```python
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# Matrix multiply (all equivalent for 2D)
C = A @ B                    # (3, 5) — preferred syntax
C = np.matmul(A, B)          # same
C = np.dot(A, B)             # same for 2D, different semantics for nD

# dot vs matmul for higher dims:
# np.dot(a, b) contracts last axis of a with second-to-last of b
# np.matmul(a, b) treats as stack of matrices, broadcasts batch dims

# Einsum — Einstein summation (most flexible)
C = np.einsum('ij,jk->ik', A, B)             # matmul
trace = np.einsum('ii->', np.eye(4))          # trace
batched = np.einsum('bij,bjk->bik', X, Y)    # batch matmul
diag = np.einsum('ii->i', M)                 # extract diagonal
outer = np.einsum('i,j->ij', x, y)           # outer product
hadamard = np.einsum('ij,ij->ij', A, A)      # element-wise (pointless but shows syntax)

# Common decompositions
U, S, Vt = np.linalg.svd(A, full_matrices=False)   # economy SVD
Q, R = np.linalg.qr(A)
eigvals, eigvecs = np.linalg.eigh(symmetric_matrix)  # use eigh for symmetric
L = np.linalg.cholesky(pos_def_matrix)

# Solve linear system Ax = b (prefer over inv)
x = np.linalg.solve(A_square, b)       # NEVER do np.linalg.inv(A) @ b
x_lstsq, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

# Norms
np.linalg.norm(v)               # L2 (vector)
np.linalg.norm(A, 'fro')       # Frobenius (matrix)
np.linalg.norm(A, axis=1)      # row-wise L2 norms
```

---

## Random Generation (New Generator API)

The legacy `np.random.rand/randn/randint` uses global state. Prefer the new Generator API (NumPy >= 1.17):

```python
rng = np.random.default_rng(seed=42)   # PCG64 generator, reproducible

# Uniform
rng.random((3, 4))                     # [0, 1) floats
rng.uniform(low=2, high=10, size=(5,)) # [2, 10)
rng.integers(0, 100, size=(10,))       # [0, 100) integers

# Normal
rng.standard_normal((3, 4))            # N(0, 1)
rng.normal(loc=5, scale=2, size=(100,))

# Other distributions
rng.choice(array, size=10, replace=False)  # sampling without replacement
rng.permutation(n)                          # random permutation of [0, n)
rng.shuffle(array)                          # in-place shuffle (no return)
rng.exponential(scale=1.0, size=(5,))
rng.poisson(lam=5, size=(100,))
rng.multinomial(n=20, pvals=[0.2, 0.3, 0.5])
rng.multivariate_normal(mean, cov, size=100)

# Spawn independent generators for parallel work
child_rng1, child_rng2 = rng.spawn(2)
```

---

## Advanced Indexing

### Fancy Indexing (integer array)

```python
a = np.arange(20).reshape(4, 5)

# Select specific rows
rows = a[[0, 2, 3]]                    # shape (3, 5)

# Select specific elements (paired indices)
vals = a[[0, 1, 2], [1, 3, 4]]        # a[0,1], a[1,3], a[2,4] → shape (3,)

# Assign via fancy indexing
a[[0, 2], :] = 99

# np.take / np.put for explicit axis control
np.take(a, [0, 2], axis=0)            # same as a[[0,2], :]
```

### Boolean Indexing

```python
mask = a > 10                           # boolean array same shape as a
filtered = a[mask]                      # 1D array of matching elements
a[mask] = 0                            # set all >10 to 0

# Combine conditions (use & | ~, NOT and/or/not)
mask = (a > 5) & (a < 15)
mask = ~np.isnan(data)

# np.where — ternary selection
result = np.where(a > 10, a, -1)       # keep if >10, else -1
indices = np.where(a > 10)             # tuple of index arrays

# np.argwhere — coordinates of True elements
coords = np.argwhere(a > 10)           # shape (N, 2) for 2D array
```

### Index Manipulation

```python
# Argsort / argpartition
idx = np.argsort(a, axis=1)            # indices that sort each row
top_k_idx = np.argpartition(a.ravel(), -5)[-5:]  # top-5 (unsorted)

# np.ix_ — open mesh for cross-indexing
rows = np.array([0, 2])
cols = np.array([1, 3, 4])
submatrix = a[np.ix_(rows, cols)]      # (2, 3) submatrix

# np.unravel_index — flat index to multi-dim
flat_idx = np.argmax(a)
row, col = np.unravel_index(flat_idx, a.shape)
```

---

## Ufunc Patterns

Universal functions operate element-wise with broadcasting and support reduce/accumulate/outer:

```python
# Reduce along axis
np.add.reduce(a, axis=0)        # same as a.sum(axis=0)
np.multiply.reduce(a, axis=1)   # row products
np.maximum.reduce(a, axis=0)    # column-wise max

# Accumulate (running totals)
np.add.accumulate(a, axis=0)    # cumulative sum along axis 0
np.multiply.accumulate(a)       # running product

# Outer (all pairs)
np.multiply.outer(x, y)         # outer product
np.subtract.outer(x, y)         # pairwise differences

# at — unbuffered in-place operation (handles repeated indices)
np.add.at(a, [0, 0, 1], 1)     # a[0] += 2, a[1] += 1 (not a[[0,0,1]] += 1)

# Custom ufuncs via np.frompyfunc or np.vectorize (slow, avoid in hot paths)
```

---

## Memory Layout (C vs F Order)

```python
# C order (row-major, default): last axis varies fastest in memory
c_arr = np.zeros((3, 4), order='C')    # row 0 contiguous, then row 1...
c_arr.strides   # (32, 8) for float64 — 4*8=32 bytes per row step

# F order (column-major, Fortran): first axis varies fastest
f_arr = np.zeros((3, 4), order='F')
f_arr.strides   # (8, 24) — 3*8=24 bytes per column step

# Why it matters: iterate along contiguous axis for cache efficiency
# C-order: iterating over last axis (columns) is fast
# F-order: iterating over first axis (rows) is fast

# Check layout
arr.flags['C_CONTIGUOUS']   # True if C-contiguous
arr.flags['F_CONTIGUOUS']   # True if F-contiguous

# Force contiguous copy
c_copy = np.ascontiguousarray(arr)     # force C
f_copy = np.asfortranarray(arr)        # force F

# Views can be non-contiguous (e.g., slicing with step, transposing)
transposed = arr.T                      # view, F-contiguous if arr is C
sliced = arr[::2]                       # non-contiguous (stride doubled)

# Performance tip: reshape/ravel respect order kwarg
arr.ravel('C')   # C-order flattening
arr.ravel('F')   # F-order flattening (different element sequence)
```

---

## Vectorization Patterns (Replacing Loops)

```python
# ANTI-PATTERN: Python loop
result = np.empty(len(data))
for i in range(len(data)):
    result[i] = data[i] ** 2 + 3 * data[i]

# VECTORIZED:
result = data ** 2 + 3 * data

# Conditional logic — np.where instead of if/else loop
# Loop:
for i in range(len(x)):
    out[i] = x[i] if x[i] > 0 else 0
# Vectorized:
out = np.where(x > 0, x, 0)
# Or: np.maximum(x, 0)  — ReLU

# Accumulation with condition — np.cumsum + masking
counts = np.cumsum(mask.astype(int))

# Binning — np.digitize + np.bincount
bins = np.digitize(values, bin_edges)          # which bin each value falls in
counts = np.bincount(bins, minlength=n_bins)   # count per bin
sums = np.bincount(bins, weights=values)       # sum per bin

# Group-by reduction without loops
labels = np.array([0, 1, 0, 1, 2, 0])
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
group_sums = np.bincount(labels, weights=data)  # [10., 6., 5.]

# Sliding window — np.lib.stride_tricks.sliding_window_view (NumPy >= 1.20)
from numpy.lib.stride_tricks import sliding_window_view
windows = sliding_window_view(signal, window_shape=5)  # (N-4, 5)
moving_avg = windows.mean(axis=1)

# Batch distance calculation (avoid nested loops)
# Instead of: for i: for j: dist[i,j] = norm(A[i] - B[j])
diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # (M, N, D)
dists = np.sqrt((diff ** 2).sum(axis=-1))          # (M, N)
# Or einsum:
dists_sq = np.einsum('id,id->i', A, A)[:, None] + \
           np.einsum('jd,jd->j', B, B)[None, :] - \
           2 * A @ B.T
```

---

## Common Shape Mismatches & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `could not broadcast (3,4) with (3,)` | Trailing dims don't match | Reshape `(3,)` to `(3,1)` or `(1,3)` depending on intent |
| `matmul: Input operand 1 has mismatch` | Inner dims don't align | Check shapes: `(m,k) @ (k,n)` required |
| `cannot reshape (12,) to (3,5)` | Element count mismatch | Verify `prod(old_shape) == prod(new_shape)` |
| `indexing with boolean of wrong shape` | Mask shape ≠ array shape | Ensure mask was derived from same array or same shape |
| `setting array element with sequence` | Ragged nested list | Use `dtype=object` or pad to uniform shape |

```python
# Fix: 1D vector won't broadcast with 2D matrix along axis 0
weights = np.array([1, 2, 3])        # (3,)
data = np.random.randn(3, 10)        # (3, 10)
# FAILS: data * weights → (3,10) * (3,) tries to align last dims: 10 vs 3
# FIX:
data * weights[:, np.newaxis]         # (3,10) * (3,1) → (3, 10) ✓

# Fix: concatenate requires matching dims on non-concat axis
a = np.zeros((3, 4))
b = np.zeros((3, 5))
# np.vstack([a, b]) → ERROR (axis 1: 4 vs 5)
# np.hstack([a, b]) → (3, 9) ✓

# Fix: dot product of matrices vs vectors
A = np.random.randn(3, 3)
v = np.random.randn(3)
# A @ v → (3,) ✓ (matrix-vector product)
# v @ A → (3,) ✓ (row-vector times matrix)
# v @ v → scalar ✓ (dot product)
```

---

## Structured Arrays

Typed records — useful for heterogeneous columnar data without pandas:

```python
# Define dtype
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('weight', 'f8')])

# Create
people = np.array([('Alice', 30, 55.0), ('Bob', 25, 80.5)], dtype=dt)

# Access fields by name
people['age']         # array([30, 25])
people['name'][0]     # 'Alice'

# Boolean filter
young = people[people['age'] < 28]

# Record arrays (attribute access)
rec = people.view(np.recarray)
rec.age              # array([30, 25])

# From file
data = np.genfromtxt('data.csv', delimiter=',', dtype=dt, skip_header=1)

# Nested dtypes
nested_dt = np.dtype([('pos', [('x', 'f4'), ('y', 'f4')]), ('label', 'i4')])
```

---

## Performance Tips

1. **Avoid copies**: Use views (slicing, reshape, transpose) when possible. Check with `np.shares_memory(a, b)`.
2. **Pre-allocate**: `out = np.empty(shape)` then fill, vs growing lists.
3. **Use `out=` parameter**: `np.add(a, b, out=result)` avoids temp allocation.
4. **Prefer in-place ops**: `a += b` vs `a = a + b` (latter allocates new array).
5. **dtype matters**: Use `float32` for large arrays when precision allows. `int8/int16` for indices.
6. **Avoid fancy indexing in hot loops**: It always copies. Slice views are free.
7. **np.einsum with `optimize=True`**: For complex contractions, enables BLAS paths.

```python
# In-place normalization (no temporary arrays)
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
np.divide(vectors, norms, out=vectors)

# Efficient batch operations with einsum
# Batch matrix-vector: (B, M, N) @ (B, N) -> (B, M)
result = np.einsum('bmn,bn->bm', matrices, vectors, optimize=True)
```

## When to Use

| ✅ Use NumPy | ❌ Don't Use |
|---|---|
| Numerical computing, matrix operations | Labeled data with columns (use pandas/polars) |
| Custom algorithm implementation | GPU needed (use PyTorch/JAX) |
| Performance-critical vectorized inner loops | High-level ML training (use frameworks) |
| Preprocessing before feeding to models | When broadcasting rules confuse the team |
| Scientific computing, linear algebra | Distributed computing (use Dask/Ray) |

**Decision rule**: Foundation layer for everything. Use directly for math/algorithms. Wrap with pandas for labeled data, PyTorch for GPU/autograd.

---

## References

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [NumPy GitHub](https://github.com/numpy/numpy)