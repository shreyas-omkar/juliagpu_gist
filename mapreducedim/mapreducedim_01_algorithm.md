# mapreducedim!   Algorithm: Tiled Shared-Memory Tree Reduction

## What mapreducedim! Does

`mapreducedim!(f, op, R, A)` applies map function `f` to each element of `A`, then reduces the results along specified dimensions using binary operator `op`, writing into pre-allocated output array `R`.

It is the kernel beneath **every high-level reduction in Julia**:

| High-level call | Equivalent |
|-----------------|-----------|
| `sum(A)` | `mapreducedim!(identity, +, R, A)` |
| `sum(abs2, A)` | `mapreducedim!(abs2, +, R, A)` |
| `maximum(A; dims=2)` | `mapreducedim!(identity, max, R, A; dims=2)` |
| `any(A)` | `mapreducedim!(identity, |, R, A)` |
| `all(A)` | `mapreducedim!(identity, &, R, A)` |

---

## Why Sequential Reduction Fails on GPU

The naive sequential algorithm is strictly serial   each step depends on the previous:

```
acc = init
for i in 1:n
    acc = op(acc, f(A[i]))
end
R[out] = acc
```

Depth: O(n). On GPU with thousands of threads available, this wastes all parallelism.

---

## Tiled Shared-Memory Tree Reduction

The GPU strategy: assign one **threadgroup** per output element. Threads within the group collaborate via shared memory to reduce their slice of the input.

### Phase 1: Load into shared memory

Each thread loads one (or more) input elements into a fast shared memory buffer (L1-cache speed, ~100× faster than global memory).

```
shared[tid] = f(A[input_slice_index])
barrier()   ← all threads must finish loading before reduction starts
```

### Phase 2: Tree reduction

Reduce the shared buffer in log₂(T) synchronised steps, where T is the threadgroup size:

```
stride = T ÷ 2
while stride > 0:
    if tid ≤ stride:
        shared[tid] = op(shared[tid], shared[tid + stride])
    barrier()
    stride ÷= 2
```

At each step, half the threads are idle. The active threads combine pairs. After log₂(T) steps, `shared[1]` holds the reduction of all T elements.

### Phase 3: Write output

Thread 1 (or thread 0 in 0-indexed backends) writes the result:
```
if tid == 1:
    R[out_idx] = op(R[out_idx], shared[1])
```

---

## Complexity

| Metric | Value |
|--------|-------|
| Work | O(n)   optimal, matches sequential |
| Depth | O(log T)   T = threadgroup size, typically 256 |
| Synchronisations | log₂(T) barriers (intra-group only) |
| Global memory passes | ~2 (read input, write output) |

---

## Worked Example: Sum of 8 elements, one threadgroup

```
Input (global memory):  [ 3  7  2  5  1  8  4  6 ]

Step 0: Load into shared memory (all 8 threads)
  shared: [ 3  7  2  5  1  8  4  6 ]
  barrier()

Step 1: stride=4, threads 1-4 active
  shared[1] = 3 + 1 = 4
  shared[2] = 7 + 8 = 15
  shared[3] = 2 + 4 = 6
  shared[4] = 5 + 6 = 11
  shared: [ 4  15  6  11  1  8  4  6 ]   ← upper half untouched
  barrier()

Step 2: stride=2, threads 1-2 active
  shared[1] = 4  + 6  = 10
  shared[2] = 15 + 11 = 26
  barrier()

Step 3: stride=1, thread 1 active
  shared[1] = 10 + 26 = 36
  barrier()

Output:  R[out_idx] = 36   ← correct: 3+7+2+5+1+8+4+6 = 36
```

3 steps (log₂(8) = 3) vs 7 sequential additions. At T=256: 8 steps vs 255.

---

## Multi-Block Reduction

When the reduction slice is larger than one threadgroup (n > T), multiple blocks are launched per output element. Each block reduces T elements into one partial result. A second pass (or atomic operation) combines partial results.

This is why the `n=10^6` test case matters   it exercises the multi-block path.

---

## The Map Step

Function `f` is applied **before** accumulation into shared memory:

```
shared[tid] = f(A[input_idx])    ← fuse map into load
```

This fusion avoids allocating a temporary `f.(A)` array, saving one full global memory pass. For `sum(abs2, A)` this means `abs2` is computed in registers, not written to GPU memory.

---

## Neutral Elements

The shared memory must be initialised for out-of-bounds threads (when n is not a multiple of T). The initialisation value is the **neutral element** (identity) of the operator:

| Operator | Neutral element |
|----------|----------------|
| `+` | `zero(T)` |
| `*` | `one(T)` |
| `max` | `typemin(T)` |
| `min` | `typemax(T)` |
| `\|` (any) | `false` |
| `&` (all) | `true` |

GPUArrays already provides `neutral_element(op, T)` for all standard operators   the kernel uses this directly.
