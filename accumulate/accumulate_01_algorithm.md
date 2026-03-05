# accumulate! / cumsum / cumprod   Algorithm Deep Dive

## What Is a Prefix Scan?

`accumulate!` computes a *prefix scan*: given operator `⊕` and array `A`, it produces output `B` where each element is the reduction of all preceding elements:

- **Inclusive:** `B[i] = A[1] ⊕ A[2] ⊕ … ⊕ A[i]`
- **Exclusive:** `B[i] = A[1] ⊕ … ⊕ A[i-1]`, with `B[1]` = identity element

`cumsum` and `cumprod` are special cases with `⊕ = +` and `⊕ = *` respectively.

---

## Why Sequential Is Insufficient

The naive algorithm:
```
B[1] = A[1]
for i in 2:n
    B[i] = B[i-1] ⊕ A[i]
```
is **inherently sequential**   each step depends on the previous result. On GPU with thousands of threads, this bottleneck is catastrophic. Direct parallelisation is non-trivial.

---

## Blelloch Work-Efficient Parallel Scan (1990)

Blelloch decouples the sequential dependency by splitting into two tree-structured phases, each of depth `log₂ n`.

### Phase 1: Up-Sweep (Reduce)

Build a reduction tree bottom-up. At each level `d = 0, 1, …, log₂(n)-1`, threads at active positions compute:

```
stride = 2^d
for each active index i (step 2^(d+1), from right):
    A[i] = A[i - stride] ⊕ A[i]
```

After up-sweep: `A[n-1]` holds the total reduction of the entire array (the "root").

### Phase 2: Down-Sweep (Propagate)

Set `A[n-1] = identity`, then propagate back down:

```
stride = 2^d  (d from log₂(n)-1 down to 0)
for each active index i:
    temp         = A[i - stride]
    A[i - stride] = A[i]           ← left child gets parent
    A[i]         = temp ⊕ A[i]    ← right child gets left ⊕ parent
```

After down-sweep: `A[i]` = exclusive prefix sum at position `i`.  
Inclusive result: add `A_original[i]` to each output.

### Complexity

| | Sequential | Blelloch |
|---|---|---|
| Work | O(n) | O(n)   optimal |
| Depth | O(n) | O(log n)   parallel |

---

## Worked Example: Prefix Sum of `[3, 1, 7, 0, 4, 1, 6, 3]`

```
Input:             [  3,  1,  7,  0,  4,  1,  6,  3 ]

Up-sweep:
  stride=1:        [  3,  4,  7,  7,  4,  5,  6,  9 ]
  stride=2:        [  3,  4,  7, 11,  4,  5,  6, 14 ]
  stride=4:        [  3,  4,  7, 11,  4,  5,  6, 25 ]   ← root = 25

Down-sweep (set root=0):
  stride=4:        [  3,  4,  7,  0,  4,  5,  6, 11 ]
  stride=2:        [  3,  4,  0,  7,  4, 11,  6, 15 ]
  stride=1:        [  0,  3,  4,  7, 11, 11, 15, 16 ]

Exclusive output:  [  0,  3,  4,  7, 11, 11, 15, 16 ]
Inclusive output:  [  3,  4, 11, 11, 15, 16, 22, 25 ]
```

---

## AcceleratedKernels.jl Variants

`AcceleratedKernels.jl` ships two scan variants:

| Variant | Algorithm | Requirement |
|---------|-----------|-------------|
| Standard | Blelloch two-phase scan | Any backend |
| `DecoupledLookback` | Single-pass with inter-block communication | Requires `memory_order_acq_rel` atomics |

Metal cannot use `DecoupledLookback`   its shader model does not expose `memory_order_acq_rel` to compute kernels. Metal.jl implements Blelloch from scratch instead.
